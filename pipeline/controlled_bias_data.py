"""Pinned controller inputs and fitting-only sampling for the bias diagnostic."""

import json
from pathlib import Path

import torch
import torch.nn.functional as functional

from pipeline.runtime import file_sha256

CONTROLLER_COMMIT = "451349aa463cd5b7ec37ea3a97fd99b8bf50a5d2"
CONTROLLER_ARCHIVE_SHA256 = "c57f28b986321607270da729e2184ae6c9eab2d4808a642b49ea8b3338aac662"
CONTROLLER_REPORT_SHA256 = "f1e7e37d45367b8d81c0a21e2cdf9ced07dc0cde927bc218dcf99b9cc5164459"
CONTROLLER_GEOMETRY_SHA256 = "32d56c1fcf4b3c449f8d42ce84a440e507be2cd9ad333c1be3130be4b937a388"
CONTROLLER_ASSET_ID = 766429438
EXPECTED_CLASSES = list(range(34))
EXPECTED_COUNTS = {"eligible": 6992, "reference": 1372, "unbiased": 5620, "development": 1730}


def load_controller(root):
    """Read only the pinned saved seed-0 geometry, without replaying clustering."""
    root = Path(root)
    report = root / "diagnostics/seed0/controls.json"
    geometry = root / "diagnostics/seed0/geometry.pt"
    if not report.is_file() or file_sha256(report) != CONTROLLER_REPORT_SHA256:
        raise ValueError("Pinned controller report is missing or changed")
    saved = json.loads(report.read_text())
    if (saved.get("geometry_sha256") != CONTROLLER_GEOMETRY_SHA256
            or not geometry.is_file() or file_sha256(geometry) != CONTROLLER_GEOMETRY_SHA256):
        raise ValueError("Pinned controller geometry is missing or changed")
    return torch.load(geometry, map_location="cpu", weights_only=True), {
        "results_commit": CONTROLLER_COMMIT, "archive_sha256": CONTROLLER_ARCHIVE_SHA256,
        "report_sha256": CONTROLLER_REPORT_SHA256, "geometry_sha256": CONTROLLER_GEOMETRY_SHA256,
    }


def validate_cache_alignment(bundles):
    """All masks are cache-row positions shared by the three representations."""
    first = bundles[0]
    for seed, bundle in bundles.items():
        for name in ("labels", "development_labels"):
            if not torch.equal(bundle[name], first[name]):
                raise ValueError(f"Cache label order differs for seed {seed}")
        for name in ("fitting_sha256", "development_sha256"):
            if bundle["inputs"][name] != first["inputs"][name]:
                raise ValueError(f"Cache split identity differs for seed {seed}")


def prepare_sampling(labels, original_counts, geometry, config):
    """Select R/U/T using fitting labels and the controller, never development data."""
    if len(geometry) != len(original_counts):
        raise ValueError("Controller must cover every original class")
    groups = torch.full_like(labels, -1)
    eligible = []
    eligibility = []
    for c, original_count in enumerate(original_counts):
        rows = torch.where(labels == c)[0]
        current = geometry[c]
        assignments, centers = current["assignments"], current["centers"]
        sizes = None
        if assignments is not None:
            if (current["class_index"] != c or assignments.dtype != torch.int64
                    or assignments.shape != (len(rows),) or (assignments < 0).any()
                    or (assignments > 1).any() or centers is None
                    or centers.shape != (2, 64) or not torch.isfinite(centers).all()):
                raise ValueError(f"Invalid controller geometry for class {c}")
            sizes = torch.bincount(assignments, minlength=2)
        is_eligible = original_count >= 100 and sizes is not None and bool(
            (sizes >= config["minimum_controller_size"]).all())
        eligibility.append({"original_class": c, "original_count": int(original_count),
                            "cluster_sizes": sizes.tolist() if sizes is not None else None,
                            "is_eligible": is_eligible})
        if is_eligible:
            eligible.append(c)
            groups[rows] = assignments
    if eligible != EXPECTED_CLASSES:
        raise ValueError(f"Controller eligibility changed: expected {EXPECTED_CLASSES}, got {eligible}")
    generators = {name: torch.Generator().manual_seed(config[f"{name}_seed"])
                  for name in ("reference", "retention", "biased_group")}
    references, unbiased, retained = [], [], []
    probabilities = torch.zeros(len(labels), dtype=torch.float64)
    biased_groups, classes = [], []
    for c in eligible:
        group_rows = []
        counts = []
        for g in (0, 1):
            rows = torch.where((labels == c) & (groups == g))[0]
            reference_count = int(config["reference_fraction"] * len(rows))
            permutation = torch.randperm(len(rows), generator=generators["reference"])
            references.append(rows[permutation[:reference_count]])
            remaining = rows[permutation[reference_count:]].sort().values
            group_rows.append(remaining)
            unbiased.append(remaining)
            counts.append({"controller_group": g, "original": len(rows),
                           "reference": reference_count, "unbiased": len(remaining)})
        biased_group = int(torch.randint(2, (), generator=generators["biased_group"]))
        biased_groups.append(biased_group)
        for g, rows in enumerate(group_rows):
            keep_count = ((len(rows) + config["retention_divisor"] - 1) // config["retention_divisor"]
                          if g == biased_group else len(rows))
            selected = rows
            if g == biased_group:
                selected = rows[torch.randperm(len(rows), generator=generators["retention"])[:keep_count]]
            retained.append(selected)
            probability = keep_count / len(rows)
            probabilities[selected] = probability
            counts[g].update(retained=keep_count, inclusion_probability=probability)
        classes.append({"original_class": c, "biased_group": biased_group, "groups": counts})
    reference = torch.cat(references).sort().values
    unbiased = torch.cat(unbiased).sort().values
    retained = torch.cat(retained).sort().values
    eligible_rows = torch.where(groups >= 0)[0]
    removed = unbiased[~torch.isin(unbiased, retained)]
    cover = torch.cat((reference, retained, removed)).sort().values
    if not torch.equal(cover, eligible_rows) or len(torch.unique(cover)) != len(cover):
        raise RuntimeError("Reference, retained and removed positions must partition fitting rows")
    label_map = torch.full((len(original_counts),), -1, dtype=torch.int64)
    label_map[eligible] = torch.arange(len(eligible))
    arrays = {
        "eligible": eligible_rows, "reference": reference, "unbiased": unbiased,
        "retained": retained, "removed": removed, "controller_groups": groups,
        "inclusion_probabilities": probabilities[retained], "biased_groups": torch.tensor(biased_groups),
        "eligible_original_classes": torch.tensor(eligible), "label_map": label_map,
    }
    return arrays, {"eligibility": eligibility, "classes": classes,
                    "counts": {name: len(arrays[name]) for name in ("eligible", "reference", "unbiased", "retained", "removed")},
                    "seeds": {name: config[f"{name}_seed"] for name in generators}}


def development_context(bundle, geometry, sampling):
    """Reporter-only groups: always seed-0 development coordinates and centers."""
    labels = bundle["development_labels"]
    positions = torch.where(sampling["label_map"][labels] >= 0)[0]
    selected_labels = labels[positions]
    groups = torch.empty(len(positions), dtype=torch.int64)
    for c in sampling["eligible_original_classes"].tolist():
        selected = selected_labels == c
        features = functional.normalize(bundle["development_features"][positions[selected]], dim=1)
        groups[selected] = (features @ geometry[c]["centers"].T).argmax(dim=1)
    return {"positions": positions, "labels": sampling["label_map"][selected_labels], "controller_groups": groups}
