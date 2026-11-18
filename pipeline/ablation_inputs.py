"""Validate the published fitting-only audit inputs without loading images."""

import json
from pathlib import Path

import torch

from pipeline.methods import baseline, prototype_weights
from pipeline.runtime import file_sha256

AUDIT_COMMIT = "51c7829d505143bd556ff1f3038b6559751e3428"
ARCHIVE_SHA256 = "e157e664288bd849a0dc5d6fb5eed2f61269a3452327d9a1407145a02331aeb2"
AUDIT_REPORT_HASHES = {
    "weight_audit.json": "e3b32935fd8749a71437a8f7d359f4850134f3ffb0f6394977f49503c555fadf",
    "weight_artifacts.json": "fd002a8478ab3f72df558d29b1914978c783b9029193b1bd71cfe78e869313e2",
    "study_config.json": "69e4eac1ca6eea210d5dfc1db44ca37993e090bf3953ed4865e98d65febe4f41",
    "implementation.json": "0669ec16c96773672114109e6a65a16ac0a7982d2a937a4399947aec8b2e03a8",
    "split/split.json": "940b63029513b2b6a940ee31b40e94f27adc797a68fc0a7bfa7ab8d8ec93ba00",
}


def _read(path):
    return json.loads(Path(path).read_text())


def _counts(values, name):
    if (not isinstance(values, list) or len(values) != 100
            or any(type(n) is not int or n < 1 for n in values)):
        raise ValueError(f"Invalid {name} class counts")
    return torch.tensor(values, dtype=torch.int64)


def _features(bundle, prefix, counts, dtype):
    features, labels = bundle[f"{prefix}features"], bundle[f"{prefix}labels"]
    if (not isinstance(features, torch.Tensor) or features.shape != (int(counts.sum()), 64)
            or features.dtype != dtype or not torch.isfinite(features).all()
            or not isinstance(labels, torch.Tensor) or labels.dtype != torch.int64
            or labels.shape != (len(features),) or (labels < 0).any() or (labels >= 100).any()
            or not torch.equal(torch.bincount(labels, minlength=100), counts)):
        raise ValueError(f"Invalid {prefix or 'fitting_'}feature cache or label counts")


def load_audit_inputs(root, config):
    """Check every seed before returning any training inputs; paths are not identity."""
    root = Path(root)
    if config["input_results_commit"] != AUDIT_COMMIT or config["archive_sha256"] != ARCHIVE_SHA256:
        raise ValueError("The assignment study requires the pinned rebuilt audit inputs")
    reports = {}
    for name, digest in AUDIT_REPORT_HASHES.items():
        path = root / "audit" / name
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"Published audit report is missing or changed: {name}")
        reports[name] = _read(path)
    audit, artifacts = reports["weight_audit.json"], reports["weight_artifacts.json"]
    split, study = reports["split/split.json"], reports["study_config.json"]
    if artifacts["sha256"] != ARCHIVE_SHA256 or study["split_seed"] != 43 or split["split_seed"] != 43:
        raise ValueError("Audit archive or split identity does not match the protocol")
    if study["seeds"] != [0, 1, 2] or set(audit["feature_caches"]) != {"0", "1", "2"}:
        raise ValueError("The published audit must contain all three baseline seeds")
    original = _counts(split["original_class_counts"], "original")
    fitting = _counts(split["fitting_class_counts"], "fitting")
    development = _counts(split["development_class_counts"], "development")
    if not torch.equal(fitting + development, original):
        raise ValueError("Fitting/development counts do not cover the original pool")
    if not torch.equal(development, torch.clamp(original // 5, min=1)):
        raise ValueError("Development counts do not follow the fixed split protocol")
    all_positions = []
    for name, counts in (("fitting", fitting), ("development", development)):
        positions = split[f"{name}_positions"]
        if (not isinstance(positions, list) or len(positions) != int(counts.sum())
                or any(type(i) is not int for i in positions)
                or positions != sorted(set(positions))):
            raise ValueError(f"Invalid ordered {name} positions")
        all_positions.extend(positions)
    if sorted(all_positions) != list(range(int(original.sum()))):
        raise ValueError("Split positions overlap or do not cover the original pool")
    bundles, checkpoints, hashes = {}, {}, {}
    for seed in config["seeds"]:
        checkpoint_path = root / f"baselines/seed{seed}/final.pt"
        cache_path = root / f"caches/seed{seed}/features_weights.pt"
        if not cache_path.is_file() or file_sha256(cache_path) != audit["feature_caches"][str(seed)]:
            raise ValueError(f"Missing or changed feature cache for seed {seed}")
        bundle = torch.load(cache_path, map_location="cpu", weights_only=True)
        identity = bundle["inputs"]
        if not checkpoint_path.is_file() or file_sha256(checkpoint_path) != identity["baseline_sha256"]:
            raise ValueError(f"Missing or incompatible baseline checkpoint for seed {seed}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        source = checkpoint["config"]
        settings = {"imbalance_factor": 0.01, "data_seed": 42}
        if (source.get("method") != "baseline" or source.get("seed") != seed
                or baseline.prepare_method_options(source["method_options"]) != study["baseline"]
                or study["baseline"]["epochs"] != 200
                or checkpoint["data_settings"] != settings
                or not torch.equal(checkpoint["class_counts"], fitting)
                or identity["seed"] != seed
                or identity["fitting_sha256"] != split["file_hashes"]["fitting"]
                or identity["development_sha256"] != split["file_hashes"]["development"]
                or identity["prototype_options"] != prototype_weights.prepare_options({"weighting": "sinkhorn"})):
            raise ValueError(f"Baseline/cache recipe or split mismatch for seed {seed}")
        dtype = checkpoint["model_state_dict"]["fc.weight"].dtype
        _features(bundle, "", fitting, dtype)
        _features(bundle, "development_", development, dtype)
        weights = bundle["ot_weights"]
        if (not isinstance(weights, torch.Tensor) or weights.shape != bundle["labels"].shape
                or weights.dtype != dtype or not torch.isfinite(weights).all() or (weights <= 0).any()):
            raise ValueError("Invalid original OT weights")
        diagnostics = bundle["prototype_diagnostics"]
        if (not isinstance(diagnostics, list) or len(diagnostics) != 100
                or [d["class_index"] for d in diagnostics] != list(range(100))):
            raise ValueError("Expected ordered diagnostics for all 100 classes")
        for c, diagnostic in enumerate(diagnostics):
            values = weights[bundle["labels"] == c]
            if not torch.isclose(values.double().sum(), torch.tensor(0.01, dtype=torch.float64),
                                 rtol=1e-5, atol=1e-8):
                raise ValueError(f"Invalid OT class mass for class {c}")
            active = diagnostic["used_ot"]
            if type(active) is not bool or (active and diagnostic["fallback_reason"] is not None):
                raise ValueError("Invalid OT active/fallback diagnostic")
            if not active and (not diagnostic["fallback_reason"] or not torch.equal(values, values[0].expand_as(values))):
                raise ValueError("Fallback classes must have a reason and uniform weights")
        bundles[seed], checkpoints[seed] = bundle, checkpoint
        hashes[str(seed)] = {"baseline": file_sha256(checkpoint_path), "cache": file_sha256(cache_path)}
    provenance = {
        "input_results_commit": AUDIT_COMMIT, "archive": artifacts,
        "report_hashes": AUDIT_REPORT_HASHES.copy(), "seed_files": hashes,
        "split_sha256": AUDIT_REPORT_HASHES["split/split.json"],
        "validation": "passed", "image_identity": "Trusted published extraction; images are not reloaded.",
    }
    return {"bundles": bundles, "checkpoints": checkpoints, "split": split, "provenance": provenance}
