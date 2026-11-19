"""Stage B rules on the original fitting-only prototype geometry."""

import math

import torch
import torch.nn.functional as functional

from pipeline.ablation_weights import _midranks, reassign_by_priority, transform_weights
from pipeline.methods import prototype_weights

CONDITIONS = (
    "cluster_balanced", "prototype_distance", "easy_rank",
    "cluster_rank_strong", "distance_rank_strong",
)
NATIVE_CONDITIONS = CONDITIONS[:3]


def replay_geometry(bundle, seed):
    """Replay the original RNG sequence and solver before accepting any geometry."""
    geometry = []
    replayed, diagnostics = prototype_weights.build_weights(
        bundle["features"], bundle["labels"],
        {"seed": seed, "method_options": bundle["inputs"]["prototype_options"]},
        torch.device("cpu"), geometry=geometry,
    )
    for original, current in zip(bundle["prototype_diagnostics"], diagnostics):
        for key in ("class_index", "used_ot", "cluster_sizes", "fallback_reason"):
            if original[key] != current[key]:
                raise ValueError(f"Prototype geometry replay mismatch: class {current['class_index']} {key}")
    if not torch.allclose(replayed, bundle["ot_weights"], rtol=1e-5, atol=1e-8):
        raise ValueError("Replayed original OT weights disagree with the published cache")
    return geometry


def build_controls(bundle, geometry, scores, strong):
    """Keep original fallbacks uniform; native rules never solve transport."""
    labels = bundle["labels"]
    if not torch.equal(strong, transform_weights(bundle["ot_weights"], labels, "strong_ot")):
        raise ValueError("Parent strong weights disagree with the pinned exponent-3 rule")
    margins, tie_order = scores["margins"], scores["tie_order"]
    if margins.shape != labels.shape or not torch.isfinite(margins).all() or len(geometry) != 100:
        raise ValueError("Invalid saved margins or prototype geometry")
    weights = {name: strong.clone() for name in NATIVE_CONDITIONS}
    priorities = {name: torch.zeros_like(margins, dtype=torch.float64)
                  for name in ("cluster_rank_strong", "distance_rank_strong")}
    classes = []
    for c, diagnostic in enumerate(bundle["prototype_diagnostics"]):
        selected = labels == c
        record = {"class_index": c, "original_ot_active": diagnostic["used_ot"]}
        if diagnostic["used_ot"]:
            current = geometry[c]
            assignments, centers = current["assignments"], current["centers"]
            count = int(selected.sum())
            if (current["class_index"] != c or assignments is None or centers is None
                    or assignments.shape != (count,) or assignments.dtype != torch.int64
                    or (assignments < 0).any() or (assignments > 1).any()
                    or centers.shape != (2, bundle["features"].shape[1])
                    or not torch.isfinite(centers).all()):
                raise ValueError(f"Invalid prototype geometry for class {c}")
            sizes = torch.bincount(assignments, minlength=2)
            if sizes.tolist() != diagnostic["cluster_sizes"] or (sizes < 5).any():
                raise ValueError(f"Changed prototype cluster sizes for class {c}")
            features = functional.normalize(bundle["features"][selected], dim=1)
            distance = (1 - features @ centers.T).clamp_min(0).min(dim=1).values.double()
            inverse_size = 1 / sizes[assignments].double()
            weights["cluster_balanced"][selected] = (inverse_size / 200).to(strong.dtype)
            ranks = _midranks(margins[selected].double())
            scaled = (ranks - ranks.min()) / (ranks.max() - ranks.min()) if ranks.max() > ranks.min() else ranks * 0
            for name, logits in (("prototype_distance", -distance / 0.1),
                                 ("easy_rank", math.log(4) * scaled)):
                values = torch.softmax(logits, dim=0) / 100
                weights[name][selected] = values.to(strong.dtype)
            for name, priority in (("cluster_rank_strong", inverse_size),
                                   ("distance_rank_strong", -distance)):
                priorities[name][selected] = priority
                unique_count = len(torch.unique(priority))
                record[name] = {"unique_priorities": unique_count,
                                "has_ties": unique_count < count, "is_constant": unique_count == 1}
        classes.append(record)
    for name, priority in priorities.items():
        weights[name] = reassign_by_priority(strong, labels, priority, tie_order)
    for name, values in weights.items():
        transform_weights(values, labels, "ot")  # Validate finite, positive, fixed class masses.
        for c, diagnostic in enumerate(bundle["prototype_diagnostics"]):
            selected = labels == c
            if not diagnostic["used_ot"] and not torch.equal(values[selected], strong[selected]):
                raise RuntimeError(f"Fallback weights changed: {name}, class {c}")
            if name in priorities:
                positions = tie_order[labels[tie_order] == c]
                ordered = positions[torch.argsort(priorities[name][positions], stable=True)]
                if (values[ordered].diff() < 0).any():
                    raise RuntimeError(f"Priority direction changed: {name}, class {c}")
    counts = {name: {
        "active_classes_with_ties": sum(r.get(name, {}).get("has_ties", False) for r in classes),
        "active_classes_with_constant_priority": sum(r.get(name, {}).get("is_constant", False) for r in classes),
    } for name in priorities}
    return weights, {"classes": classes, "priority_counts": counts,
                     "distance_temperature": 0.1, "easy_rank_factor": 4,
                     "fallback_policy": "Preserve the parent's uniform stronger weights exactly.",
                     "native_comparisons": "Complete recipes; native weight spectra need not match OT."}
