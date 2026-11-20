"""Fixed Stage C rules on replayed real-only prototype geometry."""

import math

import torch
import torch.nn.functional as functional

from pipeline.ablation_weights import _midranks, baseline_scores, weight_diagnostics
from pipeline.methods.prototype_weights import _relaxed_sinkhorn

IMAGE_CONDITIONS = ("image_cb", "image_ot", "image_easy_rank",
                    "image_ot_shuffle0", "image_ot_shuffle1", "image_ot_shuffle2")
FEATURE_CONDITIONS = ("feature_real_cb", "feature_real_ot", "feature_duplicate_cb", "feature_duplicate_ot",
                      "feature_smote_cb", "feature_smote_ot", "feature_smote_prototype_distance",
                      "feature_smote_ot_shuffle0", "feature_smote_ot_shuffle1", "feature_smote_ot_shuffle2")


def validate_weights(weights, labels):
    if (weights.dtype != torch.float64 or weights.shape != labels.shape
            or not torch.isfinite(weights).all() or (weights <= 0).any()):
        raise ValueError("Weights must be positive finite float64 values aligned with labels")
    masses = torch.zeros(100, dtype=torch.float64).scatter_add_(0, labels, weights)
    if (not torch.allclose(masses, torch.full_like(masses, .01), rtol=1e-10, atol=1e-12)
            or not torch.isclose(weights.sum(), weights.new_tensor(1.), rtol=1e-10, atol=1e-12)):
        raise ValueError("Weights must give every class mass 1/100 and total mass one")


def normalize_weights(values, labels):
    values = values.double().clone()
    if values.shape != labels.shape or not torch.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Invalid unnormalized weights")
    for c in range(100):
        selected = labels == c
        if not selected.any():
            raise ValueError("Every class requires fitting rows")
        values[selected] /= 100 * values[selected].sum()
    validate_weights(values, labels)
    return values


def shuffle_weights(weights, labels, seed, origins=None):
    validate_weights(weights, labels)
    if origins is None:
        origins = torch.zeros_like(labels, dtype=torch.bool)
    if origins.shape != labels.shape or origins.dtype != torch.bool:
        raise ValueError("Invalid shuffle origins")
    generator = torch.Generator().manual_seed(seed)
    shuffled = weights.clone()
    for c in range(100):
        for origin in (False, True):
            positions = torch.where((labels == c) & (origins == origin))[0]
            values = weights[positions]
            shuffled[positions] = values[torch.randperm(len(positions), generator=generator)]
            if not torch.equal(values.sort().values, shuffled[positions].sort().values):
                raise RuntimeError("Shuffle changed a class/origin weight multiset")
    validate_weights(shuffled, labels)
    return shuffled


def weights_on_geometry(features, labels, geometry, diagnostics, options, rule="ot"):
    if rule not in ("ot", "prototype_distance") or len(geometry) != 100 or len(diagnostics) != 100:
        raise ValueError("Invalid fixed-geometry weight request")
    weights = torch.ones(len(labels), dtype=torch.float64)
    solver = []
    for c in range(100):
        if not diagnostics[c]["used_ot"]:
            continue
        centers = geometry[c]["centers"]
        if centers is None or centers.shape != (2, features.shape[1]) or not torch.isfinite(centers).all():
            raise ValueError("Missing active real-only centers")
        selected = labels == c
        # Match the replayed audit's cosine arithmetic before the float64 solver.
        cost = (1 - functional.normalize(features[selected], dim=1) @ centers.T).clamp_min(0)
        if rule == "ot":
            plan, record = _relaxed_sinkhorn(cost, **{k: options[k] for k in (
                "epsilon", "row_kl_strength", "sinkhorn_iterations", "sinkhorn_tolerance")})
            weights[selected] = plan.sum(dim=1)
            solver.append({"class_index": c, **record})
        else:
            weights[selected] = torch.softmax(-cost.min(dim=1).values.double() / .1, dim=0)
    return normalize_weights(weights, labels), solver


def build_image_weights(bundle, geometry, state):
    labels = bundle["labels"]
    options = bundle["inputs"]["prototype_options"]
    ot, solver = weights_on_geometry(bundle["features"], labels, geometry, bundle["prototype_diagnostics"], options)
    original = normalize_weights(bundle["ot_weights"], labels)
    if not torch.allclose(ot, original, rtol=1e-5, atol=1e-8):
        raise ValueError("Fixed geometry OT does not reproduce audit weights")
    margins, _ = baseline_scores(bundle["features"], labels, state)
    easy = torch.ones(len(labels), dtype=torch.float64)
    for c, diagnostic in enumerate(bundle["prototype_diagnostics"]):
        if diagnostic["used_ot"]:
            selected = labels == c
            ranks = _midranks(margins[selected].double())
            scaled = (ranks - ranks.min()) / (ranks.max() - ranks.min()) if ranks.max() > ranks.min() else ranks * 0
            easy[selected] = (math.log(4) * scaled).exp()
    weights = {"image_cb": normalize_weights(torch.ones_like(ot), labels), "image_ot": ot,
               "image_easy_rank": normalize_weights(easy, labels)}
    weights.update({f"image_ot_shuffle{s}": shuffle_weights(ot, labels, s) for s in (0, 1, 2)})
    return weights, {"solver": solver, "margins": margins}


def build_feature_weights(bundle, geometry, banks):
    weights, solvers = {}, {}
    for kind in ("real", "duplicate", "smote"):
        bank = bundle if kind == "real" else banks[kind]
        features, labels = bank["features"], bank["labels"]
        weights[f"feature_{kind}_cb"] = normalize_weights(torch.ones(len(labels)), labels)
        ot, solvers[kind] = weights_on_geometry(features, labels, geometry, bundle["prototype_diagnostics"],
                                              bundle["inputs"]["prototype_options"])
        weights[f"feature_{kind}_ot"] = ot
    if not torch.allclose(weights["feature_real_ot"], normalize_weights(bundle["ot_weights"], bundle["labels"]),
                          rtol=1e-5, atol=1e-8):
        raise ValueError("Real-only OT replay changed")
    bank = banks["smote"]
    weights["feature_smote_prototype_distance"], _ = weights_on_geometry(
        bank["features"], bank["labels"], geometry, bundle["prototype_diagnostics"],
        bundle["inputs"]["prototype_options"], "prototype_distance")
    for s in (0, 1, 2):
        weights[f"feature_smote_ot_shuffle{s}"] = shuffle_weights(
            weights["feature_smote_ot"], bank["labels"], s, bank["is_added"])
    return weights, solvers


def describe_weights(weights, labels, origins, original_counts, diagnostics):
    report = weight_diagnostics(weights, labels, original_counts, diagnostics)
    for c, row in enumerate(report["classes"]):
        selected = labels == c
        row.update(real_mass=float(weights[selected & ~origins].sum()),
                   synthetic_mass=float(weights[selected & origins].sum()),
                   has_added_rows=bool((selected & origins).any()))
    report["augmented_active_class_count"] = sum(r["used_ot"] and r["has_added_rows"] for r in report["classes"])
    report["augmented_fallback_class_count"] = sum(not r["used_ot"] and r["has_added_rows"] for r in report["classes"])
    return report
