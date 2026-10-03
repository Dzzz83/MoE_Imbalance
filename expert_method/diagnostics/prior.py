"""Array-only comparisons between saved router weights and prior controls."""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts.base_trainer import compute_class_groups

from .complementarity import classify_predictions
from .contracts import DiagnosticsError, aligned_labeled_arrays, finite_array, simplex_array
from .stability import compare_profiles


def _departure_rows(weights: np.ndarray, reference: np.ndarray, labels: np.ndarray,
                    class_counts: np.ndarray) -> list[dict[str, Any]]:
    """Summarize L1 profile departures by all rows, class, and canonical group."""
    groups = compute_class_groups(class_counts)
    scopes: list[tuple[str, int | str, np.ndarray]] = [
        ("overall", "overall", np.ones(len(labels), dtype=bool))
    ]
    scopes.extend(("class", class_id, labels == class_id) for class_id in range(len(class_counts)))
    scopes.extend((name, name, np.isin(labels, class_ids)) for name, class_ids in groups.items())
    distances = np.abs(weights - reference[None, :]).sum(axis=1)
    rows = []
    for scope, scope_id, mask in scopes:
        values = weights[mask]
        delta = distances[mask]
        present = len(values) > 0
        mean = values.mean(axis=0) if present else None
        missing = (
            [int(class_id) for class_id in range(len(class_counts)) if not np.any(labels == class_id)]
            if scope == "overall"
            else ([int(class_id) for class_id in groups[scope_id] if not np.any(labels == class_id)]
                  if scope in groups else ([int(scope_id)] if not present else []))
        )
        rows.append({
            "scope": scope, "scope_id": scope_id, "sample_count": int(mask.sum()),
            "mean_weights": None if mean is None else mean.tolist(),
            "mean_weight_minus_reference": None if mean is None else (mean - reference).tolist(),
            "mean_l1_departure": float(delta.mean()) if present else None,
            "l1_departure_quantiles": (
                {key: float(np.quantile(delta, q)) for key, q in
                 (("p05", .05), ("p25", .25), ("p50", .50), ("p75", .75), ("p95", .95))}
                if present else None
            ),
            "missing_class_ids": missing,
        })
    return rows


def analyze_prior_relation(
    weights: Any, reference: Any, labels: Any, class_counts: Any,
) -> dict[str, Any]:
    """Measure adaptive allocation departures from one saved prior profile.

    Weights have shape (N,4); reference is the constant prior vector (4,).
    Overall distance compares mean allocations, while samplewise quantiles
    retain per-row departure. Class and Head/Medium/Tail values are
    sample-weighted; missing classes remain explicitly listed.
    """
    adaptive = simplex_array(weights, name="weights")
    prior = finite_array(reference, name="reference", ndim=1)
    if prior.shape != (4,) or np.any(prior < 0.0) or not np.isclose(prior.sum(), 1.0, rtol=0.0, atol=1e-10):
        raise DiagnosticsError("reference must be a four-expert probability profile")
    targets = np.asarray(labels)
    counts = finite_array(class_counts, name="class_counts", ndim=1)
    if targets.shape != (len(adaptive),) or not np.issubdtype(targets.dtype, np.integer):
        raise DiagnosticsError("labels must align with weights")
    if counts.shape[0] < 2 or np.any(counts <= 0) or not np.equal(counts, np.floor(counts)).all():
        raise DiagnosticsError("class_counts must be positive integer counts")
    if np.any(targets < 0) or np.any(targets >= len(counts)):
        raise DiagnosticsError("labels fall outside class_counts")
    mean = adaptive.mean(axis=0)
    details = _departure_rows(adaptive, prior, targets.astype(np.int64), counts.astype(np.int64))
    return {
        "mean_profile": mean.tolist(),
        "reference_profile": prior.tolist(),
        "mean_profile_comparison": compare_profiles(mean, prior),
        "overall_samplewise_l1_departure": details[0]["l1_departure_quantiles"],
        "scoped_departures": details,
    }


def correctness_transition(
    labels: Any, baseline_predictions: Any, candidate_predictions: Any,
    class_counts: Any, *, baseline_name: str, candidate_name: str,
) -> dict[str, Any]:
    """Report label-dependent helped/hurt transitions between predictions."""
    targets = np.asarray(labels)
    baseline = np.asarray(baseline_predictions)
    candidate = np.asarray(candidate_predictions)
    if targets.ndim != 1 or baseline.shape != targets.shape or candidate.shape != targets.shape:
        raise DiagnosticsError("transition labels and predictions must be aligned vectors")
    if any(not np.issubdtype(array.dtype, np.integer) for array in (targets, baseline, candidate)):
        raise DiagnosticsError("transition labels and predictions must be integer vectors")
    before, after = baseline == targets, candidate == targets
    return {
        "baseline_method": baseline_name,
        "candidate_method": candidate_name,
        "baseline_metrics": classify_predictions(targets, baseline, class_counts),
        "candidate_metrics": classify_predictions(targets, candidate, class_counts),
        "candidate_helped_count": int(np.sum(~before & after)),
        "candidate_hurt_count": int(np.sum(before & ~after)),
        "correct_to_correct_count": int(np.sum(before & after)),
        "wrong_to_wrong_count": int(np.sum(~before & ~after)),
        "changed_prediction_count": int(np.sum(baseline != candidate)),
        "n_samples": int(len(targets)),
        "interpretation": (
            "candidate_helped means baseline wrong and candidate correct; "
            "candidate_hurt means baseline correct and candidate wrong"
        ),
    }


def scoped_correctness_transitions(
    labels: Any, baseline_predictions: Any, candidate_predictions: Any,
    class_counts: Any, *, baseline_name: str, candidate_name: str,
) -> list[dict[str, Any]]:
    """Report helped/hurt sample counts overall, per class, and for Head/M/T."""
    targets = np.asarray(labels)
    baseline = np.asarray(baseline_predictions)
    candidate = np.asarray(candidate_predictions)
    counts = np.asarray(class_counts)
    if targets.ndim != 1 or baseline.shape != targets.shape or candidate.shape != targets.shape:
        raise DiagnosticsError("transition labels and predictions must be aligned vectors")
    scopes: list[tuple[str, int | str, np.ndarray]] = [
        ("overall", "overall", np.ones(len(targets), dtype=bool)),
    ]
    scopes.extend(("class", class_id, targets == class_id) for class_id in range(len(counts)))
    scopes.extend((str(group).title(), str(group).title(), np.isin(targets, classes))
                  for group, classes in compute_class_groups(counts).items())
    rows = []
    for scope, scope_id, mask in scopes:
        before = baseline[mask] == targets[mask]
        after = candidate[mask] == targets[mask]
        rows.append({
            "scope": scope, "scope_id": scope_id, "sample_count": int(mask.sum()),
            "candidate_helped_count": int(np.sum(~before & after)),
            "candidate_hurt_count": int(np.sum(before & ~after)),
            "correct_to_correct_count": int(np.sum(before & after)),
            "wrong_to_wrong_count": int(np.sum(~before & ~after)),
            "changed_prediction_count": int(np.sum(baseline[mask] != candidate[mask])),
            "baseline_method": baseline_name, "candidate_method": candidate_name,
            "interpretation": (
                "candidate_helped means baseline wrong and candidate correct; "
                "candidate_hurt means baseline correct and candidate wrong"
            ),
        })
    return rows


def mean_weight_counterfactual(
    logits: Any, labels: Any, weights: Any, predictions: Any, class_counts: Any,
) -> dict[str, Any]:
    """Compare saved predictions with constant mean-weight weighted-logit scores."""
    scores, targets = aligned_labeled_arrays(logits, labels)
    adaptive = simplex_array(weights, name="weights", rows=len(scores))
    saved = np.asarray(predictions)
    if saved.shape != targets.shape or not np.issubdtype(saved.dtype, np.integer):
        raise DiagnosticsError("saved predictions must be an aligned integer vector")
    mean = adaptive.mean(axis=0)
    counterfactual = np.einsum("e,nec->nc", mean, scores, optimize=True).argmax(axis=1).astype(np.int64)
    transition = correctness_transition(
        targets, counterfactual, saved, class_counts,
        baseline_name="constant_mean_weight_counterfactual",
        candidate_name="saved_adaptive_router",
    )
    scoped_transitions = scoped_correctness_transitions(
        targets, counterfactual, saved, class_counts,
        baseline_name="constant_mean_weight_counterfactual",
        candidate_name="saved_adaptive_router",
    )
    return {
        "mean_weights": mean.tolist(),
        "counterfactual_metrics": classify_predictions(targets, counterfactual, class_counts),
        "saved_router_metrics": classify_predictions(targets, saved, class_counts),
        "adaptive_helped_count": transition["candidate_helped_count"],
        "adaptive_hurt_count": transition["candidate_hurt_count"],
        "counterfactual_helped_count": transition["candidate_hurt_count"],
        "counterfactual_hurt_count": transition["candidate_helped_count"],
        "changed_prediction_count": transition["changed_prediction_count"],
        "scoped_transitions": scoped_transitions,
        "n_samples": len(targets),
        "interpretation": (
            "adaptive_helped means saved adaptive state was correct where "
            "constant mean weights were wrong"
        ),
    }
