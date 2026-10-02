"""Pure router-weight profiles and fit-set prediction diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from scripts.base_trainer import compute_class_groups

from .complementarity import (
    _spearman_rho,
    _validate_class_counts,
    _validate_labels_and_predictions,
    classify_predictions,
)
from .contracts import DiagnosticsError, EXPERT_NAMES, aligned_labeled_arrays, json_safe, simplex_array


def _entropy_bits(probabilities: np.ndarray) -> float:
    """Shannon entropy in bits, with the conventional ``0 log 0 = 0`` term."""
    values = np.asarray(probabilities, dtype=np.float64)
    positive = values > 0.0
    return float(-np.sum(values[positive] * np.log2(values[positive])))


def _profile_row(
    weights: np.ndarray,
    *,
    method: str,
    scope: str,
    scope_id: int | str | None,
) -> dict[str, Any]:
    if len(weights) == 0:
        return {
            "method": method,
            "scope": scope,
            "scope_id": scope_id,
            "sample_count": 0,
            "mean_weights": None,
            "std_weights": None,
            "allocation_entropy_bits": None,
            "normalized_allocation_entropy": None,
            "mean_sample_entropy_bits": None,
            "mean_normalized_sample_entropy": None,
            "preferred_expert": None,
            "preferred_expert_ties": None,
            "preferred_expert_tie_count": None,
            "preferred_expert_tie_policy": "first in EXPERT_NAMES order (CE first)",
            "per_row_preferred_tie_count_histogram": None,
            "preferred_fraction": None,
            "weight_quantiles": None,
        }
    means = weights.mean(axis=0)
    preferred = np.argmax(means)
    tied = np.flatnonzero(np.isclose(means, means.max(), rtol=0.0, atol=1e-12))
    per_row_preferred = np.argmax(weights, axis=1)
    fractions = np.bincount(per_row_preferred, minlength=len(EXPERT_NAMES)) / len(weights)
    row_tie_counts = np.isclose(
        weights, weights.max(axis=1, keepdims=True), rtol=0.0, atol=1e-12
    ).sum(axis=1)
    quantile_values = np.quantile(weights, [0.05, 0.25, 0.5, 0.75, 0.95], axis=0)
    mean_sample_entropy = float(np.mean([_entropy_bits(row) for row in weights]))
    return {
        "method": method,
        "scope": scope,
        "scope_id": scope_id,
        "sample_count": int(len(weights)),
        "mean_weights": means.tolist(),
        "std_weights": weights.std(axis=0).tolist(),
        "allocation_entropy_bits": _entropy_bits(means),
        "normalized_allocation_entropy": float(_entropy_bits(means) / 2.0),
        "mean_sample_entropy_bits": mean_sample_entropy,
        "mean_normalized_sample_entropy": float(mean_sample_entropy / 2.0),
        "preferred_expert": EXPERT_NAMES[int(preferred)],
        "preferred_expert_ties": [EXPERT_NAMES[int(index)] for index in tied],
        "preferred_expert_tie_count": int(len(tied)),
        "preferred_expert_tie_policy": "first in EXPERT_NAMES order (CE first)",
        "per_row_preferred_tie_count_histogram": {
            str(tie_count): int(np.sum(row_tie_counts == tie_count))
            for tie_count in range(1, len(EXPERT_NAMES) + 1)
        },
        "preferred_fraction": fractions.tolist(),
        "weight_quantiles": {
            quantile: quantile_values[index].tolist()
            for index, quantile in enumerate(("p05", "p25", "p50", "p75", "p95"))
        },
    }


def analyze_router_weights(
    logits: Any,
    labels: Any,
    weights_by_method: Mapping[str, Any],
    class_counts: Any,
    predictions_by_method: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize four-expert weights and weighted-logit fit-set metrics.

    Logits have shape ``(N, 4, C)``; labels have shape ``(N,)``; each method's
    row-stochastic weights have shape ``(N, 4)`` in the frozen
    ``EXPERT_NAMES`` order. Profiles include sample-weighted means by class
    group and separate class-macro means. Missing-class profiles are ``None``;
    fit-set BA/group recalls are also null whenever their canonical class set
    is incomplete. These are descriptive fit-set metrics, not held-out
    estimates. Supply ``predictions_by_method`` for any method using a
    classifier other than weighted-logit averaging (such as uniform
    probability averaging). If supplied, its method keys must exactly match
    ``weights_by_method`` and each value must be an aligned integer ``(N,)``
    prediction vector. Otherwise weighted-logit argmax is used for
    caller-owned methods. Prediction ties resolve to the first class index.
    Weight quantiles use NumPy's default linear quantile rule across the rows
    in each profile; normalized entropies divide bits by ``log2(4)=2``. The
    preferred expert uses first-index argmax, and all tied maxima are also
    reported; row-level preferred fractions therefore give ties to CE first.
    """
    scores, targets = aligned_labeled_arrays(logits, labels)
    counts = _validate_class_counts(class_counts, scores.shape[2])
    if not isinstance(weights_by_method, Mapping) or not weights_by_method:
        raise DiagnosticsError("weights_by_method must be a non-empty mapping")
    if predictions_by_method is not None:
        if (
            not isinstance(predictions_by_method, Mapping)
            or set(predictions_by_method) != set(weights_by_method)
        ):
            raise DiagnosticsError(
                "predictions_by_method keys must exactly match weights_by_method"
            )
    elif any("probability" in str(name).lower() for name in weights_by_method):
        raise DiagnosticsError(
            "probability-based methods require explicit predictions_by_method"
        )
    groups = compute_class_groups(counts)
    sample_counts = np.bincount(targets, minlength=len(counts))
    per_method: dict[str, Any] = {}
    overall_rows: list[dict[str, Any]] = []
    class_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    class_frequency_correlations: list[dict[str, Any]] = []

    for method, raw_weights in weights_by_method.items():
        if not isinstance(method, str) or not method:
            raise DiagnosticsError("method names must be non-empty strings")
        weights = simplex_array(raw_weights, name=f"weights[{method}]", rows=len(scores))
        if weights.shape[1] != len(EXPERT_NAMES):
            raise DiagnosticsError("router weights must follow the four-expert contract")
        if predictions_by_method is None:
            mixed_logits = np.einsum("ne,nec->nc", weights, scores, optimize=True)
            if not np.isfinite(mixed_logits).all():
                raise DiagnosticsError(f"weighted logits for {method} are non-finite")
            predictions = mixed_logits.argmax(axis=1).astype(np.int64)
        else:
            _, predictions = _validate_labels_and_predictions(
                targets, predictions_by_method[method], scores.shape[2]
            )
        metrics = classify_predictions(targets, predictions, counts)

        overall = _profile_row(
            weights, method=method, scope="overall", scope_id=None
        )
        overall_rows.append(overall)
        class_profile_by_id: dict[int, dict[str, Any]] = {}
        for class_id in range(len(counts)):
            class_weights = weights[targets == class_id]
            row = _profile_row(
                class_weights,
                method=method,
                scope="class",
                scope_id=class_id,
            )
            row["class_id"] = class_id
            row["training_count"] = int(counts[class_id])
            class_profile_by_id[class_id] = row
            class_rows.append(row)

        for group_name, class_ids in groups.items():
            group_mask = np.isin(targets, class_ids)
            row = _profile_row(
                weights[group_mask],
                method=method,
                scope="group",
                scope_id=group_name,
            )
            missing = [int(class_id) for class_id in class_ids if sample_counts[class_id] == 0]
            row["class_ids"] = [int(class_id) for class_id in class_ids]
            row["missing_class_ids"] = missing
            if len(class_ids) and not missing:
                class_macro = np.mean(
                    np.stack(
                        [class_profile_by_id[int(class_id)]["mean_weights"] for class_id in class_ids],
                        axis=0,
                    ),
                    axis=0,
                )
                row["class_macro_mean_weights"] = class_macro.tolist()
                row["class_macro_allocation_entropy_bits"] = _entropy_bits(class_macro)
            else:
                row["class_macro_mean_weights"] = None
                row["class_macro_allocation_entropy_bits"] = None
            group_rows.append(row)

        present_classes = np.flatnonzero(sample_counts > 0)
        for expert_id, expert_name in enumerate(EXPERT_NAMES):
            class_means = np.asarray(
                [
                    class_profile_by_id[int(class_id)]["mean_weights"][expert_id]
                    for class_id in present_classes
                ],
                dtype=np.float64,
            )
            class_frequency_correlations.append(
                {
                    "method": method,
                    "expert": expert_name,
                    "frequency_variable": "training_count",
                    "outcome": "class_mean_routing_weight",
                    "n_classes": int(len(present_classes)),
                    "spearman_rho": _spearman_rho(counts[present_classes], class_means),
                    "interpretation": (
                        "descriptive class-level rank association; not a causal estimate"
                    ),
                }
            )

        per_method[method] = {
            "fit_set_metrics": metrics,
            "overall_profile": overall,
            "class_profiles": [class_profile_by_id[class_id] for class_id in range(len(counts))],
            "group_profiles": [row for row in group_rows if row["method"] == method],
            "class_frequency_correlations": [
                row for row in class_frequency_correlations if row["method"] == method
            ],
        }

    return json_safe(
        {
            "n_samples": int(len(targets)),
            "methods": per_method,
            "overall_profiles": overall_rows,
            "class_profiles": class_rows,
            "group_profiles": group_rows,
            "class_frequency_correlations": class_frequency_correlations,
        }
    )
