"""Pure diagnostics for frozen-price Sinkhorn allocation stages."""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts.base_trainer import compute_class_groups
from scripts.ridge_sinkhorn_ot import apply_log_bias

from .complementarity import _validate_class_counts, classify_predictions
from .contracts import DiagnosticsError, aligned_labeled_arrays, finite_array, json_safe, simplex_array


def _entropy_bits(values: np.ndarray) -> float:
    positive = values > 0.0
    return float(-np.sum(values[positive] * np.log2(values[positive])))


def _kl_nats(p: np.ndarray, q: np.ndarray) -> float:
    positive = p > 0.0
    return float(np.sum(p[positive] * np.log(p[positive] / q[positive])))


def _rowwise_kl_nats(probabilities: np.ndarray, references: np.ndarray) -> np.ndarray:
    """Compute KL(p_i || q_i) row-wise, allowing zero p and requiring q>0."""
    log_ratio = np.zeros_like(probabilities, dtype=np.float64)
    positive = probabilities > 0.0
    log_ratio[positive] = np.log(probabilities[positive] / references[positive])
    return np.sum(probabilities * log_ratio, axis=1)


def _prediction_from_weights(logits: np.ndarray, weights: np.ndarray) -> np.ndarray:
    mixed = np.einsum("ne,nec->nc", weights, logits, optimize=True)
    if not np.isfinite(mixed).all():
        raise DiagnosticsError("weighted logits contain non-finite values")
    return mixed.argmax(axis=1).astype(np.int64)


def _correctness_transition(
    left: np.ndarray,
    right: np.ndarray,
    labels: np.ndarray,
    *,
    from_stage: str,
    to_stage: str,
) -> dict[str, Any]:
    left_correct = left == labels
    right_correct = right == labels
    wrong_to_correct = int(np.sum(~left_correct & right_correct))
    correct_to_wrong = int(np.sum(left_correct & ~right_correct))
    correct_to_correct = int(np.sum(left_correct & right_correct))
    wrong_to_wrong = int(np.sum(~left_correct & ~right_correct))
    total = int(len(labels))
    return {
        "from_stage": from_stage,
        "to_stage": to_stage,
        "helped_count": wrong_to_correct,
        "hurt_count": correct_to_wrong,
        "helped_fraction": float(wrong_to_correct / total),
        "hurt_fraction": float(correct_to_wrong / total),
        "correctness_contingency": {
            "wrong_to_correct": wrong_to_correct,
            "correct_to_wrong": correct_to_wrong,
            "correct_to_correct": correct_to_correct,
            "wrong_to_wrong": wrong_to_wrong,
        },
        "n_samples": total,
    }


def sinkhorn_stage_diagnostics(
    logits: Any,
    labels: Any,
    raw_weights: Any,
    smoothed_weights: Any,
    adjusted_weights: Any,
    prices: Any,
    target_prior: Any,
    class_counts: Any,
    method: str,
) -> dict[str, Any]:
    """Diagnose the raw -> smoothed -> frozen-price Sinkhorn decomposition.

    ``logits`` has shape ``(N, 4, C)``; labels and class counts have shapes
    ``(N,)`` and ``(C,)``; all three weight arrays have shape ``(N, 4)``;
    frozen log ``prices`` and positive ``target_prior`` have shape ``(4,)``.
    The expected adjusted weights are independently recomputed with the
    project's ``apply_log_bias`` rule. Mean adjusted allocation KL is
    ``KL(mean(adjusted_weights) || target_prior)`` in nats; it is a marginal
    discrepancy, not the per-row OT objective. The separate mean per-row
    ``KL(adjusted_i || smoothed_i)`` describes the adjustment from the kernel.
    Class and canonical Head/Medium/Tail rows report sample-weighted allocation
    deltas and correctness transitions; macro recalls require every class in a
    group. Label-dependent helped/hurt counts are descriptive transitions,
    not evidence of causal improvement.
    """
    scores, targets = aligned_labeled_arrays(logits, labels)
    counts = _validate_class_counts(class_counts, scores.shape[2])
    raw = simplex_array(raw_weights, name="raw_weights", rows=len(scores))
    smooth = simplex_array(smoothed_weights, name="smoothed_weights", rows=len(scores))
    adjusted = simplex_array(adjusted_weights, name="adjusted_weights", rows=len(scores))
    if not isinstance(method, str) or not method:
        raise DiagnosticsError("method must be a non-empty string")
    price_array = finite_array(prices, name="prices", ndim=1)
    if price_array.shape != (4,):
        raise DiagnosticsError("prices must have shape (4,)")
    prior = finite_array(target_prior, name="target_prior", ndim=1)
    if prior.shape != (4,) or np.any(prior <= 0.0) or not np.isclose(
        prior.sum(), 1.0, rtol=0.0, atol=1e-12
    ):
        raise DiagnosticsError("target_prior must be a positive four-expert simplex")
    if np.any(smooth <= 0.0):
        raise DiagnosticsError("smoothed_weights must be strictly positive for log bias")

    raw_predictions = _prediction_from_weights(scores, raw)
    smoothed_predictions = _prediction_from_weights(scores, smooth)
    adjusted_predictions = _prediction_from_weights(scores, adjusted)
    stage_arrays = {
        "raw": (raw, raw_predictions),
        "smoothed": (smooth, smoothed_predictions),
        "adjusted": (adjusted, adjusted_predictions),
    }
    stage_rows: list[dict[str, Any]] = []
    for stage_name, (weights, predictions) in stage_arrays.items():
        mean_weights = weights.mean(axis=0)
        stage_rows.append(
            {
                "stage": stage_name,
                "fit_set_metrics": classify_predictions(targets, predictions, counts),
                "mean_weights": mean_weights.tolist(),
                "std_weights": weights.std(axis=0).tolist(),
                "allocation_entropy_bits": _entropy_bits(mean_weights),
                "mean_sample_entropy_bits": float(
                    np.mean([_entropy_bits(row) for row in weights])
                ),
            }
        )

    canonical_groups = compute_class_groups(counts)
    scopes: list[tuple[str, int | str, np.ndarray, np.ndarray]] = []
    for class_id in range(len(counts)):
        class_mask = targets == class_id
        scopes.append(("class", class_id, class_mask, np.asarray([class_id], dtype=np.int64)))
    for group_name, class_ids in canonical_groups.items():
        group_mask = np.isin(targets, class_ids)
        scopes.append(("group", group_name, group_mask, np.asarray(class_ids, dtype=np.int64)))

    scoped_rows: list[dict[str, Any]] = []
    all_transitions = (
        ("raw", "smoothed", raw_predictions, smoothed_predictions),
        ("smoothed", "adjusted", smoothed_predictions, adjusted_predictions),
        ("raw", "adjusted", raw_predictions, adjusted_predictions),
    )
    rowwise_adjustment_kl = _rowwise_kl_nats(adjusted, smooth)
    for scope_name, scope_id, mask, scope_class_ids in scopes:
        scope_n = int(mask.sum())
        stage_summaries: dict[str, Any] = {}
        for stage_name, (weights, predictions) in stage_arrays.items():
            if scope_n:
                stage_sample_accuracy: float | None = float(
                    np.mean(predictions[mask] == targets[mask])
                )
                if scope_name == "class":
                    stage_macro_recall = stage_sample_accuracy
                    missing_scope_classes: list[int] = []
                else:
                    per_class_recalls = []
                    missing_scope_classes = []
                    for class_id in scope_class_ids:
                        class_rows_mask = targets == class_id
                        if not class_rows_mask.any():
                            missing_scope_classes.append(int(class_id))
                        else:
                            per_class_recalls.append(
                                float(np.mean(predictions[class_rows_mask] == class_id))
                            )
                    stage_macro_recall = (
                        float(np.mean(per_class_recalls))
                        if len(scope_class_ids) > 0 and not missing_scope_classes
                        else None
                    )
            else:
                stage_sample_accuracy = None
                stage_macro_recall = None
                missing_scope_classes = [int(value) for value in scope_class_ids]
            stage_summaries[stage_name] = {
                "mean_weights": weights[mask].mean(axis=0).tolist() if scope_n else None,
                "sample_accuracy": stage_sample_accuracy,
                "macro_recall": stage_macro_recall,
            }
        transition_rows = []
        for from_name, to_name, first_prediction, second_prediction in all_transitions:
            first_correct = first_prediction[mask] == targets[mask]
            second_correct = second_prediction[mask] == targets[mask]
            if scope_n:
                helped = int(np.sum(~first_correct & second_correct))
                hurt = int(np.sum(first_correct & ~second_correct))
                both_correct = int(np.sum(first_correct & second_correct))
                both_wrong = int(np.sum(~first_correct & ~second_correct))
                helped_fraction: float | None = float(helped / scope_n)
                hurt_fraction: float | None = float(hurt / scope_n)
            else:
                helped = hurt = both_correct = both_wrong = None
                helped_fraction = hurt_fraction = None
            transition_rows.append(
                {
                    "from_stage": from_name,
                    "to_stage": to_name,
                    "helped_count": helped,
                    "hurt_count": hurt,
                    "helped_fraction": helped_fraction,
                    "hurt_fraction": hurt_fraction,
                    "correctness_contingency": {
                        "wrong_to_correct": helped,
                        "correct_to_wrong": hurt,
                        "correct_to_correct": both_correct,
                        "wrong_to_wrong": both_wrong,
                    },
                }
            )
        scoped_rows.append(
            {
                "scope": scope_name,
                "scope_id": scope_id,
                "class_ids": [int(value) for value in scope_class_ids],
                "sample_count": scope_n,
                "missing_class_ids": missing_scope_classes,
                "stage_summaries": stage_summaries,
                "mean_raw_to_smoothed_l1": (
                    float(np.mean(np.abs(raw[mask] - smooth[mask]).sum(axis=1)))
                    if scope_n
                    else None
                ),
                "mean_smoothed_to_adjusted_l1": (
                    float(np.mean(np.abs(smooth[mask] - adjusted[mask]).sum(axis=1)))
                    if scope_n
                    else None
                ),
                "mean_raw_to_adjusted_l1": (
                    float(np.mean(np.abs(raw[mask] - adjusted[mask]).sum(axis=1)))
                    if scope_n
                    else None
                ),
                "mean_kl_adjusted_vs_smoothed_nats": (
                    float(np.mean(rowwise_adjustment_kl[mask])) if scope_n else None
                ),
                "transitions": transition_rows,
            }
        )

    expected_adjusted = apply_log_bias(smooth, price_array)
    decomposition_error = np.abs(adjusted - expected_adjusted)
    exact_decomposition = bool(
        np.allclose(adjusted, expected_adjusted, rtol=0.0, atol=1e-10)
    )
    mean_adjusted = adjusted.mean(axis=0)
    marginal_delta = mean_adjusted - prior
    centered_prices = price_array - price_array.mean()
    transitions = [
        _correctness_transition(
            raw_predictions,
            smoothed_predictions,
            targets,
            from_stage="raw",
            to_stage="smoothed",
        ),
        _correctness_transition(
            smoothed_predictions,
            adjusted_predictions,
            targets,
            from_stage="smoothed",
            to_stage="adjusted",
        ),
        _correctness_transition(
            raw_predictions,
            adjusted_predictions,
            targets,
            from_stage="raw",
            to_stage="adjusted",
        ),
    ]
    return json_safe(
        {
            "method": method,
            "n_samples": int(len(targets)),
            "stages": stage_rows,
            "decomposition": {
                "raw_to_smoothed_mean_l1": float(np.mean(np.abs(raw - smooth).sum(axis=1))),
                "smoothed_to_adjusted_mean_l1": float(np.mean(np.abs(smooth - adjusted).sum(axis=1))),
                "raw_to_adjusted_mean_l1": float(np.mean(np.abs(raw - adjusted).sum(axis=1))),
                "expected_adjusted_max_abs_error": float(np.max(decomposition_error)),
                "exact_frozen_price_decomposition": exact_decomposition,
                "decomposition_status": "exact" if exact_decomposition else "mismatch",
            },
            "sinkhorn": {
                "target_prior": prior.tolist(),
                "adjusted_marginal": mean_adjusted.tolist(),
                "marginal_delta": marginal_delta.tolist(),
                "max_abs_marginal_residual": float(np.max(np.abs(marginal_delta))),
                "marginal_kl_nats": _kl_nats(mean_adjusted, prior),
                "mean_row_kl_adjusted_vs_smoothed_nats": float(
                    np.mean(rowwise_adjustment_kl)
                ),
                "prices": price_array.tolist(),
                "centered_prices": centered_prices.tolist(),
                "centered_price_range": float(np.max(centered_prices) - np.min(centered_prices)),
            },
            "scoped_rows": scoped_rows,
            "transitions": transitions,
        }
    )
