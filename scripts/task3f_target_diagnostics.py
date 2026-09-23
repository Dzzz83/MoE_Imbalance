"""Task 3F-E: retrospective supervised-target diagnostics.

This module consumes the already validated Task 3C OOF logits and the saved
Task 3F-A held-out Ridge scores/weights.  It does not fit a router, retrain an
expert, alter a source prediction, or access a reserved population.  Labels
are used only for the explicitly retrospective target, margin, group, and
classification-outcome calculations.

The public array helpers are intentionally small and independent of the real
artifact loader so that the target, margin, and convex perturbation contracts
can be tested on synthetic logits.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np

from data.nested_oof import NestedOOFFoldManager, OOFProtocolError
from scripts.base_trainer import compute_class_groups
from scripts.task3e_fixed import (
    EXPERT_ORDER,
    RestrictedAnalysisDataset,
    load_restricted_analysis_dataset,
)
from scripts.task3f_ridge import (
    FEATURE_SET_CONFIDENCE,
    classification_metrics,
    combine_weighted_logits,
    compute_contribution_targets,
)


TASK3FE_SCHEMA_VERSION = "task3f_target_diagnostics.v1"
TASK3FE_TASK_IDENTIFIER = "Task 3F-E"
TASK3FA_TASK_IDENTIFIER = "Task 3F-A"
DATASET_NAME = "CIFAR-100-LT"
IMBALANCE_RATIO = 100.0
CANONICAL_POPULATION_SIZE = 10_847
ANALYSIS_SAMPLE_COUNT = 6_507
NUM_CLASSES = 100
TRAINING_SEED = 78
FOLD_GENERATION_SEED = 42
OUTER_FOLD = 0
PERMITTED_ANALYSIS_INNER_FOLDS = (1, 2, 3)
RESERVED_ROUTER_SELECTION_INNER_FOLDS = (0,)
OUTER_EVALUATION_SIZE = 2_170
EPSILONS = (0.1, 0.25, 0.5, 1.0)
MIXUP_INDEX = EXPERT_ORDER.index("Mixup")
HIGHLIGHTED_CONFIGURATION_ID = (
    "adaptive_confidence_only_alpha1000_gamma1_temperature2_lambda0p75"
)
RIDGE_SCORE_MODEL_TEMPLATE = (
    "ridge_fold{fold}_confidence_only_alpha1000_gamma1"
)


class Task3FETargetDiagnosticError(OOFProtocolError):
    """Raised when Task 3F-E input validation or calculations fail."""


def _as_numeric(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise Task3FETargetDiagnosticError(f"{name} must contain real numeric values")
    if not np.isfinite(array).all():
        raise Task3FETargetDiagnosticError(f"{name} contains non-finite values")
    return array.astype(np.float64, copy=False)


def _as_integer_vector(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise Task3FETargetDiagnosticError(f"{name} must be one-dimensional")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise Task3FETargetDiagnosticError(f"{name} must contain integer values")
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        raise Task3FETargetDiagnosticError(
            f"{name} must contain finite integer-valued entries"
        )
    return array.astype(np.int64, copy=False)


def _validate_logits(logits: Any, *, name: str = "logits") -> np.ndarray:
    array = _as_numeric(logits, name=name)
    if array.ndim != 3 or array.shape[1] != len(EXPERT_ORDER):
        raise Task3FETargetDiagnosticError(
            f"{name} must have shape (samples, {len(EXPERT_ORDER)}, classes), "
            f"got {array.shape}"
        )
    if array.shape[0] == 0 or array.shape[2] < 2:
        raise Task3FETargetDiagnosticError(
            f"{name} must contain samples and at least two classes"
        )
    return array


def _validate_labels(
    labels: Any,
    *,
    num_samples: int,
    num_classes: int,
    name: str = "labels",
) -> np.ndarray:
    array = _as_integer_vector(labels, name=name)
    if array.shape != (num_samples,):
        raise Task3FETargetDiagnosticError(
            f"{name} must have shape ({num_samples},), got {array.shape}"
        )
    if np.any(array < 0) or np.any(array >= num_classes):
        raise Task3FETargetDiagnosticError(
            f"{name} must be in [0, {num_classes})"
        )
    return array


def _validate_class_counts(class_counts: Any, *, num_classes: int) -> np.ndarray:
    counts = _as_integer_vector(class_counts, name="class_counts")
    if len(counts) != num_classes or np.any(counts < 1):
        raise Task3FETargetDiagnosticError(
            f"class_counts must contain {num_classes} positive counts"
        )
    return counts


def compute_strongest_incorrect_classes(
    logits: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Return the uniform ensemble's strongest incorrect class per image.

    Ties use NumPy's stable first-index ``argmax`` behavior after the true
    class is masked.  The function is label-dependent by definition and is
    intended only for the retrospective margin diagnostic.
    """
    array = _validate_logits(logits)
    labels_array = _validate_labels(
        labels,
        num_samples=array.shape[0],
        num_classes=array.shape[2],
    )
    ensemble = array.mean(axis=1)
    masked = ensemble.copy()
    masked[np.arange(len(labels_array)), labels_array] = -np.inf
    competitors = masked.argmax(axis=1).astype(np.int64)
    return competitors


def compute_margin_contributions(
    logits: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Compute each expert's contribution to the uniform true-class margin.

    For the current uniform ensemble competitor ``c*``, this returns

    ``(z_e,y - z_e,c*) - (z_bar,y - z_bar,c*)``.

    The returned values sum to zero across experts for every image.  This is a
    retrospective quantity and is not used to fit the saved Ridge router.
    """
    array = _validate_logits(logits)
    labels_array = _validate_labels(
        labels,
        num_samples=array.shape[0],
        num_classes=array.shape[2],
    )
    competitors = compute_strongest_incorrect_classes(array, labels_array)
    true_logits = np.take_along_axis(
        array,
        labels_array[:, None, None],
        axis=2,
    ).squeeze(axis=2)
    competitor_logits = np.take_along_axis(
        array,
        competitors[:, None, None],
        axis=2,
    ).squeeze(axis=2)
    expert_margins = true_logits - competitor_logits
    ensemble = array.mean(axis=1)
    ensemble_margin = (
        ensemble[np.arange(len(labels_array)), labels_array]
        - ensemble[np.arange(len(labels_array)), competitors]
    )
    contributions = expert_margins - ensemble_margin[:, None]
    if not np.isfinite(contributions).all():
        raise Task3FETargetDiagnosticError(
            "classification-margin contributions contain non-finite values"
        )
    if not np.allclose(contributions.sum(axis=1), 0.0, rtol=1e-9, atol=1e-12):
        raise Task3FETargetDiagnosticError(
            "classification-margin contributions do not sum to zero"
        )
    return contributions


# A descriptive alias makes the definition easy to find for callers that use
# the longer name from the task specification.
compute_classification_margin_contributions = compute_margin_contributions


def convex_weight_perturbation(
    uniform_weights_or_num_samples: Any,
    expert_index: int,
    epsilon: float,
) -> np.ndarray:
    """Move uniform weights toward one expert by a fixed convex amount.

    ``uniform_weights_or_num_samples`` may be a positive sample count, a
    length-four uniform vector, or an ``(N, 4)`` matrix of uniform rows.  The
    returned shape matches a supplied vector/matrix and is ``(N, 4)`` for a
    supplied sample count.
    """
    if isinstance(expert_index, bool) or int(expert_index) != expert_index:
        raise Task3FETargetDiagnosticError("expert_index must be an integer")
    expert_index = int(expert_index)
    if not 0 <= expert_index < len(EXPERT_ORDER):
        raise Task3FETargetDiagnosticError("expert_index is outside the expert pool")
    if (
        isinstance(epsilon, bool)
        or not np.isfinite(float(epsilon))
        or not 0.0 <= float(epsilon) <= 1.0
    ):
        raise Task3FETargetDiagnosticError("epsilon must be finite and in [0, 1]")
    epsilon = float(epsilon)

    supplied = np.asarray(uniform_weights_or_num_samples)
    was_vector = False
    if supplied.ndim == 0:
        if not np.issubdtype(supplied.dtype, np.number):
            raise Task3FETargetDiagnosticError(
                "sample count must be a positive integer"
            )
        if int(supplied) != supplied or int(supplied) < 1:
            raise Task3FETargetDiagnosticError(
                "sample count must be a positive integer"
            )
        matrix = np.full(
            (int(supplied), len(EXPERT_ORDER)),
            1.0 / len(EXPERT_ORDER),
            dtype=np.float64,
        )
    else:
        array = _as_numeric(
            uniform_weights_or_num_samples,
            name="uniform weights",
        )
        was_vector = array.ndim == 1
        if was_vector:
            if array.shape != (len(EXPERT_ORDER),):
                raise Task3FETargetDiagnosticError(
                    "uniform weight vector must contain four experts"
                )
            matrix = array[None, :].copy()
        elif array.ndim == 2 and array.shape[1] == len(EXPERT_ORDER):
            if array.shape[0] == 0:
                raise Task3FETargetDiagnosticError(
                    "uniform weight matrix must contain samples"
                )
            matrix = array.copy()
        else:
            raise Task3FETargetDiagnosticError(
                "uniform weights must be a length-four vector or an (N, 4) matrix"
            )
        if np.any(matrix < 0.0) or not np.allclose(
            matrix,
            1.0 / len(EXPERT_ORDER),
            rtol=0.0,
            atol=1e-12,
        ):
            raise Task3FETargetDiagnosticError(
                "the supplied base weights must be uniform"
            )

    perturbed = (1.0 - epsilon) * matrix
    perturbed[:, expert_index] += epsilon
    # Normalize only round-off in the convex sum; this does not change the
    # specified perturbation in meaningful precision and keeps downstream
    # validation exact.
    perturbed /= perturbed.sum(axis=1, keepdims=True)
    if not np.isfinite(perturbed).all() or np.any(perturbed < 0.0):
        raise Task3FETargetDiagnosticError(
            "convex perturbation produced invalid weights"
        )
    if not np.allclose(perturbed.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise Task3FETargetDiagnosticError(
            "convex perturbation weights do not sum to one"
        )
    return perturbed[0] if was_vector else perturbed


# Short alias used by some diagnostics-oriented callers.
perturb_uniform_weights = convex_weight_perturbation


def _validate_combined_logits(logits: Any, *, name: str = "combined logits") -> np.ndarray:
    array = _as_numeric(logits, name=name)
    if array.ndim != 2 or array.shape[1] < 2 or array.shape[0] == 0:
        raise Task3FETargetDiagnosticError(
            f"{name} must have shape (samples, classes), got {array.shape}"
        )
    return array


def _true_log_probabilities(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    array = _validate_combined_logits(logits)
    labels_array = _validate_labels(
        labels,
        num_samples=array.shape[0],
        num_classes=array.shape[1],
    )
    shifted = array - array.max(axis=1, keepdims=True)
    log_denominator = np.log(np.exp(shifted).sum(axis=1))
    log_probabilities = shifted - log_denominator[:, None]
    return log_probabilities[np.arange(len(labels_array)), labels_array]


def _true_classification_margins(
    logits: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return dynamic true-class margins and their current competitors."""
    array = _validate_combined_logits(logits)
    labels_array = _validate_labels(
        labels,
        num_samples=array.shape[0],
        num_classes=array.shape[1],
    )
    masked = array.copy()
    masked[np.arange(len(labels_array)), labels_array] = -np.inf
    competitors = masked.argmax(axis=1).astype(np.int64)
    margins = (
        array[np.arange(len(labels_array)), labels_array]
        - array[np.arange(len(labels_array)), competitors]
    )
    return margins, competitors


def _canonical_group_labels(labels: np.ndarray, class_counts: np.ndarray) -> np.ndarray:
    groups = compute_class_groups(class_counts)
    result = np.full(len(labels), "unknown", dtype=object)
    for group_name, classes in groups.items():
        result[np.isin(labels, classes)] = group_name
    if np.any(result == "unknown"):
        raise Task3FETargetDiagnosticError(
            "some labels do not belong to a canonical class group"
        )
    return result


def _fraction(count: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(count / denominator)


def _safe_float(value: Any) -> float | None:
    value = float(value)
    return None if not np.isfinite(value) else value


def _distribution(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise Task3FETargetDiagnosticError("distribution values must be one-dimensional")
    if len(values) == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "minimum": None,
            "maximum": None,
            "quantiles": {key: None for key in ("q01", "q25", "q50", "q75", "q99")},
        }
    if not np.isfinite(values).all():
        raise Task3FETargetDiagnosticError("distribution values are non-finite")
    quantiles = np.quantile(values, [0.01, 0.25, 0.50, 0.75, 0.99])
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "quantiles": {
            key: float(value)
            for key, value in zip(
                ("q01", "q25", "q50", "q75", "q99"),
                quantiles,
            )
        },
    }


def _matrix_report(
    values: np.ndarray,
    labels: np.ndarray,
    class_counts: np.ndarray,
    *,
    value_name: str,
) -> dict[str, Any]:
    """Report distributions and per-image rankings for a four-column matrix."""
    array = _as_numeric(values, name=value_name)
    if array.ndim != 2 or array.shape != (len(labels), len(EXPERT_ORDER)):
        raise Task3FETargetDiagnosticError(
            f"{value_name} must have shape (samples, {len(EXPERT_ORDER)})"
        )
    group_labels = _canonical_group_labels(labels, class_counts)
    report: dict[str, Any] = {}
    for group_name in ("head", "medium", "tail"):
        mask = group_labels == group_name
        group_values = array[mask]
        winners = group_values.argmax(axis=1)
        report[group_name] = {
            "sample_count": int(mask.sum()),
            "distribution": {
                expert: _distribution(group_values[:, index])
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "mean": {
                expert: float(group_values[:, index].mean())
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "median": {
                expert: float(np.median(group_values[:, index]))
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "highest_expert_count": {
                expert: int(np.sum(winners == index))
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "highest_expert_fraction": {
                expert: float(np.mean(winners == index))
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "highest_expert_percentage": {
                expert: float(100.0 * np.mean(winners == index))
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "positive_count": {
                expert: int(np.sum(group_values[:, index] > 0.0))
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "positive_fraction": {
                expert: float(np.mean(group_values[:, index] > 0.0))
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "positive_percentage": {
                expert: float(100.0 * np.mean(group_values[:, index] > 0.0))
                for index, expert in enumerate(EXPERT_ORDER)
            },
        }
        mixup = group_values[:, MIXUP_INDEX]
        differences: dict[str, Any] = {}
        for index, expert in enumerate(EXPERT_ORDER):
            if index == MIXUP_INDEX:
                continue
            delta = mixup - group_values[:, index]
            differences[expert] = {
                "distribution": _distribution(delta),
                "mean": float(delta.mean()),
                "median": float(np.median(delta)),
                "positive_fraction": float(np.mean(delta > 0.0)),
                "positive_percentage": float(100.0 * np.mean(delta > 0.0)),
            }
        report[group_name]["Mixup_minus_expert"] = differences
    return report


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) == 0 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _target_prediction_comparison(
    actual: np.ndarray,
    predicted: np.ndarray,
    labels: np.ndarray,
    class_counts: np.ndarray,
) -> dict[str, Any]:
    group_labels = _canonical_group_labels(labels, class_counts)
    actual_winners = actual.argmax(axis=1)
    predicted_winners = predicted.argmax(axis=1)
    report: dict[str, Any] = {}
    for group_name in ("head", "medium", "tail"):
        mask = group_labels == group_name
        actual_group = actual[mask]
        predicted_group = predicted[mask]
        actual_winner = actual_winners[mask]
        predicted_winner = predicted_winners[mask]
        pairwise: dict[str, Any] = {}
        for left in range(len(EXPERT_ORDER)):
            for right in range(left + 1, len(EXPERT_ORDER)):
                actual_order = np.sign(actual_group[:, left] - actual_group[:, right])
                predicted_order = np.sign(
                    predicted_group[:, left] - predicted_group[:, right]
                )
                pair_name = f"{EXPERT_ORDER[left]}_vs_{EXPERT_ORDER[right]}"
                pairwise[pair_name] = {
                    "agreement_count": int(np.sum(actual_order == predicted_order)),
                    "agreement_fraction": float(
                        np.mean(actual_order == predicted_order)
                    ),
                }
        error = predicted_group - actual_group
        report[group_name] = {
            "sample_count": int(mask.sum()),
            "mse": float(np.mean(error**2)),
            "mae": float(np.mean(np.abs(error))),
            "pearson_correlation": _pearson(actual_group.ravel(), predicted_group.ravel()),
            "sign_agreement_fraction": float(
                np.mean(np.sign(actual_group) == np.sign(predicted_group))
            ),
            "highest_expert_agreement_count": int(
                np.sum(actual_winner == predicted_winner)
            ),
            "highest_expert_agreement_fraction": float(
                np.mean(actual_winner == predicted_winner)
            ),
            "actual_highest_expert_fraction": {
                expert: float(np.mean(actual_winner == index))
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "predicted_highest_expert_fraction": {
                expert: float(np.mean(predicted_winner == index))
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "both_highest_Mixup_fraction": float(
                np.mean((actual_winner == MIXUP_INDEX) & (predicted_winner == MIXUP_INDEX))
            ),
            "pairwise_order": pairwise,
        }
    return report


def _tail_rebalanced_misranking(
    actual: np.ndarray,
    predicted: np.ndarray,
    labels: np.ndarray,
    class_counts: np.ndarray,
) -> dict[str, Any]:
    groups = _canonical_group_labels(labels, class_counts)
    tail = groups == "tail"
    report: dict[str, Any] = {}
    for expert in ("LAL", "BalancedSoftmax"):
        index = EXPERT_ORDER.index(expert)
        actual_higher = actual[:, index] > actual[:, MIXUP_INDEX]
        predicted_higher = predicted[:, index] > predicted[:, MIXUP_INDEX]
        selected = tail & actual_higher
        mixup_winner = predicted.argmax(axis=1) == MIXUP_INDEX
        report[expert] = {
            "tail_sample_count": int(tail.sum()),
            "actual_higher_than_Mixup_count": int(np.sum(selected)),
            "actual_higher_than_Mixup_fraction_of_Tail": _fraction(
                int(np.sum(selected)), int(tail.sum())
            ),
            "Ridge_predicted_rebalanced_higher_count": int(
                np.sum(selected & predicted_higher)
            ),
            "Ridge_predicted_rebalanced_higher_fraction": _fraction(
                int(np.sum(selected & predicted_higher)), int(np.sum(selected))
            ),
            "Ridge_predicted_Mixup_highest_count": int(
                np.sum(selected & mixup_winner)
            ),
            "Ridge_predicted_Mixup_highest_fraction": _fraction(
                int(np.sum(selected & mixup_winner)), int(np.sum(selected))
            ),
            "pairwise_order_accuracy_on_tail": _fraction(
                int(np.sum(tail & (actual_higher == predicted_higher))),
                int(tail.sum()),
            ),
        }
    union = tail & ((actual[:, 1] > actual[:, MIXUP_INDEX]) | (actual[:, 2] > actual[:, MIXUP_INDEX]))
    report["LAL_or_BalancedSoftmax"] = {
        "tail_sample_count": int(tail.sum()),
        "actual_rebalanced_higher_count": int(union.sum()),
        "Ridge_predicted_Mixup_highest_count": int(
            np.sum(union & (predicted.argmax(axis=1) == MIXUP_INDEX))
        ),
        "Ridge_predicted_Mixup_highest_fraction": _fraction(
            int(np.sum(union & (predicted.argmax(axis=1) == MIXUP_INDEX))),
            int(union.sum()),
        ),
    }
    return report


def _metric_bundle(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_counts: np.ndarray,
) -> dict[str, float]:
    metrics = classification_metrics(labels, predictions, class_counts)
    return {
        "ordinary_accuracy": float(metrics["ordinary_accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "head_accuracy": float(metrics["head_accuracy"]),
        "medium_accuracy": float(metrics["medium_accuracy"]),
        "tail_accuracy": float(metrics["tail_accuracy"]),
    }


def _group_event_report(
    *,
    mask: np.ndarray,
    group_labels: np.ndarray,
    target: np.ndarray,
    delta_log_probability: np.ndarray,
    delta_margin: np.ndarray,
    base_predictions: np.ndarray,
    new_predictions: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    expert_index: int,
    epsilon: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Summarize one fixed perturbation and return positive-target loss cases."""
    base_correct = base_predictions == labels
    new_correct = new_predictions == labels
    changed = new_predictions != base_predictions
    corrected = ~base_correct & new_correct
    deteriorated = base_correct & ~new_correct
    positive_target = target[:, expert_index] > 0.0
    report: dict[str, Any] = {}
    case_indices = np.flatnonzero(mask & positive_target & deteriorated)
    cases = [
        {
            "sample_id": int(sample_ids[index]),
            "true_class": int(labels[index]),
            "true_group": str(group_labels[index]),
            "target": float(target[index, expert_index]),
            "delta_true_log_probability": float(delta_log_probability[index]),
            "delta_true_margin": float(delta_margin[index]),
            "uniform_prediction": int(base_predictions[index]),
            "perturbed_prediction": int(new_predictions[index]),
            "epsilon": float(epsilon),
        }
        for index in case_indices
    ]
    for group_name in ("head", "medium", "tail"):
        group_mask = mask & (group_labels == group_name)
        count = int(group_mask.sum())
        group_positive = group_mask & positive_target
        positive_count = int(group_positive.sum())
        group_base_incorrect = group_mask & ~base_correct
        group_base_correct = group_mask & base_correct
        group_corrected = group_mask & corrected
        group_deteriorated = group_mask & deteriorated
        report[group_name] = {
            "sample_count": count,
            "mean_change_true_log_probability": _safe_float(
                delta_log_probability[group_mask].mean()
            ),
            "median_change_true_log_probability": _safe_float(
                np.median(delta_log_probability[group_mask])
            ),
            "mean_change_true_margin": _safe_float(delta_margin[group_mask].mean()),
            "median_change_true_margin": _safe_float(np.median(delta_margin[group_mask])),
            "prediction_changed_count": int(np.sum(group_mask & changed)),
            "prediction_changed_fraction": _fraction(
                int(np.sum(group_mask & changed)), count
            ),
            "corrected_count": int(group_corrected.sum()),
            "corrected_fraction_of_group": _fraction(int(group_corrected.sum()), count),
            "correction_rate_among_base_incorrect": _fraction(
                int(group_corrected.sum()), int(group_base_incorrect.sum())
            ),
            "deteriorated_count": int(group_deteriorated.sum()),
            "deteriorated_fraction_of_group": _fraction(
                int(group_deteriorated.sum()), count
            ),
            "deterioration_rate_among_base_correct": _fraction(
                int(group_deteriorated.sum()), int(group_base_correct.sum())
            ),
            "base_incorrect_count": int(group_base_incorrect.sum()),
            "base_correct_count": int(group_base_correct.sum()),
            "positive_target_count": positive_count,
            "positive_target_fraction": _fraction(positive_count, count),
            "positive_target_log_probability_increase_fraction": _fraction(
                int(np.sum(group_positive & (delta_log_probability > 0.0))),
                positive_count,
            ),
            "positive_target_margin_increase_fraction": _fraction(
                int(np.sum(group_positive & (delta_margin > 0.0))),
                positive_count,
            ),
            "positive_target_correction_fraction": _fraction(
                int(np.sum(group_positive & corrected)),
                positive_count,
            ),
            "positive_target_deterioration_fraction": _fraction(
                int(np.sum(group_positive & deteriorated)),
                positive_count,
            ),
            "positive_target_deterioration_count": int(
                np.sum(group_positive & deteriorated)
            ),
        }
    return report, cases


def _margin_correction_report(
    *,
    margin_targets: np.ndarray,
    expert_index: int,
    epsilon: float,
    group_labels: np.ndarray,
    base_predictions: np.ndarray,
    new_predictions: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    base_incorrect = base_predictions != labels
    corrected = base_incorrect & (new_predictions == labels)
    positive_margin = margin_targets[:, expert_index] > 0.0
    report: dict[str, Any] = {}
    for group_name in ("head", "medium", "tail"):
        mask = group_labels == group_name
        candidates = mask & base_incorrect & positive_margin
        report[group_name] = {
            "positive_margin_count": int(np.sum(mask & positive_margin)),
            "positive_margin_fraction": _fraction(
                int(np.sum(mask & positive_margin)), int(mask.sum())
            ),
            "positive_margin_base_incorrect_count": int(candidates.sum()),
            "positive_margin_correction_count": int(np.sum(candidates & corrected)),
            "positive_margin_correction_fraction": _fraction(
                int(np.sum(candidates & corrected)), int(candidates.sum())
            ),
            "positive_margin_correction_fraction_of_base_incorrect": _fraction(
                int(np.sum(candidates & corrected)), int(np.sum(mask & base_incorrect))
            ),
        }
    return {
        "expert": EXPERT_ORDER[expert_index],
        "expert_index": expert_index,
        "epsilon": float(epsilon),
        "by_true_group": report,
    }


def _subset_contribution_report(
    *,
    mask: np.ndarray,
    expert_index: int,
    actual_targets: np.ndarray,
    ridge_scores: np.ndarray,
    ridge_weights: np.ndarray,
) -> dict[str, Any]:
    count = int(mask.sum())
    actual_pair = actual_targets[:, expert_index] > actual_targets[:, MIXUP_INDEX]
    ridge_pair = ridge_scores[:, expert_index] > ridge_scores[:, MIXUP_INDEX]
    actual_positive = actual_targets[:, expert_index] > 0.0
    ridge_winner = ridge_scores.argmax(axis=1)
    weight_winner = ridge_weights.argmax(axis=1)
    return {
        "sample_count": count,
        "actual_target_mean": _safe_float(actual_targets[mask, expert_index].mean()),
        "actual_target_positive_count": int(np.sum(mask & actual_positive)),
        "actual_target_positive_fraction": _fraction(
            int(np.sum(mask & actual_positive)), count
        ),
        "actual_target_higher_than_Mixup_count": int(np.sum(mask & actual_pair)),
        "actual_target_higher_than_Mixup_fraction": _fraction(
            int(np.sum(mask & actual_pair)), count
        ),
        "Ridge_score_higher_than_Mixup_count": int(np.sum(mask & ridge_pair)),
        "Ridge_score_higher_than_Mixup_fraction": _fraction(
            int(np.sum(mask & ridge_pair)), count
        ),
        "Ridge_pairwise_order_accuracy": _fraction(
            int(np.sum(mask & (actual_pair == ridge_pair))), count
        ),
        "Ridge_highest_score_rebalanced_count": int(
            np.sum(mask & (ridge_winner == expert_index))
        ),
        "Ridge_highest_score_rebalanced_fraction": _fraction(
            int(np.sum(mask & (ridge_winner == expert_index))), count
        ),
        "Ridge_highest_weight_Mixup_count": int(
            np.sum(mask & (weight_winner == MIXUP_INDEX))
        ),
        "Ridge_highest_weight_Mixup_fraction": _fraction(
            int(np.sum(mask & (weight_winner == MIXUP_INDEX))), count
        ),
        "Ridge_highest_weight_rebalanced_count": int(
            np.sum(mask & (weight_winner == expert_index))
        ),
        "Ridge_highest_weight_rebalanced_fraction": _fraction(
            int(np.sum(mask & (weight_winner == expert_index))), count
        ),
        "mean_Ridge_weight": _safe_float(ridge_weights[mask, expert_index].mean()),
        "mean_Ridge_Mixup_weight": _safe_float(ridge_weights[mask, MIXUP_INDEX].mean()),
    }


def _opportunity_report(
    *,
    mask: np.ndarray,
    labels: np.ndarray,
    group_labels: np.ndarray,
    actual_targets: np.ndarray,
    ridge_scores: np.ndarray,
    ridge_weights: np.ndarray,
    expert_indices: tuple[int, ...],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sample_count": int(mask.sum()),
        "true_group_counts": {
            group_name: int(np.sum(mask & (group_labels == group_name)))
            for group_name in ("head", "medium", "tail")
        },
        "by_rebalanced_expert": {},
        "by_true_group": {},
    }
    for expert_index in expert_indices:
        expert_name = EXPERT_ORDER[expert_index]
        result["by_rebalanced_expert"][expert_name] = _subset_contribution_report(
            mask=mask,
            expert_index=expert_index,
            actual_targets=actual_targets,
            ridge_scores=ridge_scores,
            ridge_weights=ridge_weights,
        )
    for group_name in ("head", "medium", "tail"):
        group_mask = mask & (group_labels == group_name)
        result["by_true_group"][group_name] = {
            "sample_count": int(group_mask.sum()),
            "by_rebalanced_expert": {
                EXPERT_ORDER[expert_index]: _subset_contribution_report(
                    mask=group_mask,
                    expert_index=expert_index,
                    actual_targets=actual_targets,
                    ridge_scores=ridge_scores,
                    ridge_weights=ridge_weights,
                )
                for expert_index in expert_indices
            },
        }
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise Task3FETargetDiagnosticError(f"cannot load {name}: {path}") from exc
    if not isinstance(payload, dict):
        raise Task3FETargetDiagnosticError(f"{name} must be a JSON object")
    return payload


def _load_npz(path: Path, *, name: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: np.array(archive[key]) for key in archive.files}
    except (OSError, ValueError) as exc:
        raise Task3FETargetDiagnosticError(f"cannot load {name}: {path}") from exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Task3FETargetDiagnosticError(message)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    rendered = json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != rendered:
            raise Task3FETargetDiagnosticError(
                f"refusing to overwrite an incompatible output: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered)


def _write_text_once(path: Path, rendered: str) -> None:
    if path.exists():
        if path.read_text() != rendered:
            raise Task3FETargetDiagnosticError(
                f"refusing to overwrite an incompatible output: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered)


def _format_percentage(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.2f}%"


def _format_value(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _validate_task3f_a_inputs(
    *,
    dataset: RestrictedAnalysisDataset,
    config: Mapping[str, Any],
    predictions: Mapping[str, np.ndarray],
    scores: Mapping[str, np.ndarray],
    fold_assignments: Mapping[str, Any],
) -> None:
    _require(
        config.get("task_identifier") == TASK3FA_TASK_IDENTIFIER,
        "Task 3F-A configuration has the wrong task identifier",
    )
    _require(
        config.get("expert_order") == list(EXPERT_ORDER),
        "Task 3F-A expert order is incompatible",
    )
    _require(
        config.get("analyzed_sample_count") == ANALYSIS_SAMPLE_COUNT == dataset.num_samples,
        "Task 3F-A population is not the permitted 6,507 rows",
    )
    _require(
        config.get("permitted_analysis_inner_folds")
        == list(PERMITTED_ANALYSIS_INNER_FOLDS),
        "Task 3F-A permitted folds are incompatible",
    )
    _require(
        config.get("reserved_router_selection_inner_folds")
        == list(RESERVED_ROUTER_SELECTION_INNER_FOLDS),
        "Task 3F-A reserved fold declaration is incompatible",
    )
    restrictions = config.get("data_restrictions", {})
    _require(
        isinstance(restrictions, Mapping)
        and restrictions.get("inner_fold_zero_used") is False
        and restrictions.get("reserved_outer_evaluation_used") is False
        and restrictions.get("original_cifar_test_used") is False,
        "Task 3F-A data restrictions are incompatible",
    )
    _require(
        fold_assignments.get("outer_fold") == OUTER_FOLD
        and fold_assignments.get("permitted_inner_fold_ids")
        == list(PERMITTED_ANALYSIS_INNER_FOLDS)
        and fold_assignments.get("every_sample_held_out_exactly_once") is True,
        "Task 3F-A fold assignment provenance is incompatible",
    )

    expected_ids = np.asarray(dataset.sample_indices, dtype=np.int64)
    expected_folds = np.asarray(dataset.inner_fold_ids, dtype=np.int64)
    expected_labels = np.asarray(dataset.labels, dtype=np.int64)
    for name, artifact in (("held-out predictions", predictions), ("router scores", scores)):
        for key in ("sample_indices", "inner_fold_ids", "labels"):
            _require(key in artifact, f"Task 3F-A {name} lacks {key}")
        _require(
            np.array_equal(np.asarray(artifact["sample_indices"], dtype=np.int64), expected_ids),
            f"Task 3F-A {name} sample IDs are misaligned",
        )
        _require(
            np.array_equal(np.asarray(artifact["inner_fold_ids"], dtype=np.int64), expected_folds),
            f"Task 3F-A {name} inner folds are misaligned",
        )
        _require(
            np.array_equal(np.asarray(artifact["labels"], dtype=np.int64), expected_labels),
            f"Task 3F-A {name} labels are misaligned",
        )
        _require(
            not np.any(expected_folds == 0)
            and set(np.unique(expected_folds)) == set(PERMITTED_ANALYSIS_INNER_FOLDS),
            "Task 3F-A artifacts contain the reserved inner-fold-0 population",
        )

    folds = fold_assignments.get("folds")
    _require(isinstance(folds, Mapping), "Task 3F-A fold assignments lack fold records")
    for fold in PERMITTED_ANALYSIS_INNER_FOLDS:
        record = folds.get(str(fold))
        _require(isinstance(record, Mapping), f"missing Task 3F-A fold {fold} record")
        validation_ids = np.asarray(record.get("validation_sample_ids"), dtype=np.int64)
        training_ids = np.asarray(record.get("training_sample_ids"), dtype=np.int64)
        expected_validation = expected_ids[expected_folds == fold]
        expected_training = expected_ids[expected_folds != fold]
        _require(
            np.array_equal(validation_ids, expected_validation),
            f"Task 3F-A validation IDs for fold {fold} are incompatible",
        )
        _require(
            np.array_equal(training_ids, expected_training),
            f"Task 3F-A training IDs for fold {fold} are incompatible",
        )


def _find_unique_index(values: Any, target: str, *, name: str) -> int:
    array = np.asarray(values).astype(str)
    matches = np.flatnonzero(array == target)
    if len(matches) != 1:
        raise Task3FETargetDiagnosticError(
            f"{name} must contain exactly one {target!r}, found {len(matches)}"
        )
    return int(matches[0])


def _compose_ridge_scores(
    scores: Mapping[str, np.ndarray],
    dataset: RestrictedAnalysisDataset,
) -> tuple[np.ndarray, dict[int, str]]:
    model_ids = np.asarray(scores["model_ids"]).astype(str)
    raw_scores = np.asarray(scores["held_out_scores"])
    if not np.issubdtype(raw_scores.dtype, np.number) or np.iscomplexobj(raw_scores):
        raise Task3FETargetDiagnosticError(
            "held-out Ridge scores must contain real numeric values"
        )
    # Task 3F-A intentionally stores NaN outside each model's validation fold.
    # Those placeholders are checked against the fold mask below; only the
    # selected held-out rows are required to be finite.
    score_tensor = raw_scores.astype(np.float64, copy=False)
    _require(
        score_tensor.ndim == 3
        and score_tensor.shape[0] == len(model_ids)
        and score_tensor.shape[1:] == (dataset.num_samples, len(EXPERT_ORDER)),
        "Task 3F-A held-out Ridge scores have an incompatible shape",
    )
    composed = np.full(
        (dataset.num_samples, len(EXPERT_ORDER)),
        np.nan,
        dtype=np.float64,
    )
    selected: dict[int, str] = {}
    for fold in PERMITTED_ANALYSIS_INNER_FOLDS:
        model_key = RIDGE_SCORE_MODEL_TEMPLATE.format(fold=fold)
        model_index = _find_unique_index(model_ids, model_key, name="Ridge model IDs")
        row = score_tensor[model_index]
        validation_mask = dataset.inner_fold_ids == fold
        finite = np.isfinite(row).all(axis=1)
        _require(
            np.array_equal(finite, validation_mask),
            f"saved Ridge scores for {model_key} are not held out only on fold {fold}",
        )
        composed[validation_mask] = row[validation_mask]
        selected[fold] = model_key
    _require(np.isfinite(composed).all(), "composed Ridge scores are incomplete")
    return composed, selected


def _load_highlighted_weights(
    predictions: Mapping[str, np.ndarray],
    dataset: RestrictedAnalysisDataset,
) -> np.ndarray:
    configuration_ids = np.asarray(predictions["adaptive_configuration_ids"]).astype(str)
    configuration_index = _find_unique_index(
        configuration_ids,
        HIGHLIGHTED_CONFIGURATION_ID,
        name="adaptive configuration IDs",
    )
    weights = _as_numeric(
        predictions["adaptive_weights"][configuration_index],
        name="saved highlighted Ridge weights",
    )
    _require(
        weights.shape == (dataset.num_samples, len(EXPERT_ORDER)),
        "saved highlighted Ridge weights have an incompatible shape",
    )
    _require(
        np.all(weights >= 0.0)
        and np.allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=2e-6),
        "saved highlighted Ridge weights are not convex",
    )
    return weights / weights.sum(axis=1, keepdims=True)


def _build_perturbation_reports(
    *,
    dataset: RestrictedAnalysisDataset,
    class_counts: np.ndarray,
    actual_targets: np.ndarray,
    margin_targets: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[tuple[int, float], np.ndarray]]:
    logits = np.asarray(dataset.logits, dtype=np.float64)
    labels = np.asarray(dataset.labels, dtype=np.int64)
    sample_ids = np.asarray(dataset.sample_indices, dtype=np.int64)
    group_labels = _canonical_group_labels(labels, class_counts)
    uniform_weights = convex_weight_perturbation(dataset.num_samples, 0, 0.0)
    uniform_logits = combine_weighted_logits(logits, uniform_weights)
    uniform_predictions = uniform_logits.argmax(axis=1).astype(np.int64)
    uniform_log_probs = _true_log_probabilities(uniform_logits, labels)
    uniform_margins, _ = _true_classification_margins(uniform_logits, labels)
    reports: list[dict[str, Any]] = []
    predictions_by_perturbation: dict[tuple[int, float], np.ndarray] = {}
    for expert_index, expert_name in enumerate(EXPERT_ORDER):
        for epsilon in EPSILONS:
            weights = convex_weight_perturbation(
                dataset.num_samples,
                expert_index,
                epsilon,
            )
            perturbed_logits = combine_weighted_logits(logits, weights)
            perturbed_predictions = perturbed_logits.argmax(axis=1).astype(np.int64)
            predictions_by_perturbation[(expert_index, epsilon)] = perturbed_predictions
            delta_log_probs = _true_log_probabilities(perturbed_logits, labels) - uniform_log_probs
            perturbed_margins, _ = _true_classification_margins(
                perturbed_logits, labels
            )
            delta_margins = perturbed_margins - uniform_margins
            by_group, cases = _group_event_report(
                mask=np.ones(dataset.num_samples, dtype=bool),
                group_labels=group_labels,
                target=actual_targets,
                delta_log_probability=delta_log_probs,
                delta_margin=delta_margins,
                base_predictions=uniform_predictions,
                new_predictions=perturbed_predictions,
                labels=labels,
                sample_ids=sample_ids,
                expert_index=expert_index,
                epsilon=epsilon,
            )
            metrics = _metric_bundle(labels, perturbed_predictions, class_counts)
            reports.append(
                {
                    "expert": expert_name,
                    "expert_index": expert_index,
                    "epsilon": float(epsilon),
                    "weights": weights[0].tolist(),
                    "metrics": metrics,
                    "by_true_group": by_group,
                    "positive_target_deterioration_cases": cases,
                }
            )
    baseline_metrics = _metric_bundle(labels, uniform_predictions, class_counts)
    reports.insert(
        0,
        {
            "expert": "uniform",
            "expert_index": None,
            "epsilon": 0.0,
            "weights": uniform_weights[0].tolist(),
            "metrics": baseline_metrics,
            "prediction_count": int(len(uniform_predictions)),
        },
    )
    return reports, predictions_by_perturbation


def _build_margin_correction_reports(
    *,
    margin_targets: np.ndarray,
    labels: np.ndarray,
    group_labels: np.ndarray,
    base_predictions: np.ndarray,
    predictions_by_perturbation: Mapping[tuple[int, float], np.ndarray],
) -> list[dict[str, Any]]:
    reports = []
    for expert_index, _ in enumerate(EXPERT_ORDER):
        for epsilon in EPSILONS:
            reports.append(
                _margin_correction_report(
                    margin_targets=margin_targets,
                    expert_index=expert_index,
                    epsilon=epsilon,
                    group_labels=group_labels,
                    base_predictions=base_predictions,
                    new_predictions=predictions_by_perturbation[(expert_index, epsilon)],
                    labels=labels,
                )
            )
    return reports


def _render_summary(results: Mapping[str, Any]) -> str:
    population = results["population"]
    investigation_a = results["investigation_a"]
    actual = investigation_a["actual_target_report"]
    predicted = investigation_a["ridge_predicted_score_report"]
    comparison = investigation_a["target_vs_ridge"]
    tail_misranking = investigation_a["tail_rebalanced_misranking"]
    lines = [
        "# Task 3F-E — Supervised contribution target diagnostics",
        "",
        "Retrospective diagnostics on the frozen outer-fold-0 / inner-folds-1–3 "
        "OOF development population. No router or expert was retrained.",
        "",
        "## Protocol and safeguards",
        "",
        f"- Population: {population['sample_count']} images; Head/Medium/Tail = "
        f"{population['true_group_counts']['head']} / "
        f"{population['true_group_counts']['medium']} / "
        f"{population['true_group_counts']['tail']}",
        "- Only Task 3C OOF logits from inner folds 1–3 were loaded.",
        "- Inner fold 0, the reserved outer-evaluation population, the original "
        "CIFAR-100 test set, and original full-data expert predictions were excluded.",
        "- Labels appear only in retrospective target, margin, group, and outcome calculations.",
        "- The saved Task 3F-A Ridge scores and highlighted weights were read-only inputs.",
        "",
        "## Investigation A — target and Ridge score distributions",
        "",
        "| Group | Expert | Target mean | Target median | Target positive | Target highest | Ridge mean | Ridge highest |",
        "|:--|:--|--:|--:|--:|--:|--:|--:|",
    ]
    for group_name in ("head", "medium", "tail"):
        for expert in EXPERT_ORDER:
            lines.append(
                f"| {group_name.title()} | {expert} | "
                f"{actual[group_name]['mean'][expert]:.4f} | "
                f"{actual[group_name]['median'][expert]:.4f} | "
                f"{actual[group_name]['positive_percentage'][expert]:.2f}% | "
                f"{actual[group_name]['highest_expert_percentage'][expert]:.2f}% | "
                f"{predicted[group_name]['mean'][expert]:.4f} | "
                f"{100.0 * predicted[group_name]['highest_expert_fraction'][expert]:.2f}% |"
            )
    lines.extend(
        [
            "",
            "| Group | Target/Ridge top-expert agreement | Score MSE | Score MAE | Pearson |",
            "|:--|--:|--:|--:|--:|",
        ]
    )
    for group_name in ("head", "medium", "tail"):
        row = comparison[group_name]
        lines.append(
            f"| {group_name.title()} | {100.0 * row['highest_expert_agreement_fraction']:.2f}% | "
            f"{row['mse']:.4f} | {row['mae']:.4f} | "
            f"{_format_value(row['pearson_correlation'])} |"
        )
    lines.extend(
        [
            "",
            "### Tail cases where a rebalanced target exceeds Mixup",
            "",
            "| Expert | Actual target higher than Mixup | Ridge still predicts Mixup highest | Ridge pairwise rebalanced score higher |",
            "|:--|--:|--:|--:|",
        ]
    )
    for expert in ("LAL", "BalancedSoftmax"):
        row = tail_misranking[expert]
        lines.append(
            f"| {expert} | {row['actual_higher_than_Mixup_count']} "
            f"({ _format_percentage(row['actual_higher_than_Mixup_fraction_of_Tail'])}) | "
            f"{row['Ridge_predicted_Mixup_highest_count']} "
            f"({_format_percentage(row['Ridge_predicted_Mixup_highest_fraction'])}) | "
            f"{row['Ridge_predicted_rebalanced_higher_count']} "
            f"({_format_percentage(row['Ridge_predicted_rebalanced_higher_fraction'])}) |"
        )
    lines.extend(
        [
            "",
            "## Investigation B — fixed convex perturbations",
            "",
            "Positive-target rates below are conditional on a positive target. "
            "Correction means an incorrect uniform prediction becomes correct; "
            "deterioration means a correct uniform prediction becomes incorrect.",
            "",
            "| Expert | ε | Group | Δ true log p | Δ true margin | Correction | Deterioration | Positive target → correction | Positive target → deterioration |",
            "|:--|--:|:--|--:|--:|--:|--:|--:|--:|",
        ]
    )
    for row in results["investigation_b"]["perturbation_reports"]:
        if row["expert"] == "uniform":
            continue
        for group_name in ("head", "medium", "tail"):
            group = row["by_true_group"][group_name]
            lines.append(
                f"| {row['expert']} | {row['epsilon']:.2f} | {group_name.title()} | "
                f"{_format_value(group['mean_change_true_log_probability'])} | "
                f"{_format_value(group['mean_change_true_margin'])} | "
                f"{_format_percentage(group['correction_rate_among_base_incorrect'])} | "
                f"{_format_percentage(group['deterioration_rate_among_base_correct'])} | "
                f"{_format_percentage(group['positive_target_correction_fraction'])} | "
                f"{_format_percentage(group['positive_target_deterioration_fraction'])} |"
            )
    lines.extend(
        [
            "",
            "## Investigation C — classification-margin contribution",
            "",
            "The margin target uses the uniform ensemble's strongest incorrect "
            "competitor. Final perturbed margins recompute the strongest incorrect "
            "class, so a positive local contribution is not a guarantee of correction.",
            "",
            "| Group | Target top / Margin top agreement | Target Mixup highest | Margin Mixup highest | Both Mixup highest |",
            "|:--|--:|--:|--:|--:|",
        ]
    )
    margin_comparison = results["investigation_c"]["target_vs_margin"]
    for group_name in ("head", "medium", "tail"):
        row = margin_comparison[group_name]
        lines.append(
            f"| {group_name.title()} | {100.0 * row['highest_expert_agreement_fraction']:.2f}% | "
            f"{100.0 * row['actual_target_highest_Mixup_fraction']:.2f}% | "
            f"{100.0 * row['margin_highest_Mixup_fraction']:.2f}% | "
            f"{100.0 * row['both_highest_Mixup_fraction']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Investigation D — rebalanced-expert opportunities and losses",
            "",
            "These groups are label-dependent retrospective masks. They were not "
            "used as inference-time features or router-training targets.",
            "",
            "| Event | Group | n | Correcting target positive | Target > Mixup | Ridge pairwise order correct | Ridge Mixup highest weight |",
            "|:--|:--|--:|--:|--:|--:|--:|",
        ]
    )
    for event_name, event_reports in (
        ("LAL correcting", results["investigation_d"]["correcting_opportunities"]["LAL"]),
        (
            "BalancedSoftmax correcting",
            results["investigation_d"]["correcting_opportunities"]["BalancedSoftmax"],
        ),
        ("LAL damaging", results["investigation_d"]["damaging_opportunities"]["LAL"]),
        (
            "BalancedSoftmax damaging",
            results["investigation_d"]["damaging_opportunities"]["BalancedSoftmax"],
        ),
    ):
        expert_name = "LAL" if event_name.startswith("LAL") else "BalancedSoftmax"
        for group_name in ("overall", "head", "medium", "tail"):
            row = (
                event_reports["by_rebalanced_expert"][expert_name]
                if group_name == "overall"
                else event_reports["by_true_group"][group_name]["by_rebalanced_expert"][expert_name]
            )
            label = "All" if group_name == "overall" else group_name.title()
            lines.append(
                f"| {event_name} | {label} | {row['sample_count']} | "
                f"{_format_percentage(row['actual_target_positive_fraction'])} | "
                f"{_format_percentage(row['actual_target_higher_than_Mixup_fraction'])} | "
                f"{_format_percentage(row['Ridge_pairwise_order_accuracy'])} | "
                f"{_format_percentage(row['Ridge_highest_weight_Mixup_fraction'])} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "A target is a local true-class log-probability derivative, whereas "
            "classification outcomes depend on the full competing-class geometry. "
            "These measurements can show target preference, prediction error, and "
            "classification mismatch on this development population, but they do "
            "not establish that a different target or router would improve an "
            "independent evaluation.",
            "",
        ]
    )
    return "\n".join(lines)


def _build_margin_comparison(
    actual_targets: np.ndarray,
    margin_targets: np.ndarray,
    labels: np.ndarray,
    class_counts: np.ndarray,
) -> dict[str, Any]:
    groups = _canonical_group_labels(labels, class_counts)
    target_winners = actual_targets.argmax(axis=1)
    margin_winners = margin_targets.argmax(axis=1)
    result: dict[str, Any] = {}
    for group_name in ("head", "medium", "tail"):
        mask = groups == group_name
        target_group = actual_targets[mask]
        margin_group = margin_targets[mask]
        error = margin_group - target_group
        result[group_name] = {
            "sample_count": int(mask.sum()),
            "mse": float(np.mean(error**2)),
            "mae": float(np.mean(np.abs(error))),
            "pearson_correlation": _pearson(target_group.ravel(), margin_group.ravel()),
            "highest_expert_agreement_count": int(
                np.sum(target_winners[mask] == margin_winners[mask])
            ),
            "highest_expert_agreement_fraction": float(
                np.mean(target_winners[mask] == margin_winners[mask])
            ),
            "actual_target_highest_Mixup_fraction": float(
                np.mean(target_winners[mask] == MIXUP_INDEX)
            ),
            "margin_highest_Mixup_fraction": float(
                np.mean(margin_winners[mask] == MIXUP_INDEX)
            ),
            "both_highest_Mixup_fraction": float(
                np.mean(
                    (target_winners[mask] == MIXUP_INDEX)
                    & (margin_winners[mask] == MIXUP_INDEX)
                )
            ),
            "target_mean": {
                expert: float(target_group[:, index].mean())
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "margin_mean": {
                expert: float(margin_group[:, index].mean())
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "margin_higher_than_Mixup_fraction": {
                expert: float(
                    np.mean(margin_group[:, index] > margin_group[:, MIXUP_INDEX])
                )
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "margin_positive_fraction": {
                expert: float(np.mean(margin_group[:, index] > 0.0))
                for index, expert in enumerate(EXPERT_ORDER)
            },
        }
    return result


def run_task3f_target_diagnostics(
    *,
    data_root: str | Path = "./data",
    oof_directory: str | Path = "artifacts/oof/task3c_oof",
    ridge_directory: str | Path = "artifacts/oof/task3f_ridge",
    output_directory: str | Path = "artifacts/oof/task3f_target_diagnostics",
    project_root: str | Path | None = None,
) -> dict[str, Path]:
    """Run Task 3F-E against the frozen OOF and Task 3F-A artifacts."""
    project_root_path = Path(
        project_root or Path(__file__).resolve().parents[1]
    ).resolve()
    source_directory = Path(oof_directory).resolve()
    ridge_path = Path(ridge_directory).resolve()
    output_path = Path(output_directory).resolve()
    if output_path in (source_directory, ridge_path):
        raise Task3FETargetDiagnosticError(
            "output directory must be separate from existing input artifacts"
        )

    ridge_files = {
        name: ridge_path / filename
        for name, filename in {
            "experiment_config": "experiment_config.json",
            "ridge_results": "ridge_results.json",
            "held_out_predictions": "held_out_predictions.npz",
            "router_scores": "router_scores.npz",
            "fold_assignments": "fold_assignments.json",
        }.items()
    }
    for name, path in ridge_files.items():
        if not path.exists():
            raise Task3FETargetDiagnosticError(
                f"missing Task 3F-A artifact {name}: {path}"
            )
    ridge_hashes_before = {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in ridge_files.items()
    }
    config = _load_json(ridge_files["experiment_config"], name="Task 3F-A configuration")
    # Loading the results file is an explicit provenance check: the selected
    # model must be part of the completed Task 3F-A result, not an invented ID.
    ridge_results = _load_json(ridge_files["ridge_results"], name="Task 3F-A results")
    predictions = _load_npz(
        ridge_files["held_out_predictions"],
        name="Task 3F-A held-out predictions",
    )
    scores = _load_npz(ridge_files["router_scores"], name="Task 3F-A router scores")
    fold_assignments = _load_json(
        ridge_files["fold_assignments"],
        name="Task 3F-A fold assignments",
    )
    _require(
        any(
            row.get("model_key") == RIDGE_SCORE_MODEL_TEMPLATE.format(fold=fold)
            for row in ridge_results.get("model_fits", [])
            for fold in PERMITTED_ANALYSIS_INNER_FOLDS
        ),
        "Task 3F-A results do not contain the required held-out Ridge models",
    )

    manager = NestedOOFFoldManager.from_canonical_training_data(
        data_root,
        seed=FOLD_GENERATION_SEED,
        outer_folds=5,
        inner_folds=4,
        expert_order=EXPERT_ORDER,
    )
    dataset, source_files = load_restricted_analysis_dataset(source_directory, manager)
    _validate_task3f_a_inputs(
        dataset=dataset,
        config=config,
        predictions=predictions,
        scores=scores,
        fold_assignments=fold_assignments,
    )
    class_counts = _validate_class_counts(
        manager.canonical_class_counts,
        num_classes=dataset.logits.shape[2],
    )
    ridge_scores, selected_score_models = _compose_ridge_scores(scores, dataset)
    ridge_weights = _load_highlighted_weights(predictions, dataset)

    actual_targets = compute_contribution_targets(dataset.logits, dataset.labels)
    margin_targets = compute_margin_contributions(dataset.logits, dataset.labels)
    group_labels = _canonical_group_labels(dataset.labels, class_counts)
    perturbation_reports, predictions_by_perturbation = _build_perturbation_reports(
        dataset=dataset,
        class_counts=class_counts,
        actual_targets=actual_targets,
        margin_targets=margin_targets,
    )
    uniform_report = perturbation_reports[0]
    uniform_predictions = combine_weighted_logits(
        dataset.logits,
        convex_weight_perturbation(dataset.num_samples, 0, 0.0),
    ).argmax(axis=1).astype(np.int64)
    margin_correction_reports = _build_margin_correction_reports(
        margin_targets=margin_targets,
        labels=dataset.labels,
        group_labels=group_labels,
        base_predictions=uniform_predictions,
        predictions_by_perturbation=predictions_by_perturbation,
    )

    base_incorrect = uniform_predictions != dataset.labels
    base_correct = ~base_incorrect
    correcting_masks = {
        expert: base_incorrect
        & np.any(
            np.stack(
                [
                    predictions_by_perturbation[(index, epsilon)]
                    == dataset.labels
                    for epsilon in EPSILONS
                ],
                axis=1,
            ),
            axis=1,
        )
        for expert, index in (
            ("LAL", EXPERT_ORDER.index("LAL")),
            ("BalancedSoftmax", EXPERT_ORDER.index("BalancedSoftmax")),
        )
    }
    damaging_masks = {
        expert: base_correct
        & np.any(
            np.stack(
                [
                    predictions_by_perturbation[(index, epsilon)]
                    != dataset.labels
                    for epsilon in EPSILONS
                ],
                axis=1,
            ),
            axis=1,
        )
        for expert, index in (
            ("LAL", EXPERT_ORDER.index("LAL")),
            ("BalancedSoftmax", EXPERT_ORDER.index("BalancedSoftmax")),
        )
    }
    correcting_reports = {
        expert: _opportunity_report(
            mask=mask,
            labels=dataset.labels,
            group_labels=group_labels,
            actual_targets=actual_targets,
            ridge_scores=ridge_scores,
            ridge_weights=ridge_weights,
            expert_indices=(EXPERT_ORDER.index(expert),),
        )
        for expert, mask in correcting_masks.items()
    }
    damaging_reports = {
        expert: _opportunity_report(
            mask=mask,
            labels=dataset.labels,
            group_labels=group_labels,
            actual_targets=actual_targets,
            ridge_scores=ridge_scores,
            ridge_weights=ridge_weights,
            expert_indices=(EXPERT_ORDER.index(expert),),
        )
        for expert, mask in damaging_masks.items()
    }

    current_ridge_hashes = {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in ridge_files.items()
    }
    _require(
        current_ridge_hashes == ridge_hashes_before,
        "a Task 3F-A artifact changed during diagnostics",
    )
    source_hashes = {
        name: {
            "path": str(details["path"]),
            "sha256": str(details["sha256"]),
        }
        for name, details in source_files.items()
    }
    current_source_hashes = {
        name: {"path": str(details["path"]), "sha256": _sha256_file(Path(details["path"]))}
        for name, details in source_files.items()
    }
    _require(
        current_source_hashes == source_hashes,
        "a Task 3C source artifact changed during diagnostics",
    )
    results: dict[str, Any] = {
        "task_identifier": TASK3FE_TASK_IDENTIFIER,
        "schema_version": TASK3FE_SCHEMA_VERSION,
        "source_git_commit": _git_commit(project_root_path),
        "highlighted_configuration_id": HIGHLIGHTED_CONFIGURATION_ID,
        "population": {
            "sample_count": dataset.num_samples,
            "true_group_counts": {
                group_name: int(np.sum(group_labels == group_name))
                for group_name in ("head", "medium", "tail")
            },
            "inner_fold_counts": {
                str(fold): int(np.sum(dataset.inner_fold_ids == fold))
                for fold in PERMITTED_ANALYSIS_INNER_FOLDS
            },
        },
        "provenance": {
            "dataset": DATASET_NAME,
            "imbalance_ratio": IMBALANCE_RATIO,
            "canonical_population_size": CANONICAL_POPULATION_SIZE,
            "training_seed": TRAINING_SEED,
            "fold_generation_seed": FOLD_GENERATION_SEED,
            "outer_fold": OUTER_FOLD,
            "permitted_inner_folds": list(PERMITTED_ANALYSIS_INNER_FOLDS),
            "reserved_inner_folds": list(RESERVED_ROUTER_SELECTION_INNER_FOLDS),
            "outer_evaluation_size": OUTER_EVALUATION_SIZE,
            "expert_order": list(EXPERT_ORDER),
            "analyzed_sample_count": dataset.num_samples,
            "inner_fold_zero_used": False,
            "reserved_outer_evaluation_used": False,
            "original_cifar_test_used": False,
            "original_full_data_predictions_used": False,
            "experts_retrained": False,
            "router_refit": False,
            "oracle_weights_used": False,
            "source_oof_logits_modified": False,
            "expert_weights_modified": False,
            "labels_used_only_for_retrospective_diagnostics": True,
            "task3c_oof_artifacts": source_hashes,
            "task3f_a_artifacts": ridge_hashes_before,
            "selected_ridge_score_models_by_fold": selected_score_models,
        },
        "definitions": {
            "class_groups_source": "scripts.base_trainer.compute_class_groups",
            "supervised_target": (
                "(z_e_y-zbar_y)-sum_c softmax(zbar)_c*(z_e_c-zbar_c)"
            ),
            "margin_target": (
                "(z_e_y-z_e_cstar)-(zbar_y-zbar_cstar), with cstar from uniform ensemble"
            ),
            "uniform_weights": [0.25, 0.25, 0.25, 0.25],
            "perturbation_formula": "(1-epsilon)*uniform + epsilon*onehot(expert)",
            "epsilons": list(EPSILONS),
            "classification_margin_for_perturbation": (
                "true-class logit minus the strongest incorrect logit of that perturbed ensemble"
            ),
            "ridge_score_source": (
                "saved Task 3F-A confidence-only alpha=1000 gamma=1 held-out scores"
            ),
            "ridge_weight_source": (
                "saved Task 3F-A highlighted adaptive configuration weights"
            ),
        },
        "investigation_a": {
            "actual_target_report": _matrix_report(
                actual_targets,
                dataset.labels,
                class_counts,
                value_name="actual contribution targets",
            ),
            "ridge_predicted_score_report": _matrix_report(
                ridge_scores,
                dataset.labels,
                class_counts,
                value_name="Ridge predicted contribution scores",
            ),
            "target_vs_ridge": _target_prediction_comparison(
                actual_targets,
                ridge_scores,
                dataset.labels,
                class_counts,
            ),
            "tail_rebalanced_misranking": _tail_rebalanced_misranking(
                actual_targets,
                ridge_scores,
                dataset.labels,
                class_counts,
            ),
        },
        "investigation_b": {
            "uniform_report": uniform_report,
            "perturbation_reports": perturbation_reports,
        },
        "investigation_c": {
            "strongest_incorrect_class_definition": "argmax of uniform ensemble logits after masking y",
            "margin_target_report": _matrix_report(
                margin_targets,
                dataset.labels,
                class_counts,
                value_name="classification-margin contributions",
            ),
            "target_vs_margin": _build_margin_comparison(
                actual_targets,
                margin_targets,
                dataset.labels,
                class_counts,
            ),
            "positive_margin_correction": margin_correction_reports,
        },
        "investigation_d": {
            "event_definition": (
                "correcting = uniform incorrect and a rebalanced expert is correct at any fixed epsilon; "
                "damaging = uniform correct and that expert is incorrect at any fixed epsilon"
            ),
            "correcting_opportunities": correcting_reports,
            "damaging_opportunities": damaging_reports,
            "correcting_union_count": int(
                np.sum(correcting_masks["LAL"] | correcting_masks["BalancedSoftmax"])
            ),
            "damaging_union_count": int(
                np.sum(damaging_masks["LAL"] | damaging_masks["BalancedSoftmax"])
            ),
        },
    }
    summary = _render_summary(results)
    _write_json_once(output_path / "diagnostic_results.json", results)
    _write_text_once(output_path / "summary.md", summary)
    return {
        "diagnostic_results": output_path / "diagnostic_results.json",
        "summary": output_path / "summary.md",
    }


__all__ = [
    "EPSILONS",
    "EXPERT_ORDER",
    "HIGHLIGHTED_CONFIGURATION_ID",
    "Task3FETargetDiagnosticError",
    "compute_classification_margin_contributions",
    "compute_contribution_targets",
    "compute_margin_contributions",
    "compute_strongest_incorrect_classes",
    "convex_weight_perturbation",
    "perturb_uniform_weights",
    "run_task3f_target_diagnostics",
]
