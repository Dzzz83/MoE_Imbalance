"""Task 3F-C: diagnostics for inference-time Tail routing signals.

The module reads the validated Task 3C OOF logits and the frozen Task 3F-A
highlighted weights.  It does not fit a router, alter weights, train experts,
or use labels to construct any signal.  Labels are used only after the
prediction/confidence signals have been formed, for retrospective grouping and
correctness reports.
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
    PERMITTED_ANALYSIS_INNER_FOLDS,
    combine_weighted_logits,
    extract_features,
)


TASK3FC_SCHEMA_VERSION = "task3f_tail_signal_diagnostics.v1"
TASK3FC_TASK_IDENTIFIER = "Task 3F-C"
TASK3FA_TASK_IDENTIFIER = "Task 3F-A"
TASK3FB_TASK_IDENTIFIER = "Task 3F-B"
DATASET_NAME = "CIFAR-100-LT"
ANALYSIS_SAMPLE_COUNT = 6_507
NUM_CLASSES = 100
OUTER_FOLD = 0
HIGHLIGHTED_CONFIGURATION_ID = (
    "adaptive_confidence_only_alpha1000_gamma1_temperature2_lambda0p75"
)
GROUP_NAMES = ("head", "medium", "tail")
REBALANCED_EXPERTS = ("LAL", "BalancedSoftmax")
WEIGHT_SUM_ATOL = 2e-6


class Task3FCTailDiagnosticError(OOFProtocolError):
    """Raised when Task 3F-C inputs or signal calculations are invalid."""


def _as_numeric(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise Task3FCTailDiagnosticError(f"{name} must contain real numeric values")
    if not np.isfinite(array).all():
        raise Task3FCTailDiagnosticError(f"{name} contains non-finite values")
    return array.astype(np.float64, copy=False)


def _as_integer_vector(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise Task3FCTailDiagnosticError(f"{name} must be one-dimensional")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise Task3FCTailDiagnosticError(f"{name} must contain integer values")
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        raise Task3FCTailDiagnosticError(f"{name} must contain finite integers")
    return array.astype(np.int64, copy=False)


def _validate_class_counts(class_counts: Any, *, num_classes: int | None = None) -> np.ndarray:
    counts = _as_integer_vector(class_counts, name="class_counts")
    if num_classes is not None and len(counts) != num_classes:
        raise Task3FCTailDiagnosticError(
            f"class_counts must contain {num_classes} classes"
        )
    if np.any(counts < 1):
        raise Task3FCTailDiagnosticError("class_counts must be positive")
    return counts


def _validate_expert_matrix(
    value: Any,
    *,
    name: str,
    num_samples: int | None = None,
    integer: bool = False,
) -> np.ndarray:
    array = _as_numeric(value, name=name)
    if array.ndim != 2 or array.shape[1] != len(EXPERT_ORDER):
        raise Task3FCTailDiagnosticError(
            f"{name} must have shape (samples, {len(EXPERT_ORDER)})"
        )
    if num_samples is not None and array.shape[0] != num_samples:
        raise Task3FCTailDiagnosticError(f"{name} is not sample-aligned")
    if integer and not np.equal(array, np.floor(array)).all():
        raise Task3FCTailDiagnosticError(f"{name} must contain integer class IDs")
    return array.astype(np.int64 if integer else np.float64, copy=False)


def _validate_weights(weights: Any, *, num_samples: int) -> np.ndarray:
    array = _validate_expert_matrix(
        weights, name="routing weights", num_samples=num_samples
    )
    if np.any(array < 0.0) or not np.allclose(
        array.sum(axis=1), 1.0, rtol=0.0, atol=WEIGHT_SUM_ATOL
    ):
        raise Task3FCTailDiagnosticError(
            "routing weights must be non-negative and sum to one"
        )
    return array


def _validate_confidences(confidences: Any, *, num_samples: int) -> np.ndarray:
    array = _validate_expert_matrix(
        confidences, name="expert confidences", num_samples=num_samples
    )
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise Task3FCTailDiagnosticError("expert confidences must lie in [0, 1]")
    return array


def validate_signal_alignment(
    *,
    expected_sample_indices: Any,
    expected_inner_fold_ids: Any,
    expected_labels: Any,
    artifact_sample_indices: Any,
    artifact_inner_fold_ids: Any,
    artifact_labels: Any,
    expert_predictions: Any,
    confidences: Any,
    weights: Any | None = None,
    expert_order: Any = EXPERT_ORDER,
) -> None:
    """Validate exact sample alignment, fold restrictions, and expert order."""
    if tuple(expert_order) != EXPERT_ORDER:
        raise Task3FCTailDiagnosticError(
            "expert order must be CE, LAL, BalancedSoftmax, Mixup"
        )
    expected_ids = _as_integer_vector(
        expected_sample_indices, name="expected sample IDs"
    )
    expected_folds = _as_integer_vector(
        expected_inner_fold_ids, name="expected inner-fold IDs"
    )
    expected_labels_array = _as_integer_vector(
        expected_labels, name="expected labels"
    )
    artifact_ids = _as_integer_vector(
        artifact_sample_indices, name="artifact sample IDs"
    )
    artifact_folds = _as_integer_vector(
        artifact_inner_fold_ids, name="artifact inner-fold IDs"
    )
    artifact_labels_array = _as_integer_vector(
        artifact_labels, name="artifact labels"
    )
    arrays = (
        expected_folds,
        expected_labels_array,
        artifact_ids,
        artifact_folds,
        artifact_labels_array,
    )
    if any(len(array) != len(expected_ids) for array in arrays):
        raise Task3FCTailDiagnosticError("alignment arrays have inconsistent lengths")
    if len(np.unique(artifact_ids)) != len(artifact_ids):
        raise Task3FCTailDiagnosticError("artifact sample IDs contain duplicates")
    if np.any(artifact_folds == 0):
        raise Task3FCTailDiagnosticError(
            "artifact contains the reserved inner-fold-0 population"
        )
    if not np.all(np.isin(artifact_folds, PERMITTED_ANALYSIS_INNER_FOLDS)):
        raise Task3FCTailDiagnosticError("artifact contains an incompatible inner fold")
    if not np.array_equal(expected_ids, artifact_ids):
        raise Task3FCTailDiagnosticError("sample IDs are not exactly aligned")
    if not np.array_equal(expected_folds, artifact_folds):
        raise Task3FCTailDiagnosticError("inner-fold IDs are not exactly aligned")
    if not np.array_equal(expected_labels_array, artifact_labels_array):
        raise Task3FCTailDiagnosticError("labels are not exactly aligned")
    _validate_expert_matrix(
        expert_predictions, name="expert predictions", num_samples=len(expected_ids), integer=True
    )
    _validate_confidences(confidences, num_samples=len(expected_ids))
    if weights is not None:
        _validate_weights(weights, num_samples=len(expected_ids))


def assign_class_groups(class_ids: Any, class_counts: Any) -> np.ndarray:
    """Map class IDs to canonical ``head``, ``medium`` or ``tail`` labels."""
    ids = _as_integer_vector(class_ids, name="class IDs")
    counts = _validate_class_counts(class_counts)
    if np.any(ids < 0) or np.any(ids >= len(counts)):
        raise Task3FCTailDiagnosticError("class IDs fall outside class_counts")
    groups = compute_class_groups(counts)
    result = np.full(len(ids), "", dtype="U6")
    for name in GROUP_NAMES:
        result[np.isin(ids, groups[name])] = name
    if np.any(result == ""):
        raise Task3FCTailDiagnosticError("some class IDs have no canonical group")
    return result


def _distribution(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "std": None,
            "minimum": None,
            "maximum": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
        }
    percentiles = np.percentile(array, [10, 25, 75, 90])
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std()),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "p10": float(percentiles[0]),
        "p25": float(percentiles[1]),
        "p75": float(percentiles[2]),
        "p90": float(percentiles[3]),
    }


def _correctness(
    predictions: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    count = int(mask.sum())
    correct_count = int(np.sum(mask & (predictions == labels)))
    return {
        "count": count,
        "correct_count": correct_count,
        "correct_fraction": None if count == 0 else float(correct_count / count),
    }


def confidence_diagnostics(
    confidences: Any,
    expert_predictions: Any,
    labels: Any,
    class_counts: Any,
) -> dict[str, Any]:
    """Report raw confidence distributions and retrospective correctness splits."""
    predictions = _validate_expert_matrix(
        expert_predictions, name="expert predictions", integer=True
    )
    labels_array = _as_integer_vector(labels, name="labels")
    if len(labels_array) != len(predictions):
        raise Task3FCTailDiagnosticError("labels and expert predictions are misaligned")
    counts = _validate_class_counts(class_counts)
    if np.any(labels_array < 0) or np.any(labels_array >= len(counts)):
        raise Task3FCTailDiagnosticError("labels fall outside class_counts")
    confidence_array = _validate_confidences(
        confidences, num_samples=len(labels_array)
    )
    true_groups = assign_class_groups(labels_array, counts)
    by_true_group: dict[str, Any] = {}
    for group_name in GROUP_NAMES:
        mask = true_groups == group_name
        per_expert: dict[str, Any] = {}
        for index, expert in enumerate(EXPERT_ORDER):
            correct_mask = mask & (predictions[:, index] == labels_array)
            incorrect_mask = mask & (predictions[:, index] != labels_array)
            per_expert[expert] = {
                **_distribution(confidence_array[mask, index]),
                "correct_count": int(correct_mask.sum()),
                "incorrect_count": int(incorrect_mask.sum()),
                "correct_fraction": None
                if not mask.any()
                else float(correct_mask.sum() / mask.sum()),
                "correct_confidence": _distribution(
                    confidence_array[correct_mask, index]
                ),
                "incorrect_confidence": _distribution(
                    confidence_array[incorrect_mask, index]
                ),
            }
        by_true_group[group_name] = {
            "sample_count": int(mask.sum()),
            "per_expert": per_expert,
        }

    tail_mask = true_groups == "tail"
    tail_pairwise: dict[str, Any] = {}
    for expert in REBALANCED_EXPERTS:
        rebalanced_index = EXPERT_ORDER.index(expert)
        rebalanced_correct = predictions[:, rebalanced_index] == labels_array
        mixup_correct = predictions[:, 3] == labels_array
        categories = {
            "rebalanced_correct_mixup_wrong": tail_mask & rebalanced_correct & ~mixup_correct,
            "mixup_correct_rebalanced_wrong": tail_mask & mixup_correct & ~rebalanced_correct,
            "both_correct": tail_mask & rebalanced_correct & mixup_correct,
            "both_wrong": tail_mask & ~rebalanced_correct & ~mixup_correct,
        }
        category_report: dict[str, Any] = {}
        for category, mask in categories.items():
            delta = confidence_array[:, rebalanced_index] - confidence_array[:, 3]
            category_report[category] = {
                "count": int(mask.sum()),
                "rebalanced_confidence": _distribution(
                    confidence_array[mask, rebalanced_index]
                ),
                "mixup_confidence": _distribution(confidence_array[mask, 3]),
                "rebalanced_minus_mixup_confidence": _distribution(delta[mask]),
                "fraction_rebalanced_confidence_above_mixup": None
                if not mask.any()
                else float(np.mean(delta[mask] > 0.0)),
            }
        tail_pairwise[expert] = category_report
    return {
        "confidence_definition": "maximum softmax probability from each expert's OOF logits",
        "confidence_is_raw_and_not_assumed_calibrated_across_experts": True,
        "by_true_group": by_true_group,
        "tail_pairwise_comparisons": tail_pairwise,
    }


def _pattern_report(
    mask: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    true_groups: np.ndarray,
    *,
    shared_prediction_index: int | None = None,
) -> dict[str, Any]:
    count = int(mask.sum())
    group_counts = {
        name: int(np.sum(mask & (true_groups == name))) for name in GROUP_NAMES
    }
    report: dict[str, Any] = {
        "count": count,
        "true_group_composition": group_counts,
        "true_group_fraction": {
            name: None if count == 0 else float(group_counts[name] / count)
            for name in GROUP_NAMES
        },
        "correctness": {
            expert: _correctness(predictions[:, index], labels, mask)
            for index, expert in enumerate(EXPERT_ORDER)
        },
        "by_true_group": {},
    }
    if shared_prediction_index is not None:
        shared_correct = mask & (
            predictions[:, shared_prediction_index] == labels
        )
        report["shared_prediction_correct_count"] = int(shared_correct.sum())
        report["shared_prediction_correct_fraction"] = (
            None if count == 0 else float(shared_correct.sum() / count)
        )
    for name in GROUP_NAMES:
        group_mask = mask & (true_groups == name)
        report["by_true_group"][name] = {
            "count": int(group_mask.sum()),
            "correctness": {
                expert: _correctness(predictions[:, index], labels, group_mask)
                for index, expert in enumerate(EXPERT_ORDER)
            },
        }
    return report


def disagreement_diagnostics(
    expert_predictions: Any,
    labels: Any,
    class_counts: Any,
) -> dict[str, Any]:
    """Summarize prediction agreement, distinct-class counts, and correctness."""
    predictions = _validate_expert_matrix(
        expert_predictions, name="expert predictions", integer=True
    )
    labels_array = _as_integer_vector(labels, name="labels")
    if len(labels_array) != len(predictions):
        raise Task3FCTailDiagnosticError("labels and expert predictions are misaligned")
    counts = _validate_class_counts(class_counts)
    true_groups = assign_class_groups(labels_array, counts)
    all_agree = np.all(predictions == predictions[:, [0]], axis=1)
    lal_balanced_agree = predictions[:, 1] == predictions[:, 2]
    by_true_group: dict[str, Any] = {}
    for group_name in GROUP_NAMES:
        mask = true_groups == group_name
        distinct = np.apply_along_axis(lambda row: np.unique(row).size, 1, predictions[mask])
        distinct_counts = {
            str(number): int(np.sum(distinct == number))
            for number in range(1, len(EXPERT_ORDER) + 1)
        }
        by_true_group[group_name] = {
            "sample_count": int(mask.sum()),
            "all_four_agree_count": int(np.sum(mask & all_agree)),
            "all_four_agree_fraction": float(np.mean(all_agree[mask])),
            "mixup_disagrees_with_lal_count": int(
                np.sum(mask & (predictions[:, 3] != predictions[:, 1]))
            ),
            "mixup_disagrees_with_lal_fraction": float(
                np.mean(predictions[mask, 3] != predictions[mask, 1])
            ),
            "mixup_disagrees_with_balancedsoftmax_count": int(
                np.sum(mask & (predictions[:, 3] != predictions[:, 2]))
            ),
            "mixup_disagrees_with_balancedsoftmax_fraction": float(
                np.mean(predictions[mask, 3] != predictions[mask, 2])
            ),
            "lal_balancedsoftmax_agree_count": int(np.sum(mask & lal_balanced_agree)),
            "lal_balancedsoftmax_agree_fraction": float(
                np.mean(lal_balanced_agree[mask])
            ),
            "distinct_prediction_counts": distinct_counts,
        }
    patterns = {
        "all_four_agree": _pattern_report(
            all_agree, predictions, labels_array, true_groups
        ),
        "lal_balanced_agree_mixup_disagree": _pattern_report(
            lal_balanced_agree & (predictions[:, 3] != predictions[:, 1]),
            predictions,
            labels_array,
            true_groups,
            shared_prediction_index=1,
        ),
        "mixup_lal_agree_balanced_disagree": _pattern_report(
            (predictions[:, 3] == predictions[:, 1])
            & (predictions[:, 2] != predictions[:, 3]),
            predictions,
            labels_array,
            true_groups,
        ),
        "mixup_balanced_agree_lal_disagree": _pattern_report(
            (predictions[:, 3] == predictions[:, 2])
            & (predictions[:, 1] != predictions[:, 3]),
            predictions,
            labels_array,
            true_groups,
        ),
    }
    return {
        "by_true_group": by_true_group,
        "patterns": patterns,
        "expert_order": list(EXPERT_ORDER),
    }


def predicted_group_signal_report(
    expert_predictions: Any,
    labels: Any,
    class_counts: Any,
) -> dict[str, Any]:
    """Evaluate predicted-class-group signals after they are formed from predictions."""
    predictions = _validate_expert_matrix(
        expert_predictions, name="expert predictions", integer=True
    )
    labels_array = _as_integer_vector(labels, name="labels")
    if len(labels_array) != len(predictions):
        raise Task3FCTailDiagnosticError("labels and expert predictions are misaligned")
    counts = _validate_class_counts(class_counts)
    actual_groups = assign_class_groups(labels_array, counts)
    predicted_groups = np.column_stack(
        [assign_class_groups(predictions[:, index], counts) for index in range(4)]
    )
    per_expert: dict[str, Any] = {}
    for index, expert in enumerate(EXPERT_ORDER):
        predicted = predicted_groups[:, index]
        predicted_tail = predicted == "tail"
        actual_tail = actual_groups == "tail"
        confusion_counts = {
            actual_name: {
                predicted_name: int(
                    np.sum(
                        (actual_groups == actual_name)
                        & (predicted == predicted_name)
                    )
                )
                for predicted_name in GROUP_NAMES
            }
            for actual_name in GROUP_NAMES
        }
        predicted_counts = {
            name: int(np.sum(predicted == name)) for name in GROUP_NAMES
        }
        predicted_tail_count = int(predicted_tail.sum())
        actual_tail_count = int(actual_tail.sum())
        joint_tail_count = int(np.sum(predicted_tail & actual_tail))
        per_expert[expert] = {
            "predicted_group_counts": predicted_counts,
            "predicted_group_fractions": {
                name: float(predicted_counts[name] / len(predicted))
                for name in GROUP_NAMES
            },
            "predicted_tail_count": predicted_tail_count,
            "predicted_tail_fraction": float(predicted_tail.mean()),
            "actual_tail_count": actual_tail_count,
            "actual_tail_recall": float(joint_tail_count / actual_tail_count),
            "actual_tail_miss_count": int(np.sum(actual_tail & ~predicted_tail)),
            "actual_tail_miss_rate": float(
                np.sum(actual_tail & ~predicted_tail) / actual_tail_count
            ),
            "predicted_tail_precision": None
            if predicted_tail_count == 0
            else float(joint_tail_count / predicted_tail_count),
            "confusion_counts_actual_by_predicted": confusion_counts,
        }
    return {
        "signal_definition": (
            "map each expert's top-1 predicted class to the canonical class group; "
            "true groups are used only for retrospective evaluation"
        ),
        "per_expert": per_expert,
    }


def _weight_pattern_report(
    mask: np.ndarray,
    predictions: np.ndarray,
    labels: np.ndarray,
    true_groups: np.ndarray,
    weights: np.ndarray,
    *,
    shared_prediction_index: int | None = None,
) -> dict[str, Any]:
    report = _pattern_report(
        mask,
        predictions,
        labels,
        true_groups,
        shared_prediction_index=shared_prediction_index,
    )
    report["mean_weight"] = {
        expert: None
        if not mask.any()
        else float(weights[mask, index].mean())
        for index, expert in enumerate(EXPERT_ORDER)
    }
    report["std_weight"] = {
        expert: None
        if not mask.any()
        else float(weights[mask, index].std())
        for index, expert in enumerate(EXPERT_ORDER)
    }
    return report


def ridge_weight_pattern_diagnostics(
    expert_predictions: Any,
    labels: Any,
    weights: Any,
    class_counts: Any,
) -> dict[str, Any]:
    """Condition saved Ridge weights on prediction-only patterns and outcomes."""
    predictions = _validate_expert_matrix(
        expert_predictions, name="expert predictions", integer=True
    )
    labels_array = _as_integer_vector(labels, name="labels")
    if len(labels_array) != len(predictions):
        raise Task3FCTailDiagnosticError("labels and expert predictions are misaligned")
    counts = _validate_class_counts(class_counts)
    weight_array = _validate_weights(weights, num_samples=len(predictions))
    true_groups = assign_class_groups(labels_array, counts)
    predicted_groups = np.column_stack(
        [assign_class_groups(predictions[:, index], counts) for index in range(4)]
    )
    lal_balanced_agree = predictions[:, 1] == predictions[:, 2]
    mixup_disagrees = predictions[:, 3] != predictions[:, 1]
    patterns = {
        "lal_balanced_agree_on_tail_class": _weight_pattern_report(
            lal_balanced_agree & (predicted_groups[:, 1] == "tail"),
            predictions,
            labels_array,
            true_groups,
            weight_array,
            shared_prediction_index=1,
        ),
        "lal_balanced_agree_on_tail_class_mixup_disagrees": _weight_pattern_report(
            lal_balanced_agree
            & (predicted_groups[:, 1] == "tail")
            & (predictions[:, 3] != predictions[:, 1]),
            predictions,
            labels_array,
            true_groups,
            weight_array,
            shared_prediction_index=1,
        ),
        "lal_balanced_agree_mixup_disagrees": _weight_pattern_report(
            lal_balanced_agree & mixup_disagrees,
            predictions,
            labels_array,
            true_groups,
            weight_array,
            shared_prediction_index=1,
        ),
        "mixup_disagrees_with_both_rebalanced": _weight_pattern_report(
            (predictions[:, 3] != predictions[:, 1])
            & (predictions[:, 3] != predictions[:, 2]),
            predictions,
            labels_array,
            true_groups,
            weight_array,
        ),
    }
    by_expert_predicted_group: dict[str, Any] = {}
    for source_index, source_expert in enumerate(EXPERT_ORDER):
        by_expert_predicted_group[source_expert] = {}
        for group_name in GROUP_NAMES:
            by_expert_predicted_group[source_expert][group_name] = _weight_pattern_report(
                predicted_groups[:, source_index] == group_name,
                predictions,
                labels_array,
                true_groups,
                weight_array,
            )

    tail_mask = true_groups == "tail"
    tail_correctness_patterns: dict[str, Any] = {}
    for expert in REBALANCED_EXPERTS:
        index = EXPERT_ORDER.index(expert)
        expert_correct = predictions[:, index] == labels_array
        mixup_correct = predictions[:, 3] == labels_array
        tail_correctness_patterns[f"{expert}_correct_Mixup_wrong"] = _weight_pattern_report(
            tail_mask & expert_correct & ~mixup_correct,
            predictions,
            labels_array,
            true_groups,
            weight_array,
        )
        tail_correctness_patterns[f"Mixup_correct_{expert}_wrong"] = _weight_pattern_report(
            tail_mask & mixup_correct & ~expert_correct,
            predictions,
            labels_array,
            true_groups,
            weight_array,
        )
    return {
        "prediction_group_definition": (
            "each expert's top-1 class mapped to canonical Head/Medium/Tail; "
            "labels enter only in correctness and true-group composition"
        ),
        "overall_sample_count": int(len(predictions)),
        "overall_mean_weight": {
            expert: float(weight_array[:, index].mean())
            for index, expert in enumerate(EXPERT_ORDER)
        },
        "overall_std_weight": {
            expert: float(weight_array[:, index].std())
            for index, expert in enumerate(EXPERT_ORDER)
        },
        "patterns": patterns,
        "by_expert_predicted_group": by_expert_predicted_group,
        "tail_correctness_patterns": tail_correctness_patterns,
    }


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
        raise Task3FCTailDiagnosticError(f"cannot load {name}: {path}") from exc
    if not isinstance(payload, dict):
        raise Task3FCTailDiagnosticError(f"{name} must be a JSON object")
    return payload


def _load_npz(path: Path, *, name: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: np.array(archive[key]) for key in archive.files}
    except (OSError, ValueError) as exc:
        raise Task3FCTailDiagnosticError(f"cannot load {name}: {path}") from exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Task3FCTailDiagnosticError(message)


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


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise Task3FCTailDiagnosticError(
                f"refusing to overwrite incompatible output: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _write_text_once(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text() != text:
            raise Task3FCTailDiagnosticError(
                f"refusing to overwrite incompatible output: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


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


def _find_unique_index(values: Any, target: str, *, name: str) -> int:
    array = np.asarray(values).astype(str)
    matches = np.flatnonzero(array == target)
    if len(matches) != 1:
        raise Task3FCTailDiagnosticError(
            f"{name} must contain exactly one {target!r}, found {len(matches)}"
        )
    return int(matches[0])


def _metric_group_text(report: Mapping[str, Any], expert: str, field: str) -> str:
    value = report["by_true_group"]["tail"]["per_expert"][expert][field]
    return "n/a" if value is None else f"{value:.4f}"


def _render_summary(results: Mapping[str, Any]) -> str:
    confidence = results["investigation_a"]
    disagreement = results["investigation_b"]
    predicted_groups = results["investigation_c"]["predicted_class_group_signal"]
    ridge = results["investigation_d"]
    tail_confidence = confidence["by_true_group"]["tail"]["per_expert"]
    tail_count = confidence["by_true_group"]["tail"]["sample_count"]
    shared = disagreement["patterns"]["lal_balanced_agree_mixup_disagree"]
    shared_tail = shared["by_true_group"]["tail"]
    mixup_group = predicted_groups["per_expert"]["Mixup"]
    lal_group = predicted_groups["per_expert"]["LAL"]
    balanced_group = predicted_groups["per_expert"]["BalancedSoftmax"]
    ridge_shared = ridge["patterns"]["lal_balanced_agree_on_tail_class"]
    ridge_shared_disagree = ridge["patterns"][
        "lal_balanced_agree_on_tail_class_mixup_disagrees"
    ]
    ridge_both_disagree = ridge["patterns"]["mixup_disagrees_with_both_rebalanced"]
    lal_tail_pair = confidence["tail_pairwise_comparisons"]["LAL"][
        "rebalanced_correct_mixup_wrong"
    ]
    balanced_tail_pair = confidence["tail_pairwise_comparisons"]["BalancedSoftmax"][
        "rebalanced_correct_mixup_wrong"
    ]
    lal_pred_head = ridge["by_expert_predicted_group"]["LAL"]["head"]
    lal_pred_tail = ridge["by_expert_predicted_group"]["LAL"]["tail"]
    balanced_pred_head = ridge["by_expert_predicted_group"]["BalancedSoftmax"]["head"]
    balanced_pred_tail = ridge["by_expert_predicted_group"]["BalancedSoftmax"]["tail"]
    lal_tail_correctness = ridge["tail_correctness_patterns"][
        "LAL_correct_Mixup_wrong"
    ]
    balanced_tail_correctness = ridge["tail_correctness_patterns"][
        "BalancedSoftmax_correct_Mixup_wrong"
    ]

    lines = [
        "# Task 3F-C — Tail-specific routing-signal diagnostics",
        "",
        f"This is a retrospective diagnostic on the frozen {results['provenance']['analyzed_sample_count']:,}-image "
        "outer-fold-0 / inner-folds-1–3 population. Signals are formed from "
        "expert OOF predictions and confidences only; labels and true class groups "
        "are used afterward for evaluation. No router was refit.",
        "",
        "## Findings",
        "",
        f"1. **Confidence.** On the {tail_count} true Tail rows, mean/median raw "
        f"confidence is CE {_metric_group_text(confidence, 'CE', 'mean')}/"
        f"{_metric_group_text(confidence, 'CE', 'median')}, LAL "
        f"{_metric_group_text(confidence, 'LAL', 'mean')}/"
        f"{_metric_group_text(confidence, 'LAL', 'median')}, BalancedSoftmax "
        f"{_metric_group_text(confidence, 'BalancedSoftmax', 'mean')}/"
        f"{_metric_group_text(confidence, 'BalancedSoftmax', 'median')}, and Mixup "
        f"{_metric_group_text(confidence, 'Mixup', 'mean')}/"
        f"{_metric_group_text(confidence, 'Mixup', 'median')}. Mixup's incorrect-Tail "
        f"mean/median confidence is "
        f"{tail_confidence['Mixup']['incorrect_confidence']['mean']:.4f}/"
        f"{tail_confidence['Mixup']['incorrect_confidence']['median']:.4f}; "
        f"on rows where LAL or BalancedSoftmax is correct and Mixup is wrong, the "
        f"rebalanced-minus-Mixup confidence margins average "
        f"+{lal_tail_pair['rebalanced_minus_mixup_confidence']['mean']:.4f} "
        f"(LAL, n={lal_tail_pair['count']}) and "
        f"+{balanced_tail_pair['rebalanced_minus_mixup_confidence']['mean']:.4f} "
        f"(BalancedSoftmax, n={balanced_tail_pair['count']}). This is a retrospective "
        "association suggesting limited signal, not a correctness guarantee.",
        f"2. **Disagreement.** LAL and BalancedSoftmax agree while Mixup disagrees "
        f"on {shared['count']} rows overall, including {shared_tail['count']} Tail rows; "
        f"their shared prediction is correct on {shared['shared_prediction_correct_count']} "
        f"of the overall cases ({shared['shared_prediction_correct_fraction']:.4f}). "
        f"Within this pattern, Tail contributes {shared_tail['count']} rows and its "
        f"shared prediction is correct on "
        f"{shared_tail['correctness']['LAL']['correct_fraction']:.4f}; Tail all-four "
        f"agreement is {disagreement['by_true_group']['tail']['all_four_agree_fraction']:.4f}; "
        "disagreement is more common on Tail than Head, but the shared-prediction "
        "pattern is only weakly informative at this sample size; the full "
        "Head/Medium/Tail pattern table is in the JSON artifact.",
        f"3. **Predicted class groups.** Tail-class prediction rates are LAL "
        f"{lal_group['predicted_tail_fraction']:.4f}, BalancedSoftmax "
        f"{balanced_group['predicted_tail_fraction']:.4f}, and Mixup "
        f"{mixup_group['predicted_tail_fraction']:.4f}. Their retrospective Tail "
        f"recalls are {lal_group['actual_tail_recall']:.4f}, "
        f"{balanced_group['actual_tail_recall']:.4f}, and "
        f"{mixup_group['actual_tail_recall']:.4f}; predicted-Tail precision is "
        f"{lal_group['predicted_tail_precision']:.4f}, "
        f"{balanced_group['predicted_tail_precision']:.4f}, and "
        f"{mixup_group['predicted_tail_precision']:.4f}, with Tail miss rates "
        f"{lal_group['actual_tail_miss_rate']:.4f}, "
        f"{balanced_group['actual_tail_miss_rate']:.4f}, and "
        f"{mixup_group['actual_tail_miss_rate']:.4f}. This evaluates a prediction-only "
        "signal with useful recall differences but low precision for the rebalanced "
        "experts, and does not make the true group an input feature.",
        f"4. **Ridge weights.** When LAL and BalancedSoftmax agree on a predicted Tail "
        f"class ({ridge_shared['count']} rows), their mean saved weights are "
        f"{ridge_shared['mean_weight']['LAL']:.4f} and "
        f"{ridge_shared['mean_weight']['BalancedSoftmax']:.4f}, while Mixup receives "
        f"{ridge_shared['mean_weight']['Mixup']:.4f}. In the subset where Mixup also "
        f"disagrees ({ridge_shared_disagree['count']} rows), its mean weight is "
        f"{ridge_shared_disagree['mean_weight']['Mixup']:.4f}. When Mixup disagrees "
        f"with both rebalanced experts ({ridge_both_disagree['count']} rows), its mean "
        f"weight is {ridge_both_disagree['mean_weight']['Mixup']:.4f} versus the overall "
        f"{ridge['overall_mean_weight']['Mixup']:.4f}. For prediction-only group "
        f"conditioning, Mixup weight is {lal_pred_head['mean_weight']['Mixup']:.4f} when "
        f"LAL predicts Head and {lal_pred_tail['mean_weight']['Mixup']:.4f} when LAL "
        f"predicts Tail; the corresponding BalancedSoftmax values are "
        f"{balanced_pred_head['mean_weight']['Mixup']:.4f} and "
        f"{balanced_pred_tail['mean_weight']['Mixup']:.4f}. On true Tail rows where "
        f"LAL / BalancedSoftmax are correct and Mixup is wrong (n={lal_tail_correctness['count']} / "
        f"{balanced_tail_correctness['count']}), Ridge still assigns Mixup mean weights "
        f"{lal_tail_correctness['mean_weight']['Mixup']:.4f} / "
        f"{balanced_tail_correctness['mean_weight']['Mixup']:.4f}. These are conditioned "
        "associations, not a new routing rule.",
        "",
        "## Confidence correctness splits on Tail rows",
        "",
        "| Expert | Correct count | Incorrect count | Mean confidence when correct | Mean confidence when incorrect |",
        "|:--|--:|--:|--:|--:|",
    ]
    for expert in EXPERT_ORDER:
        row = tail_confidence[expert]
        lines.append(
            f"| {expert} | {row['correct_count']} | {row['incorrect_count']} | "
            f"{row['correct_confidence']['mean']:.4f} | "
            f"{row['incorrect_confidence']['mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            f"Only {tail_count} Tail rows are available, so subgroup percentages can be "
            "unstable, especially for multi-condition disagreement patterns. Raw "
            "softmax confidence scales may differ across the four training objectives "
            "and are not assumed calibrated or directly comparable. All class-group "
            "and correctness results are retrospective. These associations do not "
            "establish a routing improvement, justify selecting a new feature, or "
            "support independent validation.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_task3f_a_inputs(
    *,
    dataset: RestrictedAnalysisDataset,
    config: Mapping[str, Any],
    ridge_results: Mapping[str, Any],
    predictions: Mapping[str, np.ndarray],
    scores: Mapping[str, np.ndarray],
    fold_assignments: Mapping[str, Any],
) -> None:
    _require(config.get("task_identifier") == TASK3FA_TASK_IDENTIFIER, "Task 3F-A config has the wrong task identifier")
    _require(config.get("expert_order") == list(EXPERT_ORDER), "Task 3F-A expert order is incompatible")
    _require(
        config.get("analyzed_sample_count") == dataset.num_samples == ANALYSIS_SAMPLE_COUNT,
        "Task 3F-A population is not the permitted 6,507 rows",
    )
    _require(
        config.get("permitted_analysis_inner_folds") == list(PERMITTED_ANALYSIS_INNER_FOLDS),
        "Task 3F-A permitted folds are incompatible",
    )
    _require(config.get("reserved_router_selection_inner_folds") == [0], "reserved fold declaration is incompatible")
    restrictions = config.get("data_restrictions", {})
    for key in (
        "inner_fold_zero_used",
        "reserved_outer_evaluation_used",
        "original_cifar_test_used",
        "expert_training_or_checkpoints_modified",
        "source_oof_logits_modified",
    ):
        _require(restrictions.get(key) is False, f"Task 3F-A restriction {key} failed")
    _require(
        fold_assignments.get("outer_fold") == OUTER_FOLD
        and fold_assignments.get("permitted_inner_fold_ids") == list(PERMITTED_ANALYSIS_INNER_FOLDS)
        and fold_assignments.get("every_sample_held_out_exactly_once") is True,
        "Task 3F-A fold assignment provenance is incompatible",
    )
    validate_signal_alignment(
        expected_sample_indices=dataset.sample_indices,
        expected_inner_fold_ids=dataset.inner_fold_ids,
        expected_labels=dataset.labels,
        artifact_sample_indices=predictions.get("sample_indices"),
        artifact_inner_fold_ids=predictions.get("inner_fold_ids"),
        artifact_labels=predictions.get("labels"),
        expert_predictions=dataset.logits.argmax(axis=2),
        confidences=extract_features(dataset.logits, FEATURE_SET_CONFIDENCE),
    )
    validate_signal_alignment(
        expected_sample_indices=dataset.sample_indices,
        expected_inner_fold_ids=dataset.inner_fold_ids,
        expected_labels=dataset.labels,
        artifact_sample_indices=scores.get("sample_indices"),
        artifact_inner_fold_ids=scores.get("inner_fold_ids"),
        artifact_labels=scores.get("labels"),
        expert_predictions=dataset.logits.argmax(axis=2),
        confidences=extract_features(dataset.logits, FEATURE_SET_CONFIDENCE),
    )
    _require("model_fits" in ridge_results and "adaptive_results" in ridge_results, "Task 3F-A results are incomplete")


def run_task3f_tail_signal_diagnostics(
    *,
    data_root: str | Path = "./data",
    oof_directory: str | Path = "artifacts/oof/task3c_oof",
    ridge_directory: str | Path = "artifacts/oof/task3f_ridge",
    mixup_diagnostics_directory: str | Path = "artifacts/oof/task3f_mixup_diagnostics",
    output_directory: str | Path = "artifacts/oof/task3f_tail_signal_diagnostics",
    project_root: str | Path | None = None,
) -> dict[str, Path]:
    """Run Task 3F-C using existing artifacts and write separate outputs."""
    project_root_path = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    source_directory = Path(oof_directory).resolve()
    ridge_path = Path(ridge_directory).resolve()
    mixup_path = Path(mixup_diagnostics_directory).resolve()
    output_path = Path(output_directory).resolve()
    task3f_files = {
        "experiment_config": ridge_path / "experiment_config.json",
        "ridge_results": ridge_path / "ridge_results.json",
        "held_out_predictions": ridge_path / "held_out_predictions.npz",
        "router_scores": ridge_path / "router_scores.npz",
        "fold_assignments": ridge_path / "fold_assignments.json",
    }
    for name, path in task3f_files.items():
        if not path.exists():
            raise Task3FCTailDiagnosticError(f"missing Task 3F-A artifact {name}: {path}")
    task3fb_files = {
        "diagnostic_results": mixup_path / "diagnostic_results.json",
        "summary": mixup_path / "summary.md",
    }
    for name, path in task3fb_files.items():
        if not path.exists():
            raise Task3FCTailDiagnosticError(f"missing Task 3F-B artifact {name}: {path}")
    task3f_hashes_before = {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in task3f_files.items()
    }
    task3fb_hashes = {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in task3fb_files.items()
    }
    config = _load_json(task3f_files["experiment_config"], name="Task 3F-A experiment_config")
    ridge_results = _load_json(task3f_files["ridge_results"], name="Task 3F-A ridge_results")
    fold_assignments = _load_json(task3f_files["fold_assignments"], name="Task 3F-A fold_assignments")
    predictions = _load_npz(task3f_files["held_out_predictions"], name="Task 3F-A held_out_predictions")
    scores = _load_npz(task3f_files["router_scores"], name="Task 3F-A router_scores")
    mixup_results = _load_json(task3fb_files["diagnostic_results"], name="Task 3F-B diagnostic_results")
    _require(mixup_results.get("task_identifier") == TASK3FB_TASK_IDENTIFIER, "Task 3F-B result identifier is incompatible")
    _require(mixup_results.get("highlighted_configuration_id") == HIGHLIGHTED_CONFIGURATION_ID, "Task 3F-B highlighted configuration is incompatible")
    mixup_provenance = mixup_results.get("provenance", {})
    _require(
        mixup_provenance.get("analyzed_sample_count") == ANALYSIS_SAMPLE_COUNT,
        "Task 3F-B population is not the permitted 6,507 rows",
    )
    _require(
        mixup_provenance.get("outer_fold") == OUTER_FOLD
        and mixup_provenance.get("permitted_inner_folds")
        == list(PERMITTED_ANALYSIS_INNER_FOLDS)
        and mixup_provenance.get("reserved_inner_folds") == [0],
        "Task 3F-B fold provenance is incompatible",
    )
    for key in (
        "oracle_weights_used",
        "original_cifar_test_used",
        "reserved_outer_evaluation_used",
        "training_or_checkpoints_modified",
    ):
        _require(mixup_provenance.get(key) is False, f"Task 3F-B restriction {key} failed")
    recorded_task3f_a_hashes = mixup_provenance.get("task3f_a_artifacts", {})
    for name, details in task3f_hashes_before.items():
        _require(
            recorded_task3f_a_hashes.get(name, {}).get("sha256")
            == details["sha256"],
            f"Task 3F-B provenance does not match current Task 3F-A artifact {name}",
        )

    manager = NestedOOFFoldManager.from_canonical_training_data(
        data_root,
        seed=42,
        outer_folds=5,
        inner_folds=4,
        expert_order=EXPERT_ORDER,
    )
    dataset, source_files = load_restricted_analysis_dataset(source_directory, manager)
    _validate_task3f_a_inputs(
        dataset=dataset,
        config=config,
        ridge_results=ridge_results,
        predictions=predictions,
        scores=scores,
        fold_assignments=fold_assignments,
    )
    class_counts = np.asarray(manager.canonical_class_counts, dtype=np.int64)
    _require(len(class_counts) == NUM_CLASSES, "canonical class counts are incompatible")
    expert_predictions = dataset.logits.argmax(axis=2).astype(np.int64)
    confidences = extract_features(dataset.logits, FEATURE_SET_CONFIDENCE)
    adaptive_ids = np.asarray(predictions["adaptive_configuration_ids"]).astype(str)
    highlighted_index = _find_unique_index(
        adaptive_ids,
        HIGHLIGHTED_CONFIGURATION_ID,
        name="adaptive configuration IDs",
    )
    highlighted_weights = np.asarray(
        predictions["adaptive_weights"][highlighted_index], dtype=np.float64
    )
    highlighted_predictions = np.asarray(
        predictions["adaptive_predictions"][highlighted_index], dtype=np.int64
    )
    validate_signal_alignment(
        expected_sample_indices=dataset.sample_indices,
        expected_inner_fold_ids=dataset.inner_fold_ids,
        expected_labels=dataset.labels,
        artifact_sample_indices=dataset.sample_indices,
        artifact_inner_fold_ids=dataset.inner_fold_ids,
        artifact_labels=dataset.labels,
        expert_predictions=expert_predictions,
        confidences=confidences,
        weights=highlighted_weights,
    )
    normalized_weights = highlighted_weights / highlighted_weights.sum(axis=1, keepdims=True)
    recomputed_predictions = combine_weighted_logits(
        dataset.logits, normalized_weights
    ).argmax(axis=1)
    _require(
        np.array_equal(recomputed_predictions, highlighted_predictions),
        "saved highlighted predictions do not match the saved Ridge weights",
    )
    _require(
        np.array_equal(
            dataset.sample_indices,
            np.asarray(predictions["sample_indices"], dtype=np.int64),
        ),
        "saved Task 3F-A sample IDs changed during diagnostic setup",
    )

    investigation_a = confidence_diagnostics(
        confidences, expert_predictions, dataset.labels, class_counts
    )
    investigation_b = disagreement_diagnostics(
        expert_predictions, dataset.labels, class_counts
    )
    investigation_c = {
        "predicted_class_group_signal": predicted_group_signal_report(
            expert_predictions, dataset.labels, class_counts
        ),
        "rebalanced_expert_agreement": {
            key: value
            for key, value in investigation_b["patterns"].items()
            if key != "all_four_agree"
        },
    }
    investigation_d = ridge_weight_pattern_diagnostics(
        expert_predictions, dataset.labels, highlighted_weights, class_counts
    )

    task3f_hashes_after = {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in task3f_files.items()
    }
    _require(task3f_hashes_before == task3f_hashes_after, "a Task 3F-A artifact changed during diagnostics")
    task3fb_hashes_after = {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in task3fb_files.items()
    }
    _require(task3fb_hashes == task3fb_hashes_after, "a Task 3F-B artifact changed during diagnostics")
    source_hashes = {
        name: {"path": str(details["path"]), "sha256": str(details["sha256"])}
        for name, details in source_files.items()
    }
    results: dict[str, Any] = {
        "task_identifier": TASK3FC_TASK_IDENTIFIER,
        "schema_version": TASK3FC_SCHEMA_VERSION,
        "source_git_commit": _git_commit(project_root_path),
        "highlighted_configuration_id": HIGHLIGHTED_CONFIGURATION_ID,
        "provenance": {
            "dataset": DATASET_NAME,
            "outer_fold": OUTER_FOLD,
            "permitted_inner_folds": list(PERMITTED_ANALYSIS_INNER_FOLDS),
            "reserved_inner_folds": [0],
            "analyzed_sample_count": dataset.num_samples,
            "expert_order": list(EXPERT_ORDER),
            "true_class_group_used_only_retrospectively": True,
            "labels_used_only_for_retrospective_evaluation": True,
            "router_refit": False,
            "experts_retrained": False,
            "oracle_weights_used": False,
            "original_cifar_test_used": False,
            "reserved_outer_evaluation_used": False,
            "task3f_a_artifacts": task3f_hashes_before,
            "task3f_b_artifacts": task3fb_hashes,
            "task3c_oof_artifacts": source_hashes,
            "task3f_a_config_input_artifacts": config.get("input_artifacts", {}),
        },
        "signal_contract": {
            "inference_time_inputs": [
                "four experts' OOF logits",
                "four experts' maximum softmax probabilities",
                "four experts' top-1 predicted class IDs",
            ],
            "not_inference_time_inputs": [
                "true labels",
                "true Head/Medium/Tail group",
                "oracle contribution targets",
            ],
            "canonical_group_source": "scripts.base_trainer.compute_class_groups",
        },
        "canonical_class_counts": [int(value) for value in class_counts.tolist()],
        "investigation_a": investigation_a,
        "investigation_b": investigation_b,
        "investigation_c": investigation_c,
        "investigation_d": investigation_d,
    }
    summary = _render_summary(results)
    result_path = output_path / "diagnostic_results.json"
    summary_path = output_path / "summary.md"
    _write_json_once(result_path, results)
    _write_text_once(summary_path, summary)
    return {"diagnostic_results": result_path, "summary": summary_path}


__all__ = [
    "HIGHLIGHTED_CONFIGURATION_ID",
    "Task3FCTailDiagnosticError",
    "assign_class_groups",
    "confidence_diagnostics",
    "disagreement_diagnostics",
    "predicted_group_signal_report",
    "ridge_weight_pattern_diagnostics",
    "run_task3f_tail_signal_diagnostics",
    "validate_signal_alignment",
]
