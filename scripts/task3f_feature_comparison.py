"""Task 3F-F: read-only comparison of the two existing Ridge feature sets.

This module consumes the frozen Task 3C OOF logits and the saved Task 3F-A
held-out scores, predictions, weights, and model diagnostics.  It never fits
Ridge, retrains an expert, changes a saved weight, or accesses the reserved
outer population or the original CIFAR-100 test set.

Labels are used only after the saved inference-time artifacts have been
loaded, for retrospective contribution-target, class-group, and
classification diagnostics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from data.nested_oof import NestedOOFFoldManager, OOFProtocolError
from scripts.analysis import ArtifactReader, ImmutableArtifactWriter
from scripts.base_trainer import compute_class_groups
from scripts.task3e_fixed import (
    EXPERT_ORDER,
    RestrictedAnalysisDataset,
    load_restricted_analysis_dataset,
)
from scripts.task3f_ridge import (
    ALPHAS,
    FEATURE_NAMES,
    FEATURE_SET_CONFIDENCE,
    FEATURE_SET_FULL,
    FEATURE_SETS,
    GAMMAS,
    PERMITTED_ANALYSIS_INNER_FOLDS,
    SHRINKAGES,
    TEMPERATURES,
    _adaptive_configuration_id,
    _adaptive_model_key,
    classification_metrics,
    combine_weighted_logits,
    compute_contribution_targets,
    extract_features,
)


TASK3FF_SCHEMA_VERSION = "task3f_feature_comparison.v1"
TASK3FF_TASK_IDENTIFIER = "Task 3F-F"
TASK3FA_TASK_IDENTIFIER = "Task 3F-A"
TASK3FE_TASK_IDENTIFIER = "Task 3F-E"
DATASET_NAME = "CIFAR-100-LT"
IMBALANCE_RATIO = 100.0
CANONICAL_POPULATION_SIZE = 10_847
ANALYSIS_SAMPLE_COUNT = 6_507
NUM_CLASSES = 100
TRAINING_SEED = 78
FOLD_GENERATION_SEED = 42
OUTER_FOLD = 0
RESERVED_ROUTER_SELECTION_INNER_FOLDS = (0,)
OUTER_EVALUATION_SIZE = 2_170
GROUP_NAMES = ("head", "medium", "tail")
REBALANCED_EXPERTS = ("LAL", "BalancedSoftmax")
MIXUP = "Mixup"
PRIMARY_ALPHA = 1000.0
PRIMARY_GAMMA = 1.0
PRIMARY_TEMPERATURE = 2.0
PRIMARY_SHRINKAGE = 0.75
WEIGHT_SUM_ATOL = 2e-6
SCORE_RECONSTRUCTION_ATOL = 2e-5
MODEL_RECONSTRUCTION_ATOL = 5e-5


class Task3FFeatureComparisonError(OOFProtocolError):
    """Raised when Task 3F-F input or provenance validation fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Task3FFeatureComparisonError(message)


def _as_numeric(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise Task3FFeatureComparisonError(f"{name} must contain real numeric values")
    if not np.isfinite(array).all():
        raise Task3FFeatureComparisonError(f"{name} contains non-finite values")
    return array.astype(np.float64, copy=False)


def _as_integer_vector(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise Task3FFeatureComparisonError(f"{name} must be one-dimensional")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise Task3FFeatureComparisonError(f"{name} must contain integer values")
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        raise Task3FFeatureComparisonError(
            f"{name} must contain finite integer-valued entries"
        )
    return array.astype(np.int64, copy=False)


def _load_json(path: Path, *, name: str) -> dict[str, Any]:
    return ArtifactReader(error_type=Task3FFeatureComparisonError).read_json(
        path, name=name
    )


def _load_npz(path: Path, *, name: str) -> dict[str, np.ndarray]:
    return ArtifactReader(error_type=Task3FFeatureComparisonError).read_npz(
        path, name=name
    )


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hash of one source artifact."""
    return ArtifactReader(error_type=Task3FFeatureComparisonError).sha256_file(
        path, description="source artifact"
    )


def _sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    ImmutableArtifactWriter(
        error_type=Task3FFeatureComparisonError
    ).write_json_once(path, payload)


def _write_text_once(path: Path, rendered: str) -> None:
    ImmutableArtifactWriter(
        error_type=Task3FFeatureComparisonError
    ).write_text_once(path, rendered)


def _validate_alignment_arrays(
    *,
    expected_sample_indices: Any,
    expected_inner_fold_ids: Any,
    expected_labels: Any,
    artifact_sample_indices: Any,
    artifact_inner_fold_ids: Any,
    artifact_labels: Any,
) -> None:
    expected_ids = _as_integer_vector(expected_sample_indices, name="expected sample IDs")
    expected_folds = _as_integer_vector(expected_inner_fold_ids, name="expected inner-fold IDs")
    expected_y = _as_integer_vector(expected_labels, name="expected labels")
    artifact_ids = _as_integer_vector(artifact_sample_indices, name="artifact sample IDs")
    artifact_folds = _as_integer_vector(artifact_inner_fold_ids, name="artifact inner-fold IDs")
    artifact_y = _as_integer_vector(artifact_labels, name="artifact labels")
    expected_length = len(expected_ids)
    if any(
        len(array) != expected_length
        for array in (expected_folds, expected_y, artifact_ids, artifact_folds, artifact_y)
    ):
        raise Task3FFeatureComparisonError("alignment arrays have inconsistent lengths")
    if len(np.unique(expected_ids)) != expected_length:
        raise Task3FFeatureComparisonError("expected sample IDs contain duplicates")
    if len(np.unique(artifact_ids)) != expected_length:
        raise Task3FFeatureComparisonError("artifact sample IDs contain duplicates")
    if np.any(artifact_folds == 0):
        raise Task3FFeatureComparisonError(
            "artifact contains the reserved inner-fold-0 population"
        )
    if not np.all(np.isin(artifact_folds, PERMITTED_ANALYSIS_INNER_FOLDS)):
        raise Task3FFeatureComparisonError("artifact contains an incompatible inner fold")
    if not np.array_equal(expected_ids, artifact_ids):
        raise Task3FFeatureComparisonError("sample IDs are not exactly aligned")
    if not np.array_equal(expected_folds, artifact_folds):
        raise Task3FFeatureComparisonError("inner-fold IDs are not exactly aligned")
    if not np.array_equal(expected_y, artifact_y):
        raise Task3FFeatureComparisonError("labels are not exactly aligned")


def validate_artifact_alignment(
    *,
    expected_sample_indices: Any,
    expected_inner_fold_ids: Any,
    expected_labels: Any,
    artifact_sample_indices: Any,
    artifact_inner_fold_ids: Any,
    artifact_labels: Any,
) -> None:
    """Validate exact row order and reject all non-permitted folds."""
    _validate_alignment_arrays(
        expected_sample_indices=expected_sample_indices,
        expected_inner_fold_ids=expected_inner_fold_ids,
        expected_labels=expected_labels,
        artifact_sample_indices=artifact_sample_indices,
        artifact_inner_fold_ids=artifact_inner_fold_ids,
        artifact_labels=artifact_labels,
    )


def _canonical_group_labels(labels: np.ndarray, class_counts: np.ndarray) -> np.ndarray:
    labels_array = _as_integer_vector(labels, name="labels")
    counts = _as_integer_vector(class_counts, name="class counts")
    if len(counts) != NUM_CLASSES or np.any(counts < 1):
        raise Task3FFeatureComparisonError("class counts must contain 100 positive entries")
    if np.any(labels_array < 0) or np.any(labels_array >= len(counts)):
        raise Task3FFeatureComparisonError("labels fall outside the canonical class range")
    groups = compute_class_groups(counts)
    result = np.full(len(labels_array), "", dtype="U6")
    for group_name in GROUP_NAMES:
        result[np.isin(labels_array, groups[group_name])] = group_name
    if np.any(result == ""):
        raise Task3FFeatureComparisonError("some labels have no canonical class group")
    return result


def pearson_correlation(left: Any, right: Any) -> float | None:
    """Return Pearson correlation, or ``None`` for a degenerate input."""
    left_array = _as_numeric(left, name="left values").reshape(-1)
    right_array = _as_numeric(right, name="right values").reshape(-1)
    if left_array.shape != right_array.shape or len(left_array) < 2:
        return None
    left_centered = left_array - left_array.mean()
    right_centered = right_array - right_array.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator == 0.0 or not np.isfinite(denominator):
        return None
    result = float(np.dot(left_centered, right_centered) / denominator)
    return result if np.isfinite(result) else None


def _validate_score_matrix(
    scores: Any,
    *,
    num_samples: int,
    name: str = "scores",
) -> np.ndarray:
    array = _as_numeric(scores, name=name)
    if array.shape != (num_samples, len(EXPERT_ORDER)):
        raise Task3FFeatureComparisonError(
            f"{name} must have shape ({num_samples}, {len(EXPERT_ORDER)})"
        )
    return array


def _error_metrics(predicted: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    predicted_array = _as_numeric(predicted, name="predicted scores")
    actual_array = _as_numeric(actual, name="actual targets")
    if predicted_array.shape != actual_array.shape:
        raise Task3FFeatureComparisonError("predicted scores and targets are misaligned")
    error = predicted_array - actual_array
    return {
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
        "pearson_correlation": pearson_correlation(
            predicted_array.reshape(-1), actual_array.reshape(-1)
        ),
        "mean_actual": float(actual_array.mean()),
        "mean_predicted": float(predicted_array.mean()),
    }


def contribution_prediction_report(
    predicted_scores: Any,
    actual_targets: Any,
    labels: Any,
    class_counts: Any,
) -> dict[str, Any]:
    """Report per-expert target error for pooled and canonical groups."""
    labels_array = _as_integer_vector(labels, name="labels")
    predicted_array = _validate_score_matrix(
        predicted_scores, num_samples=len(labels_array), name="predicted contribution scores"
    )
    actual_array = _validate_score_matrix(
        actual_targets, num_samples=len(labels_array), name="actual contribution targets"
    )
    groups = _canonical_group_labels(labels_array, np.asarray(class_counts))
    masks = {"pooled": np.ones(len(labels_array), dtype=bool)}
    masks.update({group_name: groups == group_name for group_name in GROUP_NAMES})
    report: dict[str, Any] = {}
    for group_name, mask in masks.items():
        if not np.any(mask):
            raise Task3FFeatureComparisonError(f"no rows are available for {group_name}")
        report[group_name] = {
            "sample_count": int(mask.sum()),
            "experts": {
                expert: _error_metrics(
                    predicted_array[mask, index], actual_array[mask, index]
                )
                for index, expert in enumerate(EXPERT_ORDER)
            },
        }
    return report


def _mean_expert_error(report: Mapping[str, Any], metric: str) -> float:
    values = [float(report["experts"][expert][metric]) for expert in EXPERT_ORDER]
    return float(np.mean(values))


def _error_difference(
    confidence_report: Mapping[str, Any], full_report: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for group_name in ("pooled", *GROUP_NAMES):
        result[group_name] = {
            "experts": {
                expert: {
                    "mse_full_minus_confidence": float(
                        full_report[group_name]["experts"][expert]["mse"]
                        - confidence_report[group_name]["experts"][expert]["mse"]
                    ),
                    "mae_full_minus_confidence": float(
                        full_report[group_name]["experts"][expert]["mae"]
                        - confidence_report[group_name]["experts"][expert]["mae"]
                    ),
                    "pearson_full_minus_confidence": (
                        None
                        if full_report[group_name]["experts"][expert][
                            "pearson_correlation"
                        ]
                        is None
                        or confidence_report[group_name]["experts"][expert][
                            "pearson_correlation"
                        ]
                        is None
                        else float(
                            full_report[group_name]["experts"][expert][
                                "pearson_correlation"
                            ]
                            - confidence_report[group_name]["experts"][expert][
                                "pearson_correlation"
                            ]
                        )
                    ),
                }
                for expert in EXPERT_ORDER
            },
            "mean_expert_mse_full_minus_confidence": float(
                _mean_expert_error(full_report[group_name], "mse")
                - _mean_expert_error(confidence_report[group_name], "mse")
            ),
            "mean_expert_mae_full_minus_confidence": float(
                _mean_expert_error(full_report[group_name], "mae")
                - _mean_expert_error(confidence_report[group_name], "mae")
            ),
        }
    return result


def pairwise_contribution_ranking(
    predicted_scores: Any,
    actual_targets: Any,
    labels: Any,
    class_counts: Any,
) -> dict[str, Any]:
    """Compare both directions of LAL/BS versus Mixup rankings on Tail rows."""
    labels_array = _as_integer_vector(labels, name="labels")
    predicted_array = _validate_score_matrix(
        predicted_scores, num_samples=len(labels_array), name="predicted contribution scores"
    )
    actual_array = _validate_score_matrix(
        actual_targets, num_samples=len(labels_array), name="actual contribution targets"
    )
    group_labels = _canonical_group_labels(labels_array, np.asarray(class_counts))
    tail_mask = group_labels == "tail"
    result: dict[str, Any] = {
        "tail_sample_count": int(tail_mask.sum()),
        "pairs": {},
    }
    mixup_index = EXPERT_ORDER.index(MIXUP)
    for rebalanced in REBALANCED_EXPERTS:
        rebalanced_index = EXPERT_ORDER.index(rebalanced)
        actual_delta = actual_array[tail_mask, rebalanced_index] - actual_array[
            tail_mask, mixup_index
        ]
        predicted_delta = predicted_array[tail_mask, rebalanced_index] - predicted_array[
            tail_mask, mixup_index
        ]
        actual_rebalanced = actual_delta > 0.0
        actual_mixup = actual_delta < 0.0
        actual_tie = actual_delta == 0.0
        predicted_rebalanced = predicted_delta > 0.0
        predicted_mixup = predicted_delta < 0.0
        predicted_tie = predicted_delta == 0.0
        non_tied = ~actual_tie
        correct = (actual_rebalanced & predicted_rebalanced) | (
            actual_mixup & predicted_mixup
        )
        incorrect = non_tied & ~correct
        result["pairs"][f"{rebalanced}_vs_{MIXUP}"] = {
            "actual_rebalanced_higher_count": int(actual_rebalanced.sum()),
            "actual_mixup_higher_count": int(actual_mixup.sum()),
            "actual_tie_count": int(actual_tie.sum()),
            "predicted_rebalanced_higher_count": int(predicted_rebalanced.sum()),
            "predicted_mixup_higher_count": int(predicted_mixup.sum()),
            "predicted_tie_count": int(predicted_tie.sum()),
            "correct_rebalanced_on_actual_rebalanced_higher": int(
                (actual_rebalanced & predicted_rebalanced).sum()
            ),
            "incorrect_mixup_on_actual_rebalanced_higher": int(
                (actual_rebalanced & predicted_mixup).sum()
            ),
            "predicted_tie_on_actual_rebalanced_higher": int(
                (actual_rebalanced & predicted_tie).sum()
            ),
            "correct_mixup_on_actual_mixup_higher": int(
                (actual_mixup & predicted_mixup).sum()
            ),
            "incorrect_rebalanced_on_actual_mixup_higher": int(
                (actual_mixup & predicted_rebalanced).sum()
            ),
            "predicted_tie_on_actual_mixup_higher": int(
                (actual_mixup & predicted_tie).sum()
            ),
            "pairwise_ranking_correct_count": int(correct.sum()),
            "pairwise_ranking_incorrect_count": int(incorrect.sum()),
            "pairwise_ranking_accuracy": (
                None
                if int(non_tied.sum()) == 0
                else float(correct.sum() / non_tied.sum())
            ),
            "pairwise_ranking_denominator_non_tied_actual": int(non_tied.sum()),
        }
    return result


def reproduce_predictions_from_saved_weights(
    logits: Any,
    saved_weights: Any,
) -> np.ndarray:
    """Reproduce top-1 predictions from original logits and saved weights."""
    logits_array = _as_numeric(logits, name="original logits")
    if logits_array.ndim != 3 or logits_array.shape[1] != len(EXPERT_ORDER):
        raise Task3FFeatureComparisonError(
            "original logits must have shape (samples, 4, classes)"
        )
    weights_array = _as_numeric(saved_weights, name="saved weights")
    if weights_array.ndim == 1:
        if weights_array.shape != (len(EXPERT_ORDER),):
            raise Task3FFeatureComparisonError("saved weights must have four experts")
        weights_array = np.repeat(weights_array[None, :], logits_array.shape[0], axis=0)
    if weights_array.shape != (logits_array.shape[0], len(EXPERT_ORDER)):
        raise Task3FFeatureComparisonError("saved weights and logits are misaligned")
    if np.any(weights_array < 0.0) or not np.allclose(
        weights_array.sum(axis=1), 1.0, rtol=0.0, atol=WEIGHT_SUM_ATOL
    ):
        raise Task3FFeatureComparisonError("saved weights are not convex")
    normalized = weights_array / weights_array.sum(axis=1, keepdims=True)
    return combine_weighted_logits(logits_array, normalized).argmax(axis=1).astype(np.int64)


def _validate_weights(weights: Any, *, num_samples: int) -> np.ndarray:
    array = _as_numeric(weights, name="routing weights")
    if array.shape != (num_samples, len(EXPERT_ORDER)):
        raise Task3FFeatureComparisonError(
            f"routing weights must have shape ({num_samples}, {len(EXPERT_ORDER)})"
        )
    if np.any(array < 0.0) or not np.allclose(
        array.sum(axis=1), 1.0, rtol=0.0, atol=WEIGHT_SUM_ATOL
    ):
        raise Task3FFeatureComparisonError("routing weights are not convex")
    return array / array.sum(axis=1, keepdims=True)


def weight_group_statistics(
    weights: Any,
    labels: Any,
    class_counts: Any,
) -> dict[str, Any]:
    """Summarize saved weights by canonical Head/Medium/Tail membership."""
    labels_array = _as_integer_vector(labels, name="labels")
    weights_array = _validate_weights(weights, num_samples=len(labels_array))
    group_labels = _canonical_group_labels(labels_array, np.asarray(class_counts))
    result: dict[str, Any] = {}
    highest = weights_array.argmax(axis=1)
    for group_name in GROUP_NAMES:
        mask = group_labels == group_name
        group_weights = weights_array[mask]
        result[group_name] = {
            "sample_count": int(mask.sum()),
            "mean_weight": {
                expert: float(group_weights[:, index].mean())
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "std_weight": {
                expert: float(group_weights[:, index].std())
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "highest_weight_count": {
                expert: int(np.sum(highest[mask] == index))
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "highest_weight_fraction": {
                expert: float(np.mean(highest[mask] == index))
                for index, expert in enumerate(EXPERT_ORDER)
            },
        }
    return result


def prediction_change_report(
    confidence_predictions: Any,
    full_predictions: Any,
    labels: Any,
    sample_indices: Any,
    class_counts: Any,
) -> dict[str, Any]:
    """Count paired gains/losses and record Tail IDs/classes."""
    confidence = _as_integer_vector(confidence_predictions, name="confidence predictions")
    full = _as_integer_vector(full_predictions, name="full-feature predictions")
    labels_array = _as_integer_vector(labels, name="labels")
    sample_ids = _as_integer_vector(sample_indices, name="sample IDs")
    if not (confidence.shape == full.shape == labels_array.shape == sample_ids.shape):
        raise Task3FFeatureComparisonError("paired predictions and labels are misaligned")
    group_labels = _canonical_group_labels(labels_array, np.asarray(class_counts))
    gained = (confidence != labels_array) & (full == labels_array)
    lost = (confidence == labels_array) & (full != labels_array)
    both_correct = (confidence == labels_array) & (full == labels_array)
    both_incorrect = (confidence != labels_array) & (full != labels_array)
    result: dict[str, Any] = {"by_group": {}, "tail_prediction_records": {"gained": [], "lost": []}}
    for group_name in GROUP_NAMES:
        mask = group_labels == group_name
        result["by_group"][group_name] = {
            "sample_count": int(mask.sum()),
            "full_corrected_confidence_missed": int((mask & gained).sum()),
            "confidence_corrected_full_missed": int((mask & lost).sum()),
            "correct_by_both": int((mask & both_correct).sum()),
            "incorrect_by_both": int((mask & both_incorrect).sum()),
        }
    for name, mask in (("gained", gained & (group_labels == "tail")), ("lost", lost & (group_labels == "tail"))):
        result["tail_prediction_records"][name] = [
            {"sample_id": int(sample_ids[index]), "true_class": int(labels_array[index])}
            for index in np.flatnonzero(mask)
        ]
    return result


def _compact_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "ordinary_accuracy",
        "balanced_accuracy",
        "head_accuracy",
        "medium_accuracy",
        "tail_accuracy",
    )
    return {key: float(metrics[key]) for key in keys}


def _tail_correct_count(labels: np.ndarray, predictions: np.ndarray, class_counts: np.ndarray) -> int:
    group_labels = _canonical_group_labels(labels, class_counts)
    mask = group_labels == "tail"
    return int(np.sum(mask & (predictions == labels)))


def _metric_with_tail_details(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_counts: np.ndarray,
) -> dict[str, Any]:
    metrics = classification_metrics(labels, predictions, class_counts)
    tail_classes = compute_class_groups(np.asarray(class_counts, dtype=np.int64))["tail"]
    return {
        "metrics": metrics,
        "compact_metrics": _compact_metrics(metrics),
        "tail_correct_count": _tail_correct_count(labels, predictions, class_counts),
        "tail_sample_count": int(np.sum(_canonical_group_labels(labels, class_counts) == "tail")),
        "per_class_tail_recall": {
            str(int(class_id)): float(metrics["per_class_recall"][str(int(class_id))])
            for class_id in tail_classes
        },
    }


def validate_saved_task3f_a_metadata(config: Mapping[str, Any]) -> None:
    """Validate the immutable Task 3F-A provenance contract."""
    _require(config.get("task_identifier") == TASK3FA_TASK_IDENTIFIER, "Task 3F-A identifier is incompatible")
    _require(config.get("schema_version") == "task3f_ridge.v1", "Task 3F-A schema is incompatible")
    _require(config.get("expert_order") == list(EXPERT_ORDER), "Task 3F-A expert ordering is incompatible")
    _require(config.get("analyzed_sample_count") == ANALYSIS_SAMPLE_COUNT, "Task 3F-A population is incompatible")
    _require(config.get("permitted_analysis_inner_folds") == list(PERMITTED_ANALYSIS_INNER_FOLDS), "Task 3F-A folds are incompatible")
    _require(config.get("reserved_router_selection_inner_folds") == list(RESERVED_ROUTER_SELECTION_INNER_FOLDS), "Task 3F-A reserved fold declaration is incompatible")
    restrictions = config.get("data_restrictions")
    _require(
        isinstance(restrictions, Mapping)
        and restrictions.get("inner_fold_zero_used") is False
        and restrictions.get("reserved_outer_evaluation_used") is False
        and restrictions.get("original_cifar_test_used") is False,
        "Task 3F-A data restrictions are incompatible",
    )


def _find_unique_index(values: Any, target: str, *, name: str) -> int:
    array = np.asarray(values).astype(str)
    matches = np.flatnonzero(array == target)
    if len(matches) != 1:
        raise Task3FFeatureComparisonError(
            f"{name} must contain exactly one {target!r}, found {len(matches)}"
        )
    return int(matches[0])


def _validate_task3f_a_artifacts(
    *,
    dataset: RestrictedAnalysisDataset,
    config: Mapping[str, Any],
    results: Mapping[str, Any],
    predictions: Mapping[str, np.ndarray],
    scores: Mapping[str, np.ndarray],
    fold_assignments: Mapping[str, Any],
) -> None:
    """Validate Task 3F-A metadata and every saved array contract."""
    validate_saved_task3f_a_metadata(config)
    _require(config.get("analyzed_sample_count") == dataset.num_samples, "Task 3F-A population is incompatible with the loaded dataset")
    _require(
        fold_assignments.get("schema_version") == "task3f_fold_assignments.v1"
        and fold_assignments.get("outer_fold") == OUTER_FOLD
        and fold_assignments.get("permitted_inner_fold_ids") == list(PERMITTED_ANALYSIS_INNER_FOLDS)
        and fold_assignments.get("every_sample_held_out_exactly_once") is True,
        "Task 3F-A fold assignments are incompatible",
    )
    expected_ids = np.asarray(dataset.sample_indices, dtype=np.int64)
    expected_folds = np.asarray(dataset.inner_fold_ids, dtype=np.int64)
    expected_labels = np.asarray(dataset.labels, dtype=np.int64)
    for name, artifact in (("held-out predictions", predictions), ("router scores", scores)):
        for key in ("sample_indices", "inner_fold_ids", "labels"):
            _require(key in artifact, f"Task 3F-A {name} lacks {key}")
        validate_artifact_alignment(
            expected_sample_indices=expected_ids,
            expected_inner_fold_ids=expected_folds,
            expected_labels=expected_labels,
            artifact_sample_indices=artifact["sample_indices"],
            artifact_inner_fold_ids=artifact["inner_fold_ids"],
            artifact_labels=artifact["labels"],
        )

    expected_counts = {
        "baseline": 7,
        "adaptive_ridge": len(FEATURE_SETS) * len(ALPHAS) * len(GAMMAS) * len(TEMPERATURES) * len(SHRINKAGES),
        "global_score_control": len(GAMMAS) * len(TEMPERATURES) * len(SHRINKAGES),
    }
    _require(results.get("configuration_counts") == expected_counts, "Task 3F-A configuration counts are incompatible")
    _require(len(results.get("model_fits", [])) == len(PERMITTED_ANALYSIS_INNER_FOLDS) * len(FEATURE_SETS) * len(ALPHAS) * len(GAMMAS), "Task 3F-A model-fit count is incompatible")
    _require(len(results.get("adaptive_results", [])) == expected_counts["adaptive_ridge"], "Task 3F-A adaptive result count is incompatible")
    _require(len(results.get("baseline_results", [])) == expected_counts["baseline"], "Task 3F-A baseline result count is incompatible")
    _require(len(results.get("global_results", [])) == expected_counts["global_score_control"], "Task 3F-A global result count is incompatible")

    _require(predictions["adaptive_configuration_ids"].shape == (expected_counts["adaptive_ridge"],), "adaptive configuration IDs have the wrong shape")
    _require(predictions["adaptive_predictions"].shape == (expected_counts["adaptive_ridge"], dataset.num_samples), "adaptive predictions have the wrong shape")
    _require(predictions["adaptive_weights"].shape == (expected_counts["adaptive_ridge"], dataset.num_samples, len(EXPERT_ORDER)), "adaptive weights have the wrong shape")
    _require(predictions["global_configuration_ids"].shape == (expected_counts["global_score_control"],), "global configuration IDs have the wrong shape")
    _require(predictions["global_predictions"].shape == (expected_counts["global_score_control"], dataset.num_samples), "global predictions have the wrong shape")
    _require(predictions["global_weights"].shape == (expected_counts["global_score_control"], dataset.num_samples, len(EXPERT_ORDER)), "global weights have the wrong shape")
    _require(scores["model_ids"].shape == (len(PERMITTED_ANALYSIS_INNER_FOLDS) * len(FEATURE_SETS) * len(ALPHAS) * len(GAMMAS),), "Ridge model IDs have the wrong shape")
    _require(scores["held_out_scores"].shape == (len(scores["model_ids"]), dataset.num_samples, len(EXPERT_ORDER)), "held-out Ridge scores have the wrong shape")
    _require(scores["global_score_gammas"].shape == (len(GAMMAS),), "global score gammas have the wrong shape")
    _require(scores["global_scores"].shape == (len(GAMMAS), dataset.num_samples, len(EXPERT_ORDER)), "global scores have the wrong shape")
    for key in ("adaptive_configuration_ids", "global_configuration_ids", "model_ids"):
        values = np.asarray(predictions.get(key, scores.get(key))).astype(str)
        _require(len(np.unique(values)) == len(values), f"{key} contain duplicate IDs")
    _require(np.isfinite(predictions["adaptive_predictions"]).all(), "adaptive predictions contain non-finite values")
    _require(np.isfinite(predictions["global_predictions"]).all(), "global predictions contain non-finite values")
    _require(np.isfinite(predictions["adaptive_weights"]).all(), "adaptive weights contain non-finite values")
    _require(np.isfinite(predictions["global_weights"]).all(), "global weights contain non-finite values")
    held_out_score_array = np.asarray(scores["held_out_scores"])
    if not np.issubdtype(held_out_score_array.dtype, np.number) or np.iscomplexobj(held_out_score_array):
        raise Task3FFeatureComparisonError("held-out scores must contain real numeric values")
    _require(
        np.isfinite(held_out_score_array).all() or np.isnan(held_out_score_array).any(),
        "held-out scores contain invalid values",
    )
    _require(
        np.logical_or(np.isfinite(held_out_score_array), np.isnan(held_out_score_array)).all(),
        "held-out scores contain infinities",
    )
    _require(np.isfinite(scores["global_scores"]).all(), "global scores contain non-finite values")


def identify_matched_model_fits(
    model_fit_records: Sequence[Mapping[str, Any]],
    score_model_ids: Sequence[str] | None = None,
) -> dict[tuple[str, float, float], dict[int, Mapping[str, Any]]]:
    """Identify exactly one saved held-out fit per fold/feature/alpha/gamma."""
    expected: dict[tuple[str, float, float], dict[int, Mapping[str, Any]]] = {}
    for record in model_fit_records:
        feature_set = str(record.get("feature_set"))
        alpha = float(record.get("alpha"))
        gamma = float(record.get("gamma"))
        fold = int(record.get("validation_inner_fold"))
        key = (feature_set, alpha, gamma)
        if feature_set not in FEATURE_SETS or alpha not in ALPHAS or gamma not in GAMMAS:
            raise Task3FFeatureComparisonError("Task 3F-A contains an out-of-grid Ridge model")
        if fold not in PERMITTED_ANALYSIS_INNER_FOLDS:
            raise Task3FFeatureComparisonError("Task 3F-A contains a non-permitted validation fold")
        expected.setdefault(key, {})
        if fold in expected[key]:
            raise Task3FFeatureComparisonError(
                f"duplicate Ridge model for {feature_set}, alpha={alpha}, gamma={gamma}, fold={fold}"
            )
        expected[key][fold] = record
        expected_key = _adaptive_model_key(fold, feature_set, alpha, gamma)
        if record.get("model_key") != expected_key:
            raise Task3FFeatureComparisonError(
                f"Ridge model ID does not match its saved configuration: {record.get('model_key')}"
            )
    expected_keys = {
        (feature_set, float(alpha), float(gamma))
        for feature_set in FEATURE_SETS
        for alpha in ALPHAS
        for gamma in GAMMAS
    }
    if set(expected) != expected_keys:
        missing = sorted(expected_keys - set(expected), key=str)
        extra = sorted(set(expected) - expected_keys, key=str)
        raise Task3FFeatureComparisonError(
            f"missing or incompatible matched Ridge configurations; missing={missing}, extra={extra}"
        )
    for key, by_fold in expected.items():
        if set(by_fold) != set(PERMITTED_ANALYSIS_INNER_FOLDS):
            raise Task3FFeatureComparisonError(f"Ridge configuration {key} is missing a validation fold")
    if score_model_ids is not None:
        actual_ids = {str(value) for value in score_model_ids}
        expected_ids = {
            _adaptive_model_key(fold, feature_set, alpha, gamma)
            for feature_set, alpha, gamma in expected_keys
            for fold in PERMITTED_ANALYSIS_INNER_FOLDS
        }
        if actual_ids != expected_ids or len(actual_ids) != len(expected_ids):
            raise Task3FFeatureComparisonError("saved score model IDs do not match the complete Ridge grid")
    return expected


def _compose_saved_scores(
    scores: Mapping[str, np.ndarray],
    dataset: RestrictedAnalysisDataset,
    model_key_by_fold: Mapping[int, str],
) -> np.ndarray:
    model_ids = np.asarray(scores["model_ids"]).astype(str)
    raw = np.asarray(scores["held_out_scores"])
    composed = np.full((dataset.num_samples, len(EXPERT_ORDER)), np.nan, dtype=np.float64)
    for fold, model_key in model_key_by_fold.items():
        index = _find_unique_index(model_ids, model_key, name="saved Ridge model IDs")
        raw_row = np.asarray(raw[index])
        if not np.issubdtype(raw_row.dtype, np.number) or np.iscomplexobj(raw_row):
            raise Task3FFeatureComparisonError(
                f"saved scores for {model_key} must contain real numeric values"
            )
        validation_mask = dataset.inner_fold_ids == fold
        finite = np.isfinite(raw_row).all(axis=1)
        # The saved arrays intentionally contain NaN outside each validation fold.
        if not np.array_equal(finite, validation_mask):
            raise Task3FFeatureComparisonError(
                f"saved scores for {model_key} are not held out only on fold {fold}"
            )
        if not np.isnan(raw_row[~validation_mask]).all():
            raise Task3FFeatureComparisonError(
                f"saved scores for {model_key} use a non-NaN placeholder outside fold {fold}"
            )
        selected = raw_row[validation_mask].astype(np.float64, copy=False)
        if not np.isfinite(selected).all():
            raise Task3FFeatureComparisonError(
                f"saved scores for {model_key} contain non-finite held-out values"
            )
        composed[validation_mask] = selected
    if not np.isfinite(composed).all():
        raise Task3FFeatureComparisonError("composed saved Ridge scores are incomplete")
    return composed


def _configuration_row(
    rows: Sequence[Mapping[str, Any]],
    configuration_id: str,
    *,
    name: str,
) -> Mapping[str, Any]:
    matches = [row for row in rows if row.get("configuration_id") == configuration_id]
    if len(matches) != 1:
        raise Task3FFeatureComparisonError(
            f"{name} must contain exactly one {configuration_id!r}, found {len(matches)}"
        )
    return matches[0]


def _validate_target_diagnostics(
    target_results: Mapping[str, Any],
    *,
    source_hashes: Mapping[str, Mapping[str, Any]],
    ridge_hashes: Mapping[str, Mapping[str, Any]],
) -> None:
    _require(target_results.get("task_identifier") == TASK3FE_TASK_IDENTIFIER, "Task 3F-E identifier is incompatible")
    _require(target_results.get("schema_version") == "task3f_target_diagnostics.v1", "Task 3F-E schema is incompatible")
    population = target_results.get("population", {})
    _require(population.get("sample_count") == ANALYSIS_SAMPLE_COUNT, "Task 3F-E population is incompatible")
    provenance = target_results.get("provenance", {})
    for key in (
        "inner_fold_zero_used",
        "reserved_outer_evaluation_used",
        "original_cifar_test_used",
        "original_full_data_predictions_used",
        "experts_retrained",
        "router_refit",
        "oracle_weights_used",
        "source_oof_logits_modified",
        "expert_weights_modified",
    ):
        _require(provenance.get(key) is False, f"Task 3F-E provenance flag {key} is incompatible")
    _require(provenance.get("labels_used_only_for_retrospective_diagnostics") is True, "Task 3F-E label-use declaration is incompatible")
    _require(provenance.get("expert_order") == list(EXPERT_ORDER), "Task 3F-E expert order is incompatible")
    target_source_hashes = provenance.get("task3c_oof_artifacts", {})
    ridge_source_hashes = provenance.get("task3f_a_artifacts", {})
    _require(
        all(target_source_hashes.get(name, {}).get("sha256") == details["sha256"] for name, details in source_hashes.items()),
        "Task 3F-E does not reference the current Task 3C source hashes",
    )
    _require(
        all(ridge_source_hashes.get(name, {}).get("sha256") == details["sha256"] for name, details in ridge_hashes.items()),
        "Task 3F-E does not reference the current Task 3F-A source hashes",
    )


def _reconstruct_saved_training_scores(
    record: Mapping[str, Any],
    logits: np.ndarray,
    *,
    feature_set: str,
) -> np.ndarray:
    features = extract_features(logits, feature_set)
    scaler_mean = _as_numeric(record.get("scaler_mean"), name="saved scaler mean")
    scaler_scale = _as_numeric(record.get("scaler_scale"), name="saved scaler scale")
    coefficients = _as_numeric(record.get("coefficients"), name="saved Ridge coefficients")
    intercept = _as_numeric(record.get("intercept"), name="saved Ridge intercept")
    _require(scaler_mean.shape == (features.shape[1],), "saved scaler mean has the wrong shape")
    _require(scaler_scale.shape == (features.shape[1],), "saved scaler scale has the wrong shape")
    _require(coefficients.shape == (len(EXPERT_ORDER), features.shape[1]), "saved Ridge coefficients have the wrong shape")
    _require(intercept.shape == (len(EXPERT_ORDER),), "saved Ridge intercept has the wrong shape")
    _require(np.all(scaler_scale > 0.0), "saved scaler scale must be positive")
    scores = (features - scaler_mean[None, :]) / scaler_scale[None, :]
    scores = scores @ coefficients.T + intercept[None, :]
    if not np.isfinite(scores).all():
        raise Task3FFeatureComparisonError("reconstructed saved training scores are non-finite")
    return scores


def _training_vs_held_out_diagnostics(
    *,
    dataset: RestrictedAnalysisDataset,
    class_counts: np.ndarray,
    targets: np.ndarray,
    scores: Mapping[str, np.ndarray],
    fold_assignments: Mapping[str, Any],
    matched_models: Mapping[tuple[str, float, float], Mapping[int, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    sample_to_position = {int(sample_id): position for position, sample_id in enumerate(dataset.sample_indices)}
    diagnostics: list[dict[str, Any]] = []
    score_model_ids = np.asarray(scores["model_ids"]).astype(str)
    for key in sorted(matched_models, key=str):
        feature_set, alpha, gamma = key
        for fold in PERMITTED_ANALYSIS_INNER_FOLDS:
            record = matched_models[key][fold]
            assignment = fold_assignments["folds"][str(fold)]
            train_ids = _as_integer_vector(assignment["training_sample_ids"], name="training sample IDs")
            validation_ids = _as_integer_vector(assignment["validation_sample_ids"], name="validation sample IDs")
            try:
                train_positions = np.asarray([sample_to_position[int(value)] for value in train_ids], dtype=np.int64)
                validation_positions = np.asarray([sample_to_position[int(value)] for value in validation_ids], dtype=np.int64)
            except KeyError as exc:
                raise Task3FFeatureComparisonError("saved fold assignments contain an unknown sample ID") from exc
            expected_train = np.flatnonzero(dataset.inner_fold_ids != fold)
            expected_validation = np.flatnonzero(dataset.inner_fold_ids == fold)
            _require(np.array_equal(train_positions, expected_train), f"saved training sample IDs are incompatible for fold {fold}")
            _require(np.array_equal(validation_positions, expected_validation), f"saved validation sample IDs are incompatible for fold {fold}")
            training_scores = _reconstruct_saved_training_scores(
                record,
                dataset.logits[train_positions],
                feature_set=feature_set,
            )
            model_key = str(record["model_key"])
            model_index = _find_unique_index(score_model_ids, model_key, name="saved Ridge model IDs")
            raw_validation_scores = np.asarray(scores["held_out_scores"])[model_index, validation_positions]
            validation_scores = _as_numeric(raw_validation_scores, name="saved held-out Ridge scores")
            training_report = contribution_prediction_report(
                training_scores,
                targets[train_positions],
                dataset.labels[train_positions],
                class_counts,
            )
            validation_report = contribution_prediction_report(
                validation_scores,
                targets[validation_positions],
                dataset.labels[validation_positions],
                class_counts,
            )
            reconstructed_mse = _mean_expert_error(training_report["pooled"], "mse")
            saved_training_mse = float(record["training_mse"])
            saved_held_out_mse = float(record["held_out_mse"])
            recomputed_held_out_mse = _mean_expert_error(validation_report["pooled"], "mse")
            _require(
                abs(reconstructed_mse - saved_training_mse) <= MODEL_RECONSTRUCTION_ATOL,
                f"saved training diagnostic cannot be reconstructed for {model_key}",
            )
            _require(
                abs(recomputed_held_out_mse - saved_held_out_mse) <= MODEL_RECONSTRUCTION_ATOL,
                f"saved held-out diagnostic cannot be reconstructed for {model_key}",
            )
            diagnostics.append(
                {
                    "model_key": model_key,
                    "feature_set": feature_set,
                    "alpha": float(alpha),
                    "gamma": float(gamma),
                    "validation_inner_fold": int(fold),
                    "training_sample_count": int(len(train_positions)),
                    "validation_sample_count": int(len(validation_positions)),
                    "saved_training_mse": saved_training_mse,
                    "reconstructed_training_mse": reconstructed_mse,
                    "saved_held_out_mse": saved_held_out_mse,
                    "recomputed_held_out_mse": recomputed_held_out_mse,
                    "held_out_minus_training_mse": float(saved_held_out_mse - saved_training_mse),
                    "training_report": training_report,
                    "held_out_report": validation_report,
                    "tail_held_out_mse_by_expert": {
                        expert: float(validation_report["tail"]["experts"][expert]["mse"])
                        for expert in EXPERT_ORDER
                    },
                    "tail_held_out_mean_expert_mse": _mean_expert_error(validation_report["tail"], "mse"),
                }
            )
    return diagnostics


def _fold_tail_accuracy_comparisons(
    *,
    adaptive_rows: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, np.ndarray],
    class_counts: np.ndarray,
    labels: np.ndarray,
    inner_fold_ids: np.ndarray,
) -> list[dict[str, Any]]:
    ids = np.asarray(predictions["adaptive_configuration_ids"]).astype(str)
    pred_array = np.asarray(predictions["adaptive_predictions"])
    row_index = {identifier: index for index, identifier in enumerate(ids)}
    result: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        for gamma in GAMMAS:
            for temperature in TEMPERATURES:
                for shrinkage in SHRINKAGES:
                    params = (float(alpha), float(gamma), float(temperature), float(shrinkage))
                    confidence_id = _adaptive_configuration_id(FEATURE_SET_CONFIDENCE, *params)
                    full_id = _adaptive_configuration_id(FEATURE_SET_FULL, *params)
                    _require(confidence_id in row_index and full_id in row_index, "matched saved classification configuration is missing")
                    confidence_predictions = pred_array[row_index[confidence_id]]
                    full_predictions = pred_array[row_index[full_id]]
                    by_fold: dict[str, Any] = {}
                    for fold in PERMITTED_ANALYSIS_INNER_FOLDS:
                        mask = inner_fold_ids == fold
                        confidence_metrics = classification_metrics(labels[mask], confidence_predictions[mask], class_counts)
                        full_metrics = classification_metrics(labels[mask], full_predictions[mask], class_counts)
                        by_fold[str(fold)] = {
                            "confidence_only": _compact_metrics(confidence_metrics),
                            "full_13": _compact_metrics(full_metrics),
                            "full_minus_confidence": {
                                metric: float(full_metrics[metric] - confidence_metrics[metric])
                                for metric in ("balanced_accuracy", "head_accuracy", "medium_accuracy", "tail_accuracy")
                            },
                        }
                    result.append(
                        {
                            "alpha": float(alpha),
                            "gamma": float(gamma),
                            "temperature": float(temperature),
                            "shrinkage": float(shrinkage),
                            "confidence_configuration_id": confidence_id,
                            "full_configuration_id": full_id,
                            "folds": by_fold,
                        }
                    )
    return result


def _validate_saved_hashes(
    results: Mapping[str, Any],
    predictions: Mapping[str, np.ndarray],
) -> None:
    adaptive_ids = np.asarray(predictions["adaptive_configuration_ids"]).astype(str)
    adaptive_predictions = np.asarray(predictions["adaptive_predictions"])
    adaptive_weights = np.asarray(predictions["adaptive_weights"])
    for row in results["adaptive_results"]:
        index = _find_unique_index(adaptive_ids, str(row["configuration_id"]), name="adaptive configuration IDs")
        _require(
            row.get("prediction_sha256") == _sha256_array(adaptive_predictions[index].astype(np.int16)),
            f"saved adaptive prediction hash does not match {row['configuration_id']}",
        )
        _require(
            row.get("weight_sha256") == _sha256_array(adaptive_weights[index].astype(np.float32)),
            f"saved adaptive weight hash does not match {row['configuration_id']}",
        )
    global_ids = np.asarray(predictions["global_configuration_ids"]).astype(str)
    global_predictions = np.asarray(predictions["global_predictions"])
    global_weights = np.asarray(predictions["global_weights"])
    for row in results["global_results"]:
        index = _find_unique_index(global_ids, str(row["configuration_id"]), name="global configuration IDs")
        _require(
            row.get("prediction_sha256") == _sha256_array(global_predictions[index].astype(np.int16)),
            f"saved global prediction hash does not match {row['configuration_id']}",
        )
        _require(
            row.get("weight_sha256") == _sha256_array(global_weights[index].astype(np.float32)),
            f"saved global weight hash does not match {row['configuration_id']}",
        )


def _source_file_records(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    return {
        name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }


def _verify_source_records(records: Mapping[str, Mapping[str, Any]]) -> None:
    for name, details in records.items():
        path = Path(str(details["path"]))
        _require(path.exists(), f"source artifact disappeared during diagnostics: {path}")
        _require(
            sha256_file(path) == str(details["sha256"]),
            f"source artifact changed during diagnostics: {name}",
        )


def _build_grid_comparisons(
    *,
    dataset: RestrictedAnalysisDataset,
    class_counts: np.ndarray,
    matched_scores: Mapping[tuple[str, float, float], np.ndarray],
    contribution_reports: Mapping[tuple[str, float, float], Mapping[str, Any]],
    ranking_reports: Mapping[tuple[str, float, float], Mapping[str, Any]],
    predictions: Mapping[str, np.ndarray],
    results: Mapping[str, Any],
) -> list[dict[str, Any]]:
    adaptive_ids = np.asarray(predictions["adaptive_configuration_ids"]).astype(str)
    adaptive_predictions = np.asarray(predictions["adaptive_predictions"])
    adaptive_weights = np.asarray(predictions["adaptive_weights"])
    row_by_id = {
        str(row["configuration_id"]): row for row in results["adaptive_results"]
    }
    index_by_id = {identifier: index for index, identifier in enumerate(adaptive_ids)}
    grid: list[dict[str, Any]] = []
    for alpha in ALPHAS:
        for gamma in GAMMAS:
            score_key_confidence = (FEATURE_SET_CONFIDENCE, float(alpha), float(gamma))
            score_key_full = (FEATURE_SET_FULL, float(alpha), float(gamma))
            confidence_scores = matched_scores[score_key_confidence]
            full_scores = matched_scores[score_key_full]
            score_delta = np.abs(full_scores - confidence_scores)
            for temperature in TEMPERATURES:
                for shrinkage in SHRINKAGES:
                    confidence_id = _adaptive_configuration_id(
                        FEATURE_SET_CONFIDENCE, alpha, gamma, temperature, shrinkage
                    )
                    full_id = _adaptive_configuration_id(
                        FEATURE_SET_FULL, alpha, gamma, temperature, shrinkage
                    )
                    _require(confidence_id in index_by_id and full_id in index_by_id, "saved matched grid configuration is missing")
                    confidence_index = index_by_id[confidence_id]
                    full_index = index_by_id[full_id]
                    confidence_predictions = adaptive_predictions[confidence_index].astype(np.int64)
                    full_predictions = adaptive_predictions[full_index].astype(np.int64)
                    confidence_weights = _validate_weights(adaptive_weights[confidence_index], num_samples=dataset.num_samples)
                    full_weights = _validate_weights(adaptive_weights[full_index], num_samples=dataset.num_samples)
                    confidence_metrics = classification_metrics(dataset.labels, confidence_predictions, class_counts)
                    full_metrics = classification_metrics(dataset.labels, full_predictions, class_counts)
                    confidence_row = row_by_id[confidence_id]
                    full_row = row_by_id[full_id]
                    confidence_contribution = contribution_reports[score_key_confidence]
                    full_contribution = contribution_reports[score_key_full]
                    grid.append(
                        {
                            "alpha": float(alpha),
                            "gamma": float(gamma),
                            "temperature": float(temperature),
                            "shrinkage": float(shrinkage),
                            "confidence_only": {
                                "configuration_id": confidence_id,
                                "compact_metrics": _compact_metrics(confidence_metrics),
                                "saved_task3f_a_metrics": _compact_metrics(confidence_row["metrics"]),
                                "mean_weight": confidence_weights.mean(axis=0).tolist(),
                                "tail_contribution_error": {
                                    "mse_by_expert": {
                                        expert: float(confidence_contribution["tail"]["experts"][expert]["mse"])
                                        for expert in EXPERT_ORDER
                                    },
                                    "mae_by_expert": {
                                        expert: float(confidence_contribution["tail"]["experts"][expert]["mae"])
                                        for expert in EXPERT_ORDER
                                    },
                                    "mean_expert_mse": _mean_expert_error(confidence_contribution["tail"], "mse"),
                                    "mean_expert_mae": _mean_expert_error(confidence_contribution["tail"], "mae"),
                                },
                            },
                            "full_13": {
                                "configuration_id": full_id,
                                "compact_metrics": _compact_metrics(full_metrics),
                                "saved_task3f_a_metrics": _compact_metrics(full_row["metrics"]),
                                "mean_weight": full_weights.mean(axis=0).tolist(),
                                "tail_contribution_error": {
                                    "mse_by_expert": {
                                        expert: float(full_contribution["tail"]["experts"][expert]["mse"])
                                        for expert in EXPERT_ORDER
                                    },
                                    "mae_by_expert": {
                                        expert: float(full_contribution["tail"]["experts"][expert]["mae"])
                                        for expert in EXPERT_ORDER
                                    },
                                    "mean_expert_mse": _mean_expert_error(full_contribution["tail"], "mse"),
                                    "mean_expert_mae": _mean_expert_error(full_contribution["tail"], "mae"),
                                },
                            },
                            "differences_full_minus_confidence": {
                                "balanced_accuracy": float(full_metrics["balanced_accuracy"] - confidence_metrics["balanced_accuracy"]),
                                "head_accuracy": float(full_metrics["head_accuracy"] - confidence_metrics["head_accuracy"]),
                                "medium_accuracy": float(full_metrics["medium_accuracy"] - confidence_metrics["medium_accuracy"]),
                                "tail_accuracy": float(full_metrics["tail_accuracy"] - confidence_metrics["tail_accuracy"]),
                                "mean_mixup_weight": float(full_weights[:, EXPERT_ORDER.index(MIXUP)].mean() - confidence_weights[:, EXPERT_ORDER.index(MIXUP)].mean()),
                                "tail_mean_expert_mse": float(_mean_expert_error(full_contribution["tail"], "mse") - _mean_expert_error(confidence_contribution["tail"], "mse")),
                                "tail_mean_expert_mae": float(_mean_expert_error(full_contribution["tail"], "mae") - _mean_expert_error(confidence_contribution["tail"], "mae")),
                                "score_mean_absolute_difference": float(score_delta.mean()),
                                "score_max_absolute_difference": float(score_delta.max()),
                                "prediction_changed_count": int(np.sum(full_predictions != confidence_predictions)),
                                "prediction_identical": bool(np.array_equal(full_predictions, confidence_predictions)),
                                "weights_mean_absolute_difference": float(np.abs(full_weights - confidence_weights).mean()),
                                "weights_max_absolute_difference": float(np.abs(full_weights - confidence_weights).max()),
                            },
                            "pairwise_ranking": {
                                "confidence_only": ranking_reports[score_key_confidence],
                                "full_13": ranking_reports[score_key_full],
                            },
                        }
                    )
    return grid


def _render_summary(results: Mapping[str, Any]) -> str:
    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{100.0 * value:.2f}%"

    def pp(value: float) -> str:
        return f"{100.0 * value:+.4f} pp"

    primary = results["primary_comparison"]
    classification = primary["classification"]
    conf_method = classification[FEATURE_SET_CONFIDENCE]
    full_method = classification[FEATURE_SET_FULL]
    conf_weight_method = primary["weight_statistics"][FEATURE_SET_CONFIDENCE]
    full_weight_method = primary["weight_statistics"][FEATURE_SET_FULL]
    contribution_summary = results["overall_patterns"]["contribution_error"]
    grid_summary = results["overall_patterns"]["classification_grid"]
    tail_ranking = primary["pairwise_ranking"]
    training_summary = results["overall_patterns"]["training_vs_held_out"]
    conf_tail = tail_ranking[FEATURE_SET_CONFIDENCE]["pairs"]["LAL_vs_Mixup"]
    full_tail = tail_ranking[FEATURE_SET_FULL]["pairs"]["LAL_vs_Mixup"]
    conf_balanced_tail = tail_ranking[FEATURE_SET_CONFIDENCE]["pairs"]["BalancedSoftmax_vs_Mixup"]
    full_balanced_tail = tail_ranking[FEATURE_SET_FULL]["pairs"]["BalancedSoftmax_vs_Mixup"]
    primary_fold_tail_rows = next(
        row
        for row in results["training_vs_held_out"]["tail_classification_accuracy_by_fold"]
        if row["alpha"] == PRIMARY_ALPHA
        and row["gamma"] == PRIMARY_GAMMA
        and row["temperature"] == PRIMARY_TEMPERATURE
        and row["shrinkage"] == PRIMARY_SHRINKAGE
    )
    primary_fold_tail_deltas = ", ".join(
        f"fold {fold}: {pp(primary_fold_tail_rows['folds'][str(fold)]['full_minus_confidence']['tail_accuracy'])}"
        for fold in PERMITTED_ANALYSIS_INNER_FOLDS
    )
    lines = [
        "# Task 3F-F — Ridge Feature Comparison on Tail Images",
        "",
        "Exploratory, read-only comparison of the existing Task 3F-A confidence-only and full 13-feature Ridge results.",
        "",
        "## Scope and safeguards",
        "",
        f"- Population: **{results['population']['sample_count']}** OOF rows from outer fold 0, inner folds 1–3; Head/Medium/Tail counts are **Head={results['population']['true_group_counts']['head']}, Medium={results['population']['true_group_counts']['medium']}, Tail={results['population']['true_group_counts']['tail']}**.",
        "- Saved held-out scores, predictions, weights, and model diagnostics were loaded; Ridge was not refit and experts were not retrained.",
        "- Inner fold 0, the reserved outer-evaluation population, the original CIFAR-100 test set, and oracle weights were not used.",
        "- Labels and canonical groups appear only in retrospective target and outcome diagnostics.",
        "",
        "## Answers to the research questions",
        "",
        f"1. **Contribution prediction:** across the 15 matched alpha/gamma settings, the full-feature model had lower pooled mean-expert MSE in **{contribution_summary['pooled_full_lower_mse_count']}/{contribution_summary['matched_alpha_gamma_count']}** settings and lower Tail MSE in **{contribution_summary['tail_full_lower_mse_count']}/{contribution_summary['matched_alpha_gamma_count']}**. The mean full-minus-confidence pooled MSE was **{contribution_summary['pooled_mean_mse_delta']:+.6f}**; Tail was **{contribution_summary['tail_mean_mse_delta']:+.6f}**.",
        f"2. **Tail ranking:** on the primary configuration's **{tail_ranking[FEATURE_SET_CONFIDENCE]['tail_sample_count']}** Tail rows, actual LAL > Mixup occurred **{conf_tail['actual_rebalanced_higher_count']}** times; confidence-only/full ranked LAL higher **{conf_tail['correct_rebalanced_on_actual_rebalanced_higher']}/{full_tail['correct_rebalanced_on_actual_rebalanced_higher']}** times. For BalancedSoftmax, the corresponding counts were **{conf_balanced_tail['actual_rebalanced_higher_count']}** actual opportunities and **{conf_balanced_tail['correct_rebalanced_on_actual_rebalanced_higher']}/{full_balanced_tail['correct_rebalanced_on_actual_rebalanced_higher']}** correct rankings. Across both ranking directions, LAL pairwise accuracy was **{pct(conf_tail['pairwise_ranking_accuracy'])}** versus **{pct(full_tail['pairwise_ranking_accuracy'])}**, and BalancedSoftmax was **{pct(conf_balanced_tail['pairwise_ranking_accuracy'])}** versus **{pct(full_balanced_tail['pairwise_ranking_accuracy'])}**.",
        f"3. **Classification:** primary Balanced Accuracy changed from **{pct(conf_method['compact_metrics']['balanced_accuracy'])}** to **{pct(full_method['compact_metrics']['balanced_accuracy'])}** ({pp(full_method['compact_metrics']['balanced_accuracy'] - conf_method['compact_metrics']['balanced_accuracy'])}); Tail changed from **{pct(conf_method['compact_metrics']['tail_accuracy'])}** to **{pct(full_method['compact_metrics']['tail_accuracy'])}** ({pp(full_method['compact_metrics']['tail_accuracy'] - conf_method['compact_metrics']['tail_accuracy'])}). Full-feature gains/losses on Tail were **{primary['prediction_changes']['by_group']['tail']['full_corrected_confidence_missed']} / {primary['prediction_changes']['by_group']['tail']['confidence_corrected_full_missed']}**.",
        f"4. **Mixup preference:** primary Tail mean Mixup weight changed from **{conf_weight_method['tail']['mean_weight'][MIXUP]:.4f}** to **{full_weight_method['tail']['mean_weight'][MIXUP]:.4f}**; its highest-weight fraction changed from **{pct(conf_weight_method['tail']['highest_weight_fraction'][MIXUP])}** to **{pct(full_weight_method['tail']['highest_weight_fraction'][MIXUP])}**.",
        f"5. **Training-data interpretation:** the saved/reconstructed training-versus-held-out diagnostics show full-feature lower training MSE in **{training_summary['full_lower_training_mse_count']}/{training_summary['matched_model_fit_count']}** matched model fits and lower held-out MSE in **{training_summary['full_lower_held_out_mse_count']}/{training_summary['matched_model_fit_count']}**. This is compatible with feature benefit and possible overfitting, but the sparse Tail population does not identify insufficient training data as the cause.",
        f"   At the predefined primary setting, the fold-level Tail-accuracy differences were **{primary_fold_tail_deltas}**, showing that the pooled Tail decrease was concentrated in fold 3 rather than uniform across all three folds.",
        "",
        "These are verified measurements on one development population. Lower contribution-target error is not evidence of better classification, and no new router or final configuration is selected here.",
        "",
        "## Primary matched classification comparison",
        "",
        "| Method | Accuracy | BA | Head | Medium | Tail | Correct Tail images |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method_name in (
        "uniform_logit",
        "global_control",
        "uniform_probability",
        "fixed_006",
        "fixed_007",
        "fixed_010",
        "fixed_011",
        "uniform_without_ce",
        FEATURE_SET_CONFIDENCE,
        FEATURE_SET_FULL,
    ):
        method = classification[method_name]
        metrics = method["compact_metrics"]
        lines.append(
            f"| {method_name} | {pct(metrics['ordinary_accuracy'])} | {pct(metrics['balanced_accuracy'])} | {pct(metrics['head_accuracy'])} | {pct(metrics['medium_accuracy'])} | {pct(metrics['tail_accuracy'])} | {method['tail_correct_count']} |"
        )
    lines.extend(
        [
            "",
            "## Grid-level pattern",
            "",
            f"The complete matched grid contains **{results['configuration_grid']['count']}** alpha/gamma/temperature/shrinkage pairs. Full features improved BA in **{grid_summary['full_better_balanced_accuracy_count']}**, Tail in **{grid_summary['full_better_tail_accuracy_count']}**, and produced identical top-1 predictions in **{grid_summary['identical_prediction_count']}** pairs.",
            f"The mean full-minus-confidence Tail accuracy difference across the grid was **{pp(grid_summary['mean_tail_accuracy_delta'])}**; the mean Mixup-weight difference was **{grid_summary['mean_mixup_weight_delta']:+.6f}**.",
            "",
            "## Limitations",
            "",
            "The Tail group has only 183 rows, with shared OOF training dependence across folds and one expert seed/outer fold. The target is label-dependent and retrospective; the inference-time feature sets themselves do not use labels. These results cannot distinguish feature insufficiency from insufficient Tail training data, and they do not establish independent or full-data-expert performance.",
            "",
        ]
    )
    return "\n".join(lines)


def _git_commit(project_root: Path) -> str | None:
    return ArtifactReader(error_type=Task3FFeatureComparisonError).git_commit(
        project_root, required=False
    )


def run_task3f_feature_comparison(
    *,
    data_root: str | Path = "./data",
    oof_directory: str | Path = "artifacts/oof/task3c_oof",
    ridge_directory: str | Path = "artifacts/oof/task3f_ridge",
    target_directory: str | Path = "artifacts/oof/task3f_target_diagnostics",
    output_directory: str | Path = "artifacts/oof/task3f_feature_comparison",
    project_root: str | Path | None = None,
) -> dict[str, Path]:
    """Run Task 3F-F from existing artifacts and write protected outputs."""
    project_root_path = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    source_directory = Path(oof_directory).resolve()
    ridge_path = Path(ridge_directory).resolve()
    target_path = Path(target_directory).resolve()
    output_path = Path(output_directory).resolve()
    for source in (source_directory, ridge_path, target_path):
        if output_path == source or source in output_path.parents:
            raise Task3FFeatureComparisonError(
                "output directory must be separate from and outside source artifact directories"
            )

    manager = NestedOOFFoldManager.from_canonical_training_data(
        data_root,
        seed=FOLD_GENERATION_SEED,
        outer_folds=5,
        inner_folds=4,
        expert_order=EXPERT_ORDER,
    )
    dataset, task3c_source_files = load_restricted_analysis_dataset(source_directory, manager)
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
    target_files = {
        "diagnostic_results": target_path / "diagnostic_results.json",
        "summary": target_path / "summary.md",
    }
    for name, path in {**ridge_files, **target_files}.items():
        _require(path.exists(), f"missing source artifact {name}: {path}")
    source_paths = {
        **{
            f"task3c_{name}": Path(details["path"])
            for name, details in task3c_source_files.items()
        },
        **{f"task3f_a_{name}": path for name, path in ridge_files.items()},
        **{f"task3f_e_{name}": path for name, path in target_files.items()},
    }
    source_hashes_before = _source_file_records(source_paths)

    config = _load_json(ridge_files["experiment_config"], name="Task 3F-A experiment_config")
    ridge_results = _load_json(ridge_files["ridge_results"], name="Task 3F-A ridge_results")
    predictions = _load_npz(ridge_files["held_out_predictions"], name="Task 3F-A held_out_predictions")
    scores = _load_npz(ridge_files["router_scores"], name="Task 3F-A router_scores")
    fold_assignments = _load_json(ridge_files["fold_assignments"], name="Task 3F-A fold_assignments")
    target_results = _load_json(target_files["diagnostic_results"], name="Task 3F-E diagnostic_results")
    _validate_task3f_a_artifacts(
        dataset=dataset,
        config=config,
        results=ridge_results,
        predictions=predictions,
        scores=scores,
        fold_assignments=fold_assignments,
    )
    _validate_target_diagnostics(
        target_results,
        source_hashes={name.removeprefix("task3c_"): details for name, details in source_hashes_before.items() if name.startswith("task3c_")},
        ridge_hashes={name.removeprefix("task3f_a_"): details for name, details in source_hashes_before.items() if name.startswith("task3f_a_")},
    )
    _validate_saved_hashes(ridge_results, predictions)
    class_counts = np.asarray(manager.canonical_class_counts, dtype=np.int64)
    targets = compute_contribution_targets(dataset.logits, dataset.labels)

    matched_models = identify_matched_model_fits(
        ridge_results["model_fits"],
        np.asarray(scores["model_ids"]).astype(str),
    )
    matched_scores: dict[tuple[str, float, float], np.ndarray] = {}
    contribution_reports: dict[tuple[str, float, float], dict[str, Any]] = {}
    ranking_reports: dict[tuple[str, float, float], dict[str, Any]] = {}
    for key, records_by_fold in matched_models.items():
        feature_set, alpha, gamma = key
        model_key_by_fold = {
            fold: str(records_by_fold[fold]["model_key"])
            for fold in PERMITTED_ANALYSIS_INNER_FOLDS
        }
        matched_scores[key] = _compose_saved_scores(scores, dataset, model_key_by_fold)
        contribution_reports[key] = contribution_prediction_report(
            matched_scores[key], targets, dataset.labels, class_counts
        )
        ranking_reports[key] = pairwise_contribution_ranking(
            matched_scores[key], targets, dataset.labels, class_counts
        )

    grid = _build_grid_comparisons(
        dataset=dataset,
        class_counts=class_counts,
        matched_scores=matched_scores,
        contribution_reports=contribution_reports,
        ranking_reports=ranking_reports,
        predictions=predictions,
        results=ridge_results,
    )
    primary_confidence_key = (FEATURE_SET_CONFIDENCE, PRIMARY_ALPHA, PRIMARY_GAMMA)
    primary_full_key = (FEATURE_SET_FULL, PRIMARY_ALPHA, PRIMARY_GAMMA)
    primary_confidence_id = _adaptive_configuration_id(
        FEATURE_SET_CONFIDENCE, PRIMARY_ALPHA, PRIMARY_GAMMA, PRIMARY_TEMPERATURE, PRIMARY_SHRINKAGE
    )
    primary_full_id = _adaptive_configuration_id(
        FEATURE_SET_FULL, PRIMARY_ALPHA, PRIMARY_GAMMA, PRIMARY_TEMPERATURE, PRIMARY_SHRINKAGE
    )
    adaptive_ids = np.asarray(predictions["adaptive_configuration_ids"]).astype(str)
    adaptive_predictions = np.asarray(predictions["adaptive_predictions"])
    adaptive_weights = np.asarray(predictions["adaptive_weights"])
    confidence_index = _find_unique_index(adaptive_ids, primary_confidence_id, name="adaptive configuration IDs")
    full_index = _find_unique_index(adaptive_ids, primary_full_id, name="adaptive configuration IDs")
    confidence_predictions = adaptive_predictions[confidence_index].astype(np.int64)
    full_predictions = adaptive_predictions[full_index].astype(np.int64)
    confidence_weights = _validate_weights(adaptive_weights[confidence_index], num_samples=dataset.num_samples)
    full_weights = _validate_weights(adaptive_weights[full_index], num_samples=dataset.num_samples)
    _require(
        np.array_equal(
            reproduce_predictions_from_saved_weights(dataset.logits, confidence_weights),
            confidence_predictions,
        ),
        "saved confidence-only primary predictions cannot be reproduced from saved weights and logits",
    )
    _require(
        np.array_equal(
            reproduce_predictions_from_saved_weights(dataset.logits, full_weights),
            full_predictions,
        ),
        "saved full-feature primary predictions cannot be reproduced from saved weights and logits",
    )

    global_ids = np.asarray(predictions["global_configuration_ids"]).astype(str)
    global_predictions_array = np.asarray(predictions["global_predictions"])
    global_weights_array = np.asarray(predictions["global_weights"])
    primary_global_id = "global_gamma1_temperature2_lambda0p75"
    global_index = _find_unique_index(global_ids, primary_global_id, name="global configuration IDs")
    global_predictions = global_predictions_array[global_index].astype(np.int64)
    global_weights = _validate_weights(global_weights_array[global_index], num_samples=dataset.num_samples)
    _require(
        np.array_equal(
            reproduce_predictions_from_saved_weights(dataset.logits, global_weights),
            global_predictions,
        ),
        "saved global-control predictions cannot be reproduced from saved weights and logits",
    )

    baseline_ids = np.asarray(predictions["baseline_ids"]).astype(str)
    baseline_predictions = np.asarray(predictions["baseline_predictions"])
    classification_methods: dict[str, Any] = {}
    for baseline_id in baseline_ids:
        index = _find_unique_index(baseline_ids, baseline_id, name="baseline IDs")
        baseline_prediction = baseline_predictions[index].astype(np.int64)
        classification_methods[str(baseline_id)] = _metric_with_tail_details(
            dataset.labels, baseline_prediction, class_counts
        )
    classification_methods["global_control"] = _metric_with_tail_details(
        dataset.labels, global_predictions, class_counts
    )
    classification_methods[FEATURE_SET_CONFIDENCE] = _metric_with_tail_details(
        dataset.labels, confidence_predictions, class_counts
    )
    classification_methods[FEATURE_SET_FULL] = _metric_with_tail_details(
        dataset.labels, full_predictions, class_counts
    )
    primary_prediction_changes = prediction_change_report(
        confidence_predictions,
        full_predictions,
        dataset.labels,
        dataset.sample_indices,
        class_counts,
    )
    primary_weight_statistics = {
        FEATURE_SET_CONFIDENCE: weight_group_statistics(confidence_weights, dataset.labels, class_counts),
        FEATURE_SET_FULL: weight_group_statistics(full_weights, dataset.labels, class_counts),
        "global_control": weight_group_statistics(global_weights, dataset.labels, class_counts),
    }
    primary_tail_ranking = {
        FEATURE_SET_CONFIDENCE: ranking_reports[primary_confidence_key],
        FEATURE_SET_FULL: ranking_reports[primary_full_key],
    }

    target_group_labels = _canonical_group_labels(dataset.labels, class_counts)
    primary_target_weight_conditioning: dict[str, Any] = {}
    for rebalanced in REBALANCED_EXPERTS:
        rebalanced_index = EXPERT_ORDER.index(rebalanced)
        mixup_index = EXPERT_ORDER.index(MIXUP)
        condition = (target_group_labels == "tail") & (targets[:, rebalanced_index] > targets[:, mixup_index])
        condition_name = f"{rebalanced}_actual_target_higher_than_Mixup"
        primary_target_weight_conditioning[condition_name] = {
            "sample_count": int(condition.sum()),
            "confidence_only_mean_weight": {
                expert: float(confidence_weights[condition, index].mean())
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "full_13_mean_weight": {
                expert: float(full_weights[condition, index].mean())
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "full_minus_confidence_mixup_weight": float(
                full_weights[condition, mixup_index].mean()
                - confidence_weights[condition, mixup_index].mean()
            ),
        }

    training_diagnostics = _training_vs_held_out_diagnostics(
        dataset=dataset,
        class_counts=class_counts,
        targets=targets,
        scores=scores,
        fold_assignments=fold_assignments,
        matched_models=matched_models,
    )
    fold_tail_accuracy = _fold_tail_accuracy_comparisons(
        adaptive_rows=ridge_results["adaptive_results"],
        predictions=predictions,
        class_counts=class_counts,
        labels=dataset.labels,
        inner_fold_ids=dataset.inner_fold_ids,
    )

    matched_configuration_reports = []
    for alpha in ALPHAS:
        for gamma in GAMMAS:
            confidence_key = (FEATURE_SET_CONFIDENCE, float(alpha), float(gamma))
            full_key = (FEATURE_SET_FULL, float(alpha), float(gamma))
            matched_configuration_reports.append(
                {
                    "alpha": float(alpha),
                    "gamma": float(gamma),
                    "confidence_only": {
                        "model_keys_by_fold": {
                            str(fold): str(matched_models[confidence_key][fold]["model_key"])
                            for fold in PERMITTED_ANALYSIS_INNER_FOLDS
                        },
                        "contribution_prediction": contribution_reports[confidence_key],
                        "pairwise_ranking": ranking_reports[confidence_key],
                    },
                    "full_13": {
                        "model_keys_by_fold": {
                            str(fold): str(matched_models[full_key][fold]["model_key"])
                            for fold in PERMITTED_ANALYSIS_INNER_FOLDS
                        },
                        "contribution_prediction": contribution_reports[full_key],
                        "pairwise_ranking": ranking_reports[full_key],
                    },
                    "prediction_error_difference_full_minus_confidence": _error_difference(
                        contribution_reports[confidence_key], contribution_reports[full_key]
                    ),
                }
            )

    pooled_deltas = [
        row["prediction_error_difference_full_minus_confidence"]["pooled"]["mean_expert_mse_full_minus_confidence"]
        for row in matched_configuration_reports
    ]
    tail_deltas = [
        row["prediction_error_difference_full_minus_confidence"]["tail"]["mean_expert_mse_full_minus_confidence"]
        for row in matched_configuration_reports
    ]
    matched_fit_count = len(ALPHAS) * len(GAMMAS) * len(PERMITTED_ANALYSIS_INNER_FOLDS)
    full_lower_training = 0
    full_lower_heldout = 0
    for alpha in ALPHAS:
        for gamma in GAMMAS:
            for fold in PERMITTED_ANALYSIS_INNER_FOLDS:
                conf_row = next(row for row in training_diagnostics if row["feature_set"] == FEATURE_SET_CONFIDENCE and row["alpha"] == float(alpha) and row["gamma"] == float(gamma) and row["validation_inner_fold"] == fold)
                full_row = next(row for row in training_diagnostics if row["feature_set"] == FEATURE_SET_FULL and row["alpha"] == float(alpha) and row["gamma"] == float(gamma) and row["validation_inner_fold"] == fold)
                full_lower_training += int(full_row["saved_training_mse"] < conf_row["saved_training_mse"])
                full_lower_heldout += int(full_row["saved_held_out_mse"] < conf_row["saved_held_out_mse"])
    grid_summary = {
        "full_better_balanced_accuracy_count": int(sum(row["differences_full_minus_confidence"]["balanced_accuracy"] > 0.0 for row in grid)),
        "full_better_tail_accuracy_count": int(sum(row["differences_full_minus_confidence"]["tail_accuracy"] > 0.0 for row in grid)),
        "full_worse_balanced_accuracy_count": int(sum(row["differences_full_minus_confidence"]["balanced_accuracy"] < 0.0 for row in grid)),
        "full_worse_tail_accuracy_count": int(sum(row["differences_full_minus_confidence"]["tail_accuracy"] < 0.0 for row in grid)),
        "identical_prediction_count": int(sum(row["differences_full_minus_confidence"]["prediction_identical"] for row in grid)),
        "mean_tail_accuracy_delta": float(np.mean([row["differences_full_minus_confidence"]["tail_accuracy"] for row in grid])),
        "mean_mixup_weight_delta": float(np.mean([row["differences_full_minus_confidence"]["mean_mixup_weight"] for row in grid])),
        "mean_weights_mean_absolute_difference": float(np.mean([row["differences_full_minus_confidence"]["weights_mean_absolute_difference"] for row in grid])),
        "mean_prediction_changed_count": float(np.mean([row["differences_full_minus_confidence"]["prediction_changed_count"] for row in grid])),
    }
    training_summary = {
        "matched_model_fit_count": matched_fit_count,
        "full_lower_training_mse_count": int(full_lower_training),
        "full_lower_held_out_mse_count": int(full_lower_heldout),
        "tail_held_out_mean_expert_mse_by_feature_set": {
            feature_set: float(np.mean([row["tail_held_out_mean_expert_mse"] for row in training_diagnostics if row["feature_set"] == feature_set]))
            for feature_set in FEATURE_SETS
        },
    }

    source_hashes_after = _source_file_records(source_paths)
    _verify_source_records(source_hashes_before)
    _require(source_hashes_after == source_hashes_before, "a source artifact changed during Task 3F-F")

    results: dict[str, Any] = {
        "task_identifier": TASK3FF_TASK_IDENTIFIER,
        "schema_version": TASK3FF_SCHEMA_VERSION,
        "source_git_commit": _git_commit(project_root_path),
        "population": {
            "sample_count": dataset.num_samples,
            "true_group_counts": {
                group_name: int(np.sum(target_group_labels == group_name))
                for group_name in GROUP_NAMES
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
            "saved_task3f_a_scores_used": True,
            "saved_task3f_a_predictions_and_weights_used": True,
            "source_hashes_verified": True,
            "source_artifacts": source_hashes_before,
        },
        "definitions": {
            "class_groups_source": "scripts.base_trainer.compute_class_groups",
            "supervised_target_source": "scripts.task3f_ridge.compute_contribution_targets",
            "supervised_target_formula": "(z_e_y-zbar_y)-sum_c softmax(zbar)_c*(z_e_c-zbar_c)",
            "ranking_rule": "strict pairwise score order; actual ties excluded from ranking denominator",
            "weight_source": "saved Task 3F-A adaptive_weights; rows renormalized only for float32 storage roundoff",
            "classification_source": "saved Task 3F-A predictions reproduced from saved weights and original OOF logits",
            "feature_sets": {
                feature_set: {"dimension": len(FEATURE_NAMES[feature_set]), "names": list(FEATURE_NAMES[feature_set])}
                for feature_set in FEATURE_SETS
            },
        },
        "matched_alpha_gamma_configurations": matched_configuration_reports,
        "contribution_prediction": {
        "actual_target_means": {
            group_name: {
                    expert: float(
                        targets[
                            np.ones(dataset.num_samples, dtype=bool)
                            if group_name == "pooled"
                            else target_group_labels == group_name,
                            index,
                        ].mean()
                    )
                    for index, expert in enumerate(EXPERT_ORDER)
                }
                for group_name in ("pooled", *GROUP_NAMES)
                if np.any(target_group_labels == group_name) or group_name == "pooled"
            },
        },
        "primary_comparison": {
            "configuration": {
                "alpha": PRIMARY_ALPHA,
                "gamma": PRIMARY_GAMMA,
                "temperature": PRIMARY_TEMPERATURE,
                "shrinkage": PRIMARY_SHRINKAGE,
                "confidence_only_id": primary_confidence_id,
                "full_13_id": primary_full_id,
                "global_control_id": primary_global_id,
            },
            "contribution_prediction": {
                FEATURE_SET_CONFIDENCE: contribution_reports[primary_confidence_key],
                FEATURE_SET_FULL: contribution_reports[primary_full_key],
                "error_difference_full_minus_confidence": _error_difference(
                    contribution_reports[primary_confidence_key], contribution_reports[primary_full_key]
                ),
            },
            "pairwise_ranking": primary_tail_ranking,
            "classification": classification_methods,
            "prediction_changes": primary_prediction_changes,
            "weight_statistics": primary_weight_statistics,
            "tail_target_weight_conditioning": primary_target_weight_conditioning,
        },
        "configuration_grid": {
            "count": len(grid),
            "expected_count": len(ALPHAS) * len(GAMMAS) * len(TEMPERATURES) * len(SHRINKAGES),
            "rows": grid,
        },
        "training_vs_held_out": {
            "model_fit_diagnostics": training_diagnostics,
            "tail_classification_accuracy_by_fold": fold_tail_accuracy,
        },
        "overall_patterns": {
            "contribution_error": {
                "matched_alpha_gamma_count": len(matched_configuration_reports),
                "pooled_full_lower_mse_count": int(sum(delta < 0.0 for delta in pooled_deltas)),
                "tail_full_lower_mse_count": int(sum(delta < 0.0 for delta in tail_deltas)),
                "pooled_mean_mse_delta": float(np.mean(pooled_deltas)),
                "tail_mean_mse_delta": float(np.mean(tail_deltas)),
                "pooled_mse_deltas": pooled_deltas,
                "tail_mse_deltas": tail_deltas,
            },
            "classification_grid": grid_summary,
            "training_vs_held_out": training_summary,
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
    "ALPHAS",
    "FEATURE_SET_CONFIDENCE",
    "FEATURE_SET_FULL",
    "GAMMAS",
    "PERMITTED_ANALYSIS_INNER_FOLDS",
    "PRIMARY_ALPHA",
    "PRIMARY_GAMMA",
    "PRIMARY_SHRINKAGE",
    "PRIMARY_TEMPERATURE",
    "Task3FFeatureComparisonError",
    "contribution_prediction_report",
    "identify_matched_model_fits",
    "pairwise_contribution_ranking",
    "pearson_correlation",
    "prediction_change_report",
    "reproduce_predictions_from_saved_weights",
    "run_task3f_feature_comparison",
    "sha256_file",
    "validate_artifact_alignment",
    "validate_saved_task3f_a_metadata",
    "weight_group_statistics",
]
