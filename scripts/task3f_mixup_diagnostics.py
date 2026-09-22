"""Task 3F-B: read-only diagnostics for Ridge's Mixup preference.

This module consumes the frozen Task 3F-A arrays and the already validated
Task 3C OOF logits restricted to inner folds 1--3.  It does not fit a model,
change a saved prediction, access a reserved fold, or read the CIFAR-100 test
set.  Labels appear only in retrospective target, group, and metric reports.
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
    FEATURE_NAMES,
    FEATURE_SET_CONFIDENCE,
    HISTORICAL_NO_CE_WEIGHTS,
    PERMITTED_ANALYSIS_INNER_FOLDS,
    FIXED_REFERENCE_UNITS,
    classification_metrics,
    combine_weighted_logits,
    compute_contribution_targets,
    extract_features,
)


TASK3FB_SCHEMA_VERSION = "task3f_mixup_diagnostics.v1"
TASK3FB_TASK_IDENTIFIER = "Task 3F-B"
TASK3FA_TASK_IDENTIFIER = "Task 3F-A"
DATASET_NAME = "CIFAR-100-LT"
ANALYSIS_SAMPLE_COUNT = 6_507
NUM_CLASSES = 100
OUTER_FOLD = 0
HIGHLIGHTED_CONFIGURATION_ID = (
    "adaptive_confidence_only_alpha1000_gamma1_temperature2_lambda0p75"
)
GLOBAL_CONFIGURATION_ID = "global_gamma1_temperature2_lambda0p75"
MIXUP_INDEX = 3
MIXUP_SENSITIVITY_FACTORS = (1.0, 0.75, 0.50, 0.25, 0.0)
WEIGHT_COMPARISON_TOLERANCE = 1e-6
SCORE_RECONSTRUCTION_ATOL = 2e-5


class Task3FBDiagnosticError(OOFProtocolError):
    """Raised when a Task 3F-B input violates the frozen diagnostic contract."""


def _as_numeric_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise Task3FBDiagnosticError(f"{name} must contain real numeric values")
    if not np.isfinite(array).all():
        raise Task3FBDiagnosticError(f"{name} contains non-finite values")
    return array.astype(np.float64, copy=False)


def _as_integer_vector(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise Task3FBDiagnosticError(f"{name} must be one-dimensional")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise Task3FBDiagnosticError(f"{name} must contain integer values")
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        raise Task3FBDiagnosticError(f"{name} must contain finite integer values")
    return array.astype(np.int64, copy=False)


def _validate_matrix(
    value: Any,
    *,
    name: str,
    columns: int,
    rows: int | None = None,
) -> np.ndarray:
    array = _as_numeric_array(value, name=name)
    if array.ndim != 2 or array.shape[1] != columns or (
        rows is not None and array.shape[0] != rows
    ):
        raise Task3FBDiagnosticError(f"{name} must have shape (samples, {columns})")
    return array


def _validate_class_counts(class_counts: Any, *, num_classes: int) -> np.ndarray:
    counts = _as_integer_vector(class_counts, name="class_counts")
    if len(counts) != num_classes or np.any(counts < 1):
        raise Task3FBDiagnosticError(
            f"class_counts must contain {num_classes} positive class counts"
        )
    return counts


def _validate_weights(weights: Any, *, num_samples: int) -> np.ndarray:
    array = _as_numeric_array(weights, name="routing weights")
    if array.ndim == 1:
        if array.shape != (len(EXPERT_ORDER),):
            raise Task3FBDiagnosticError("routing weights must have four expert columns")
        array = np.repeat(array[None, :], num_samples, axis=0)
    if array.shape != (num_samples, len(EXPERT_ORDER)):
        raise Task3FBDiagnosticError(
            f"routing weights must have shape ({num_samples}, {len(EXPERT_ORDER)})"
        )
    if np.any(array < 0.0) or not np.allclose(
        array.sum(axis=1), 1.0, rtol=0.0, atol=2e-6
    ):
        raise Task3FBDiagnosticError(
            "routing weights must be non-negative and sum to one"
        )
    return array


def _weights_for_existing_combiner(weights: Any, *, num_samples: int) -> np.ndarray:
    """Restore exact row sums lost when Task 3F-A wrote float32 arrays."""
    array = _validate_weights(weights, num_samples=num_samples)
    return array / array.sum(axis=1, keepdims=True)


def validate_artifact_alignment(
    *,
    expected_sample_indices: Any,
    expected_inner_fold_ids: Any,
    expected_labels: Any,
    artifact_sample_indices: Any,
    artifact_inner_fold_ids: Any,
    artifact_labels: Any,
) -> None:
    """Require exact row-wise alignment and reject reserved fold 0.

    Exact order is intentional: these arrays index predictions and weights, so
    a matching set of IDs in a different order is not a valid alignment.
    """
    expected_ids = _as_integer_vector(
        expected_sample_indices, name="expected sample IDs"
    )
    expected_folds = _as_integer_vector(
        expected_inner_fold_ids, name="expected inner-fold IDs"
    )
    expected_y = _as_integer_vector(expected_labels, name="expected labels")
    artifact_ids = _as_integer_vector(
        artifact_sample_indices, name="artifact sample IDs"
    )
    artifact_folds = _as_integer_vector(
        artifact_inner_fold_ids, name="artifact inner-fold IDs"
    )
    artifact_y = _as_integer_vector(artifact_labels, name="artifact labels")
    expected_length = len(expected_ids)
    if any(
        len(array) != expected_length
        for array in (expected_folds, expected_y, artifact_ids, artifact_folds, artifact_y)
    ):
        raise Task3FBDiagnosticError("artifact arrays have inconsistent sample counts")
    if len(np.unique(expected_ids)) != expected_length:
        raise Task3FBDiagnosticError("expected sample IDs contain duplicates")
    if len(np.unique(artifact_ids)) != expected_length:
        raise Task3FBDiagnosticError("artifact sample IDs contain duplicates")
    if np.any(np.isin(artifact_folds, (0,))):
        raise Task3FBDiagnosticError(
            "artifact contains the reserved inner-fold-0 population"
        )
    if not np.all(np.isin(artifact_folds, PERMITTED_ANALYSIS_INNER_FOLDS)):
        raise Task3FBDiagnosticError("artifact contains an incompatible inner fold")
    if not np.array_equal(expected_ids, artifact_ids):
        raise Task3FBDiagnosticError("sample IDs are not exactly aligned")
    if not np.array_equal(expected_folds, artifact_folds):
        raise Task3FBDiagnosticError("inner-fold IDs are not exactly aligned")
    if not np.array_equal(expected_y, artifact_y):
        raise Task3FBDiagnosticError("labels are not exactly aligned")


def compute_group_statistics(
    labels: Any,
    weights: Any,
    predicted_scores: Any,
    actual_targets: Any,
    class_counts: Any,
) -> dict[str, dict[str, Any]]:
    """Summarize saved weights and retrospective scores by canonical group."""
    labels_array = _as_integer_vector(labels, name="labels")
    weights_array = _validate_weights(weights, num_samples=len(labels_array))
    predicted_array = _validate_matrix(
        predicted_scores,
        name="predicted contribution scores",
        columns=len(EXPERT_ORDER),
        rows=len(labels_array),
    )
    actual_array = _validate_matrix(
        actual_targets,
        name="actual contribution targets",
        columns=len(EXPERT_ORDER),
        rows=len(labels_array),
    )
    counts = _validate_class_counts(class_counts, num_classes=len(class_counts))
    groups = compute_class_groups(counts)
    highest_expert = weights_array.argmax(axis=1)
    report: dict[str, dict[str, Any]] = {}
    for group_name in ("head", "medium", "tail"):
        classes = np.asarray(groups[group_name], dtype=np.int64)
        mask = np.isin(labels_array, classes)
        if not np.any(mask):
            raise Task3FBDiagnosticError(
                f"permitted rows contain no samples in the canonical {group_name} group"
            )
        group_weights = weights_array[mask]
        group_predicted = predicted_array[mask]
        group_actual = actual_array[mask]
        report[group_name] = {
            "class_ids": [int(value) for value in classes.tolist()],
            "sample_count": int(mask.sum()),
            "class_sample_count": {
                str(int(class_id)): int(np.sum(labels_array == class_id))
                for class_id in classes
            },
            "mean_weight": {
                expert: float(group_weights[:, index].mean())
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "std_weight": {
                expert: float(group_weights[:, index].std())
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "minimum_weight": {
                expert: float(group_weights[:, index].min())
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "maximum_weight": {
                expert: float(group_weights[:, index].max())
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "highest_weight_fraction": {
                expert: float(np.mean(highest_expert[mask] == index))
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "highest_weight_percentage": {
                expert: float(100.0 * np.mean(highest_expert[mask] == index))
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "mean_predicted_score": {
                expert: float(group_predicted[:, index].mean())
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "mean_actual_target": {
                expert: float(group_actual[:, index].mean())
                for index, expert in enumerate(EXPERT_ORDER)
            },
        }
    return report


def decompose_ridge_scores(
    features: Any,
    intercept: Any,
    coefficients: Any,
    scaler_mean: Any,
    scaler_scale: Any,
) -> dict[str, np.ndarray]:
    """Reconstruct ``intercept + coefficient * standardized feature`` scores."""
    feature_array = _as_numeric_array(features, name="Ridge features")
    if feature_array.ndim != 2:
        raise Task3FBDiagnosticError("Ridge features must be two-dimensional")
    dimension = feature_array.shape[1]
    intercept_array = _as_numeric_array(intercept, name="Ridge intercept")
    coefficients_array = _as_numeric_array(coefficients, name="Ridge coefficients")
    mean_array = _as_numeric_array(scaler_mean, name="feature scaler mean")
    scale_array = _as_numeric_array(scaler_scale, name="feature scaler scale")
    if intercept_array.shape != (len(EXPERT_ORDER),):
        raise Task3FBDiagnosticError("Ridge intercept must contain four expert values")
    if coefficients_array.shape != (len(EXPERT_ORDER), dimension):
        raise Task3FBDiagnosticError(
            "Ridge coefficients must have shape (4, feature_dimension)"
        )
    if mean_array.shape != (dimension,) or scale_array.shape != (dimension,):
        raise Task3FBDiagnosticError("feature scaler parameters do not match features")
    if np.any(scale_array <= 0.0):
        raise Task3FBDiagnosticError("feature scaler scales must be positive")
    standardized = (feature_array - mean_array[None, :]) / scale_array[None, :]
    feature_contributions = standardized[:, None, :] * coefficients_array[None, :, :]
    feature_term = feature_contributions.sum(axis=2)
    predicted_scores = intercept_array[None, :] + feature_term
    return {
        "standardized_features": standardized,
        "feature_contributions": feature_contributions,
        "feature_term": feature_term,
        "intercept": intercept_array,
        "predicted_scores": predicted_scores,
    }


def _validate_prediction_vector(value: Any, *, name: str, num_samples: int) -> np.ndarray:
    array = _as_integer_vector(value, name=name)
    if len(array) != num_samples:
        raise Task3FBDiagnosticError(f"{name} is not aligned with the sample IDs")
    return array


def tail_gain_loss_accounting(
    sample_ids: Any,
    labels: Any,
    uniform_predictions: Any,
    ridge_predictions: Any,
    expert_predictions: Any,
    ridge_weights: Any,
    class_counts: Any,
) -> dict[str, Any]:
    """Account for Tail changes and retain every gained/lost image record."""
    ids = _as_integer_vector(sample_ids, name="sample IDs")
    labels_array = _as_integer_vector(labels, name="labels")
    if len(np.unique(ids)) != len(ids):
        raise Task3FBDiagnosticError("sample IDs contain duplicates")
    uniform = _validate_prediction_vector(
        uniform_predictions, name="uniform predictions", num_samples=len(ids)
    )
    ridge = _validate_prediction_vector(
        ridge_predictions, name="Ridge predictions", num_samples=len(ids)
    )
    expert_array = _validate_matrix(
        expert_predictions,
        name="expert predictions",
        columns=len(EXPERT_ORDER),
        rows=len(ids),
    ).astype(np.int64)
    weights = _validate_weights(ridge_weights, num_samples=len(ids))
    raw_counts = np.asarray(class_counts)
    if raw_counts.ndim != 1:
        raise Task3FBDiagnosticError("class_counts must be one-dimensional")
    counts = _validate_class_counts(class_counts, num_classes=len(raw_counts))
    if np.any(labels_array < 0) or np.any(labels_array >= len(counts)):
        raise Task3FBDiagnosticError("labels fall outside the supplied class counts")
    groups = compute_class_groups(counts)
    tail_mask = np.isin(labels_array, groups["tail"])
    uniform_correct = uniform == labels_array
    ridge_correct = ridge == labels_array
    gained = tail_mask & ~uniform_correct & ridge_correct
    lost = tail_mask & uniform_correct & ~ridge_correct
    both_correct = tail_mask & uniform_correct & ridge_correct
    both_wrong = tail_mask & ~uniform_correct & ~ridge_correct

    def records(mask: np.ndarray) -> list[dict[str, Any]]:
        return [
            {
                "sample_id": int(ids[index]),
                "true_class": int(labels_array[index]),
                "uniform_prediction": int(uniform[index]),
                "ridge_prediction": int(ridge[index]),
                "expert_predictions": [
                    int(value) for value in expert_array[index].tolist()
                ],
                "ridge_weights": [float(value) for value in weights[index].tolist()],
            }
            for index in np.flatnonzero(mask)
        ]

    by_class: dict[str, dict[str, int]] = {}
    for class_id in groups["tail"]:
        class_mask = labels_array == class_id
        uniform_count = int(np.sum(class_mask & uniform_correct))
        ridge_count = int(np.sum(class_mask & ridge_correct))
        gained_count = int(np.sum(class_mask & gained))
        lost_count = int(np.sum(class_mask & lost))
        by_class[str(int(class_id))] = {
            "sample_count": int(class_mask.sum()),
            "uniform_correct_count": uniform_count,
            "ridge_correct_count": ridge_count,
            "gained_count": gained_count,
            "lost_count": lost_count,
            "net_correct_change": ridge_count - uniform_count,
        }
    return {
        "tail_sample_count": int(tail_mask.sum()),
        "gained_count": int(gained.sum()),
        "lost_count": int(lost.sum()),
        "both_correct_count": int(both_correct.sum()),
        "both_wrong_count": int(both_wrong.sum()),
        "by_true_class": by_class,
        "gained_cases": records(gained),
        "lost_cases": records(lost),
    }


def renormalize_mixup_weights(weights: Any, factor: float) -> np.ndarray:
    """Scale the saved Mixup column and renormalize without using labels."""
    if isinstance(factor, bool) or not np.isfinite(float(factor)) or float(factor) < 0.0:
        raise Task3FBDiagnosticError("Mixup scaling factor must be finite and non-negative")
    array = _as_numeric_array(weights, name="routing weights")
    was_vector = array.ndim == 1
    if was_vector:
        if array.shape != (len(EXPERT_ORDER),):
            raise Task3FBDiagnosticError("routing weights must contain four experts")
        matrix = array[None, :]
    elif array.ndim == 2 and array.shape[1] == len(EXPERT_ORDER):
        matrix = array
    else:
        raise Task3FBDiagnosticError("routing weights must have shape (samples, 4)")
    if np.any(matrix < 0.0) or not np.allclose(
        matrix.sum(axis=1), 1.0, rtol=0.0, atol=2e-6
    ):
        raise Task3FBDiagnosticError("routing weights must be normalized and non-negative")
    if float(factor) == 1.0:
        return array.copy()
    adjusted = matrix.copy()
    adjusted[:, MIXUP_INDEX] *= float(factor)
    denominators = adjusted.sum(axis=1)
    if np.any(denominators <= 0.0):
        raise Task3FBDiagnosticError("Mixup scaling produced a zero weight row")
    adjusted /= denominators[:, None]
    return adjusted[0] if was_vector else adjusted


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
        raise Task3FBDiagnosticError(f"cannot load {name}: {path}") from exc
    if not isinstance(payload, dict):
        raise Task3FBDiagnosticError(f"{name} must be a JSON object")
    return payload


def _load_npz(path: Path, *, name: str) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: np.array(archive[key]) for key in archive.files}
    except (OSError, ValueError) as exc:
        raise Task3FBDiagnosticError(f"cannot load {name}: {path}") from exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Task3FBDiagnosticError(message)


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


def _metric_report(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_counts: np.ndarray,
) -> dict[str, Any]:
    metrics = classification_metrics(labels, predictions, class_counts)
    return {
        "ordinary_accuracy": float(metrics["ordinary_accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
        "head_accuracy": float(metrics["head_accuracy"]),
        "medium_accuracy": float(metrics["medium_accuracy"]),
        "tail_accuracy": float(metrics["tail_accuracy"]),
    }


def _find_unique_string_index(values: np.ndarray, target: str, *, name: str) -> int:
    string_values = np.asarray(values).astype(str)
    matches = np.flatnonzero(string_values == target)
    if len(matches) != 1:
        raise Task3FBDiagnosticError(
            f"{name} must contain exactly one {target!r}, found {len(matches)}"
        )
    return int(matches[0])


def _compose_highlighted_scores(
    *,
    model_ids: np.ndarray,
    held_out_scores: np.ndarray,
    dataset: RestrictedAnalysisDataset,
    model_keys_by_fold: Mapping[int, str],
) -> tuple[np.ndarray, dict[int, int]]:
    if held_out_scores.ndim != 3 or held_out_scores.shape[2] != len(EXPERT_ORDER):
        raise Task3FBDiagnosticError("held-out score tensor has an incompatible shape")
    composed = np.full(
        (dataset.num_samples, len(EXPERT_ORDER)), np.nan, dtype=np.float64
    )
    selected_indices: dict[int, int] = {}
    for fold, model_key in model_keys_by_fold.items():
        model_index = _find_unique_string_index(model_ids, model_key, name="model IDs")
        selected_indices[int(fold)] = model_index
        validation_mask = dataset.inner_fold_ids == int(fold)
        row = np.asarray(held_out_scores[model_index], dtype=np.float64)
        finite = np.isfinite(row).all(axis=1)
        if not np.array_equal(finite, validation_mask):
            raise Task3FBDiagnosticError(
                f"saved scores for {model_key} are not held out on exactly fold {fold}"
            )
        composed[validation_mask] = row[validation_mask]
    if not np.isfinite(composed).all():
        raise Task3FBDiagnosticError("highlighted held-out scores are incomplete")
    return composed, selected_indices


def _validate_task3f_inputs(
    *,
    ridge_directory: Path,
    dataset: RestrictedAnalysisDataset,
    experiment_config: Mapping[str, Any],
    ridge_results: Mapping[str, Any],
    predictions: Mapping[str, np.ndarray],
    scores: Mapping[str, np.ndarray],
    fold_assignments: Mapping[str, Any],
) -> None:
    _require(
        experiment_config.get("task_identifier") == TASK3FA_TASK_IDENTIFIER,
        "Task 3F-A experiment_config has the wrong task identifier",
    )
    _require(
        experiment_config.get("expert_order") == list(EXPERT_ORDER),
        "Task 3F-A expert order is incompatible",
    )
    _require(
        experiment_config.get("analyzed_sample_count") == dataset.num_samples == ANALYSIS_SAMPLE_COUNT,
        "Task 3F-A analyzed population is not the permitted 6,507 rows",
    )
    _require(
        experiment_config.get("permitted_analysis_inner_folds")
        == list(PERMITTED_ANALYSIS_INNER_FOLDS),
        "Task 3F-A permitted folds are incompatible",
    )
    _require(
        experiment_config.get("reserved_router_selection_inner_folds") == [0],
        "Task 3F-A reserved fold declaration is incompatible",
    )
    restrictions = experiment_config.get("data_restrictions", {})
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
        and fold_assignments.get("permitted_inner_fold_ids")
        == list(PERMITTED_ANALYSIS_INNER_FOLDS)
        and fold_assignments.get("every_sample_held_out_exactly_once") is True,
        "Task 3F-A fold assignment provenance is incompatible",
    )
    for fold in PERMITTED_ANALYSIS_INNER_FOLDS:
        record = fold_assignments.get("folds", {}).get(str(fold))
        _require(isinstance(record, Mapping), f"missing Task 3F-A fold {fold} assignment")
        validation_ids = np.asarray(record.get("validation_sample_ids", []), dtype=np.int64)
        training_ids = np.asarray(record.get("training_sample_ids", []), dtype=np.int64)
        expected_validation = dataset.sample_indices[dataset.inner_fold_ids == fold]
        expected_training = dataset.sample_indices[dataset.inner_fold_ids != fold]
        _require(
            np.array_equal(validation_ids, expected_validation)
            and np.array_equal(training_ids, expected_training),
            f"Task 3F-A fold {fold} assignment does not match the validated dataset",
        )

    sample_ids = predictions.get("sample_indices")
    folds = predictions.get("inner_fold_ids")
    labels = predictions.get("labels")
    _require(sample_ids is not None and folds is not None and labels is not None, "prediction alignment arrays are missing")
    validate_artifact_alignment(
        expected_sample_indices=dataset.sample_indices,
        expected_inner_fold_ids=dataset.inner_fold_ids,
        expected_labels=dataset.labels,
        artifact_sample_indices=sample_ids,
        artifact_inner_fold_ids=folds,
        artifact_labels=labels,
    )
    validate_artifact_alignment(
        expected_sample_indices=dataset.sample_indices,
        expected_inner_fold_ids=dataset.inner_fold_ids,
        expected_labels=dataset.labels,
        artifact_sample_indices=scores.get("sample_indices"),
        artifact_inner_fold_ids=scores.get("inner_fold_ids"),
        artifact_labels=scores.get("labels"),
    )
    _require(
        np.array_equal(predictions["sample_indices"], scores["sample_indices"]),
        "Task 3F-A prediction and score sample IDs disagree",
    )
    _require(
        "model_fits" in ridge_results and "baseline_results" in ridge_results,
        "Task 3F-A result records are incomplete",
    )
    _require(
        ridge_directory.exists(), "Task 3F-A directory does not exist"
    )


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise Task3FBDiagnosticError(f"refusing to overwrite incompatible output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _write_text_once(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text() != text:
            raise Task3FBDiagnosticError(f"refusing to overwrite incompatible output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _round_metric_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    return {
        key: float(left[key] - right[key])
        for key in (
            "ordinary_accuracy",
            "balanced_accuracy",
            "head_accuracy",
            "medium_accuracy",
            "tail_accuracy",
        )
    }


def _render_summary(results: Mapping[str, Any]) -> str:
    diagnostic_a = results["diagnostic_a"]
    diagnostic_b = results["diagnostic_b"]
    diagnostic_c = results["diagnostic_c"]
    diagnostic_d = results["diagnostic_d"]
    folds = diagnostic_a["folds"]
    mixup_intercepts = [float(row["intercept"]["Mixup"]) for row in folds]
    mixup_feature_terms = [float(row["held_out_mean_feature_term"]["Mixup"]) for row in folds]
    mixup_absolute_feature_terms = [
        float(row["held_out_mean_absolute_feature_term"]["Mixup"]) for row in folds
    ]
    mixup_scores = [float(row["mean_held_out_predicted_score"]["Mixup"]) for row in folds]
    mixup_targets = [float(row["mean_held_out_actual_target"]["Mixup"]) for row in folds]
    target_leaders = [
        max(
            row["mean_held_out_actual_target"],
            key=row["mean_held_out_actual_target"].get,
        )
        for row in folds
    ]
    predicted_leaders = [
        max(
            row["mean_held_out_predicted_score"],
            key=row["mean_held_out_predicted_score"].get,
        )
        for row in folds
    ]
    intercept_advantages = [
        row["intercept"]["Mixup"]
        - np.mean(
            [value for expert, value in row["intercept"].items() if expert != "Mixup"]
        )
        for row in folds
    ]
    tail = diagnostic_b["groups"]["tail"]
    head = diagnostic_b["groups"]["head"]
    uniform = diagnostic_d["methods"]["uniform_logit"]["metrics"]
    ridge = diagnostic_d["methods"]["ridge_highlighted"]["metrics"]
    sensitivity = diagnostic_d["mixup_sensitivity"]
    improved_tail_factors = [
        row["factor"]
        for row in sensitivity
        if row["metrics"]["tail_accuracy"] > sensitivity[0]["metrics"]["tail_accuracy"]
    ]
    adaptive_global = diagnostic_d["adaptive_vs_global"]
    adaptation_text = (
        "non-trivial image-dependent variation is present"
        if adaptive_global["fraction_rows_with_meaningful_difference"] > 0.5
        else "most rows remain close to the corresponding global control"
    )
    class_changes = diagnostic_c["tail_accounting"]["by_true_class"]
    changed_classes = [
        class_id for class_id, row in class_changes.items() if row["net_correct_change"] != 0
    ]
    changed_class_label = "class" if len(changed_classes) == 1 else "classes"
    lines = [
        "# Task 3F-B — Ridge Mixup-preference diagnostics",
        "",
        "This is a retrospective, development-only diagnosis on the frozen 6,507 "
        "outer-fold-0 / inner-folds-1–3 rows. It does not establish an independent "
        "router result and does not use inner fold 0, the reserved outer evaluation "
        "population, or the CIFAR-100 test set.",
        "",
        "## Executive findings",
        "",
        f"1. **Why Mixup is favored.** The saved confidence-only Ridge has Mixup "
        f"intercepts {', '.join(f'{value:.4f}' for value in mixup_intercepts)} across "
        f"folds 1–3, exceeding the other-expert intercept mean by "
        f"{', '.join(f'{value:.4f}' for value in intercept_advantages)}. Its signed "
        f"mean held-out feature terms are {', '.join(f'{value:.4f}' for value in mixup_feature_terms)} "
        f"(mean absolute terms {', '.join(f'{value:.4f}' for value in mixup_absolute_feature_terms)}), "
        f"so features modulate individual images but largely cancel in the pooled mean. Its mean held-out "
        f"predicted scores are {', '.join(f'{value:.4f}' for value in mixup_scores)}; "
        f"the corresponding actual target means are "
        f"{', '.join(f'{value:.4f}' for value in mixup_targets)}. Mixup is the actual-target "
        f"leader in {sum(value == 'Mixup' for value in target_leaders)}/3 folds and the "
        f"predicted-score leader in {sum(value == 'Mixup' for value in predicted_leaders)}/3. "
        "The full coefficients, "
        "scalers, and per-fold target comparisons are in `diagnostic_results.json`.",
        f"2. **Head versus Tail.** Head has {head['sample_count']} rows and Tail has "
        f"{tail['sample_count']} rows. Mixup's mean weight is "
        f"{head['mean_weight']['Mixup']:.4f} on Head versus "
        f"{tail['mean_weight']['Mixup']:.4f} on Tail; its highest-weight fractions are "
        f"{head['highest_weight_percentage']['Mixup']:.2f}% and "
        f"{tail['highest_weight_percentage']['Mixup']:.2f}%, respectively. Tail mean "
        f"actual Mixup target is {tail['mean_actual_target']['Mixup']:.4f} versus "
        f"predicted {tail['mean_predicted_score']['Mixup']:.4f}.",
        f"3. **Tail changes.** Relative to uniform logit averaging, Ridge gains "
        f"{diagnostic_c['tail_accounting']['gained_count']} Tail rows and loses "
        f"{diagnostic_c['tail_accounting']['lost_count']}; both-correct is "
        f"{diagnostic_c['tail_accounting']['both_correct_count']} and both-wrong is "
        f"{diagnostic_c['tail_accounting']['both_wrong_count']}. "
        f"{len(changed_classes)} Tail {changed_class_label} have a non-zero net correct-count change. "
        f"Head accuracy changes by {ridge['head_accuracy'] - uniform['head_accuracy']:+.4f} "
        f"and Medium accuracy by {ridge['medium_accuracy'] - uniform['medium_accuracy']:+.4f}.",
        f"4. **Mixup sensitivity.** The predefined factors are evaluated without refitting. "
        f"Factors with Tail accuracy above factor 1.0 are {improved_tail_factors or 'none'}; "
        f"factor 1.0 has BA {sensitivity[0]['metrics']['balanced_accuracy']:.4f} and "
        f"Tail {sensitivity[0]['metrics']['tail_accuracy']:.4f}; factor 0.0 changes these "
        f"to BA {sensitivity[-1]['metrics']['balanced_accuracy']:.4f} and Tail "
        f"{sensitivity[-1]['metrics']['tail_accuracy']:.4f}. This table is not used "
        "to select a new factor.",
        f"5. **Image dependence.** Compared with the corresponding global control, "
        f"{adaptation_text}: {adaptive_global['fraction_rows_with_meaningful_difference']:.4f} "
        f"of rows differ beyond {adaptive_global['tolerance']:.1e}, with mean absolute "
        f"weight difference {adaptive_global['mean_absolute_weight_difference']:.4f}. "
        "Thus the result can contain image-dependent variation while still having a "
        "strong global Mixup preference; these observations do not establish causality.",
        "",
        "## Metric comparison",
        "",
        "| Method | BA | Head | Medium | Tail |",
        "|:--|--:|--:|--:|--:|",
    ]
    for method_name, row in diagnostic_d["methods"].items():
        metrics = row["metrics"]
        lines.append(
            f"| {method_name} | {metrics['balanced_accuracy']:.4f} | "
            f"{metrics['head_accuracy']:.4f} | {metrics['medium_accuracy']:.4f} | "
            f"{metrics['tail_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Sensitivity table",
            "",
            "| Mixup factor | BA | Head | Medium | Tail | Same predictions as factor 1 |",
            "|--:|--:|--:|--:|--:|:--:|",
        ]
    )
    for row in sensitivity:
        metrics = row["metrics"]
        lines.append(
            f"| {row['factor']:.2f} | {metrics['balanced_accuracy']:.4f} | "
            f"{metrics['head_accuracy']:.4f} | {metrics['medium_accuracy']:.4f} | "
            f"{metrics['tail_accuracy']:.4f} | "
            f"{'yes' if row['factor'] == 1.0 else '—'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "The contribution targets and class-group reports use labels only after the "
            "saved Ridge weights and predictions were loaded. The sensitivity factors "
            "are fixed before evaluation and do not constitute a new selected router. "
            "A higher Tail score after reducing Mixup would be evidence of a development "
            "trade-off, not proof that Mixup caused the Ridge Tail weakness. All findings "
            "remain exploratory until a frozen independent evaluation is run.",
            "",
        ]
    )
    return "\n".join(lines)


def run_task3f_mixup_diagnostics(
    *,
    data_root: str | Path = "./data",
    oof_directory: str | Path = "artifacts/oof/task3c_oof",
    ridge_directory: str | Path = "artifacts/oof/task3f_ridge",
    output_directory: str | Path = "artifacts/oof/task3f_mixup_diagnostics",
    project_root: str | Path | None = None,
) -> dict[str, Path]:
    """Run Task 3F-B against existing artifacts and persist write-once outputs."""
    project_root_path = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    source_directory = Path(oof_directory).resolve()
    ridge_path = Path(ridge_directory).resolve()
    output_path = Path(output_directory).resolve()

    task3f_paths = {
        name: ridge_path / filename
        for name, filename in {
            "experiment_config": "experiment_config.json",
            "ridge_results": "ridge_results.json",
            "held_out_predictions": "held_out_predictions.npz",
            "router_scores": "router_scores.npz",
            "fold_assignments": "fold_assignments.json",
        }.items()
    }
    for name, path in task3f_paths.items():
        if not path.exists():
            raise Task3FBDiagnosticError(f"missing Task 3F-A artifact {name}: {path}")
    task3f_hashes_before = {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in task3f_paths.items()
    }
    experiment_config = _load_json(task3f_paths["experiment_config"], name="experiment_config")
    ridge_results = _load_json(task3f_paths["ridge_results"], name="ridge_results")
    fold_assignments = _load_json(task3f_paths["fold_assignments"], name="fold_assignments")
    predictions = _load_npz(task3f_paths["held_out_predictions"], name="held_out_predictions")
    scores = _load_npz(task3f_paths["router_scores"], name="router_scores")

    manager = NestedOOFFoldManager.from_canonical_training_data(
        data_root,
        seed=42,
        outer_folds=5,
        inner_folds=4,
        expert_order=EXPERT_ORDER,
    )
    dataset, source_files = load_restricted_analysis_dataset(source_directory, manager)
    _validate_task3f_inputs(
        ridge_directory=ridge_path,
        dataset=dataset,
        experiment_config=experiment_config,
        ridge_results=ridge_results,
        predictions=predictions,
        scores=scores,
        fold_assignments=fold_assignments,
    )
    class_counts = np.asarray(manager.canonical_class_counts, dtype=np.int64)
    _require(len(class_counts) == NUM_CLASSES, "canonical class count vector is incompatible")
    _require(
        dataset.outer_fold_ids.shape == dataset.sample_indices.shape
        and np.all(dataset.outer_fold_ids == OUTER_FOLD),
        "validated dataset contains a nonzero outer fold",
    )

    adaptive_ids = np.asarray(predictions["adaptive_configuration_ids"]).astype(str)
    adaptive_index = _find_unique_string_index(
        adaptive_ids,
        HIGHLIGHTED_CONFIGURATION_ID,
        name="adaptive configuration IDs",
    )
    global_ids = np.asarray(predictions["global_configuration_ids"]).astype(str)
    global_index = _find_unique_string_index(
        global_ids,
        GLOBAL_CONFIGURATION_ID,
        name="global configuration IDs",
    )
    baseline_ids = np.asarray(predictions["baseline_ids"]).astype(str)
    uniform_index = _find_unique_string_index(
        baseline_ids, "uniform_logit", name="baseline IDs"
    )
    ridge_predictions = np.asarray(predictions["adaptive_predictions"][adaptive_index], dtype=np.int64)
    ridge_weights = np.asarray(predictions["adaptive_weights"][adaptive_index], dtype=np.float64)
    global_predictions = np.asarray(predictions["global_predictions"][global_index], dtype=np.int64)
    global_weights = np.asarray(predictions["global_weights"][global_index], dtype=np.float64)
    uniform_predictions = np.asarray(predictions["baseline_predictions"][uniform_index], dtype=np.int64)
    _validate_weights(ridge_weights, num_samples=dataset.num_samples)
    _validate_weights(global_weights, num_samples=dataset.num_samples)
    _validate_prediction_vector(
        ridge_predictions, name="highlighted Ridge predictions", num_samples=dataset.num_samples
    )
    _validate_prediction_vector(
        global_predictions, name="global-control predictions", num_samples=dataset.num_samples
    )
    _validate_prediction_vector(
        uniform_predictions, name="uniform predictions", num_samples=dataset.num_samples
    )
    _require(
        np.array_equal(
            uniform_predictions,
            combine_weighted_logits(
                dataset.logits,
                np.full(len(EXPERT_ORDER), 1.0 / len(EXPERT_ORDER)),
            ).argmax(axis=1),
        ),
        "saved uniform predictions do not match the validated OOF logits",
    )
    _require(
        np.array_equal(
            ridge_predictions,
            combine_weighted_logits(
                dataset.logits,
                _weights_for_existing_combiner(
                    ridge_weights, num_samples=dataset.num_samples
                ),
            ).argmax(axis=1),
        ),
        "saved highlighted Ridge predictions do not match saved weights and logits",
    )
    _require(
        np.array_equal(
            global_predictions,
            combine_weighted_logits(
                dataset.logits,
                _weights_for_existing_combiner(
                    global_weights, num_samples=dataset.num_samples
                ),
            ).argmax(axis=1),
        ),
        "saved global-control predictions do not match saved weights and logits",
    )

    actual_targets = compute_contribution_targets(dataset.logits, dataset.labels)
    expert_predictions = dataset.logits.argmax(axis=2).astype(np.int64)
    confidence_features = extract_features(dataset.logits, FEATURE_SET_CONFIDENCE)
    model_records = {
        str(record.get("model_key")): record
        for record in ridge_results.get("model_fits", [])
        if isinstance(record, Mapping)
    }
    highlighted_model_keys = {
        fold: f"ridge_fold{fold}_confidence_only_alpha1000_gamma1"
        for fold in PERMITTED_ANALYSIS_INNER_FOLDS
    }
    for model_key in highlighted_model_keys.values():
        _require(model_key in model_records, f"missing highlighted saved model {model_key}")

    score_model_ids = np.asarray(scores["model_ids"]).astype(str)
    held_out_scores = np.asarray(scores["held_out_scores"])
    highlighted_scores, score_indices = _compose_highlighted_scores(
        model_ids=score_model_ids,
        held_out_scores=held_out_scores,
        dataset=dataset,
        model_keys_by_fold=highlighted_model_keys,
    )

    diagnostic_a_folds: list[dict[str, Any]] = []
    for fold in PERMITTED_ANALYSIS_INNER_FOLDS:
        record = model_records[highlighted_model_keys[fold]]
        _require(
            record.get("validation_inner_fold") == fold
            and record.get("feature_set") == FEATURE_SET_CONFIDENCE
            and float(record.get("alpha")) == 1000.0
            and float(record.get("gamma")) == 1.0,
            f"saved highlighted model metadata is incompatible for fold {fold}",
        )
        validation_mask = dataset.inner_fold_ids == fold
        training_mask = ~validation_mask
        train_decomposition = decompose_ridge_scores(
            confidence_features[training_mask],
            record["intercept"],
            record["coefficients"],
            record["scaler_mean"],
            record["scaler_scale"],
        )
        validation_decomposition = decompose_ridge_scores(
            confidence_features[validation_mask],
            record["intercept"],
            record["coefficients"],
            record["scaler_mean"],
            record["scaler_scale"],
        )
        saved_validation_scores = held_out_scores[score_indices[fold], validation_mask].astype(
            np.float64
        )
        _require(
            np.allclose(
                saved_validation_scores,
                validation_decomposition["predicted_scores"],
                rtol=0.0,
                atol=SCORE_RECONSTRUCTION_ATOL,
            ),
            f"saved Ridge scores cannot be reconstructed for fold {fold}",
        )

        def means(decomposition: Mapping[str, np.ndarray]) -> dict[str, list[float]]:
            return {
                "feature_term": [
                    float(value) for value in decomposition["feature_term"].mean(axis=0)
                ],
                "absolute_feature_term": [
                    float(value)
                    for value in np.abs(decomposition["feature_term"]).mean(axis=0)
                ],
            }

        target_train = actual_targets[training_mask]
        target_validation = actual_targets[validation_mask]
        predicted_validation = validation_decomposition["predicted_scores"]
        diagnostic_a_folds.append(
            {
                "validation_inner_fold": int(fold),
                "model_key": highlighted_model_keys[fold],
                "feature_set": FEATURE_SET_CONFIDENCE,
                "feature_names": list(FEATURE_NAMES[FEATURE_SET_CONFIDENCE]),
                "alpha": float(record["alpha"]),
                "gamma": float(record["gamma"]),
                "score_formula": (
                    "score_e(x) = intercept_e + sum_j coefficient[e,j] * "
                    "((feature_j(x) - scaler_mean_j) / scaler_scale_j)"
                ),
                "intercept": {
                    expert: float(record["intercept"][index])
                    for index, expert in enumerate(EXPERT_ORDER)
                },
                "coefficients": {
                    expert: [
                        float(value) for value in record["coefficients"][index]
                    ]
                    for index, expert in enumerate(EXPERT_ORDER)
                },
                "scaler_mean": [float(value) for value in record["scaler_mean"]],
                "scaler_scale": [float(value) for value in record["scaler_scale"]],
                "mean_training_target": {
                    expert: float(target_train[:, index].mean())
                    for index, expert in enumerate(EXPERT_ORDER)
                },
                "mean_held_out_actual_target": {
                    expert: float(target_validation[:, index].mean())
                    for index, expert in enumerate(EXPERT_ORDER)
                },
                "mean_held_out_predicted_score": {
                    expert: float(predicted_validation[:, index].mean())
                    for index, expert in enumerate(EXPERT_ORDER)
                },
                "held_out_target_std": {
                    expert: float(target_validation[:, index].std())
                    for index, expert in enumerate(EXPERT_ORDER)
                },
                "held_out_mean_feature_term": {
                    expert: float(value)
                    for expert, value in zip(
                        EXPERT_ORDER,
                        validation_decomposition["feature_term"].mean(axis=0),
                    )
                },
                "held_out_mean_absolute_feature_term": {
                    expert: float(value)
                    for expert, value in zip(
                        EXPERT_ORDER,
                        np.abs(validation_decomposition["feature_term"]).mean(axis=0),
                    )
                },
                "training_feature_term_means": means(train_decomposition),
                "held_out_score_reconstruction_max_abs_error": float(
                    np.max(np.abs(saved_validation_scores - predicted_validation))
                ),
                "training_mse_saved": float(record["training_mse"]),
                "held_out_mse_saved": float(record["held_out_mse"]),
                "global_control_training_mse_saved": float(
                    record["global_control_training_mse"]
                ),
                "global_control_held_out_mse_saved": float(
                    record["global_control_held_out_mse"]
                ),
                "training_sample_count": int(training_mask.sum()),
                "held_out_sample_count": int(validation_mask.sum()),
            }
        )

    diagnostic_b_groups = compute_group_statistics(
        dataset.labels,
        ridge_weights,
        highlighted_scores,
        actual_targets,
        class_counts,
    )
    diagnostic_c_accounting = tail_gain_loss_accounting(
        dataset.sample_indices,
        dataset.labels,
        uniform_predictions,
        ridge_predictions,
        expert_predictions,
        ridge_weights,
        class_counts,
    )
    uniform_metrics = _metric_report(dataset.labels, uniform_predictions, class_counts)
    ridge_metrics = _metric_report(dataset.labels, ridge_predictions, class_counts)
    diagnostic_c = {
        "tail_accounting": diagnostic_c_accounting,
        "uniform_metrics": uniform_metrics,
        "ridge_metrics": ridge_metrics,
        "metric_delta_ridge_minus_uniform": _round_metric_delta(
            ridge_metrics, uniform_metrics
        ),
    }

    methods: dict[str, dict[str, Any]] = {
        "uniform_logit": {
            "metrics": uniform_metrics,
            "mean_weight": {
                expert: 0.25 for expert in EXPERT_ORDER
            },
            "weight_std": {expert: 0.0 for expert in EXPERT_ORDER},
            "source": "saved Task 3F-A uniform_logit row",
        },
        "ridge_highlighted": {
            "metrics": ridge_metrics,
            "mean_weight": {
                expert: float(ridge_weights[:, index].mean())
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "weight_std": {
                expert: float(ridge_weights[:, index].std())
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "source": HIGHLIGHTED_CONFIGURATION_ID,
        },
        "global_control": {
            "metrics": _metric_report(dataset.labels, global_predictions, class_counts),
            "mean_weight": {
                expert: float(global_weights[:, index].mean())
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "weight_std": {
                expert: float(global_weights[:, index].std())
                for index, expert in enumerate(EXPERT_ORDER)
            },
            "source": GLOBAL_CONFIGURATION_ID,
        },
    }
    for baseline_id, units in FIXED_REFERENCE_UNITS.items():
        fixed_weights = np.asarray(units, dtype=np.float64) / 4.0
        fixed_predictions = combine_weighted_logits(dataset.logits, fixed_weights).argmax(axis=1)
        methods[baseline_id] = {
            "metrics": _metric_report(dataset.labels, fixed_predictions, class_counts),
            "weight_vector": [float(value) for value in fixed_weights.tolist()],
            "source": "Task 3F-A fixed reference definition",
        }
    historical_weights = np.asarray(HISTORICAL_NO_CE_WEIGHTS, dtype=np.float64)
    historical_predictions = combine_weighted_logits(
        dataset.logits, historical_weights
    ).argmax(axis=1)
    methods["uniform_without_ce"] = {
        "metrics": _metric_report(
            dataset.labels, historical_predictions, class_counts
        ),
        "weight_vector": [float(value) for value in historical_weights.tolist()],
        "source": "Task 3F-A historical fixed reference definition",
    }
    for index, expert in enumerate(EXPERT_ORDER):
        predictions_for_expert = expert_predictions[:, index]
        methods[f"expert_{expert}"] = {
            "metrics": _metric_report(dataset.labels, predictions_for_expert, class_counts),
            "source": "validated Task 3C restricted OOF logits",
        }

    adaptive_global_difference = np.abs(ridge_weights - global_weights)
    adaptive_vs_global = {
        "tolerance": WEIGHT_COMPARISON_TOLERANCE,
        "adaptive_mean_weight": methods["ridge_highlighted"]["mean_weight"],
        "global_mean_weight": methods["global_control"]["mean_weight"],
        "adaptive_std_weight": methods["ridge_highlighted"]["weight_std"],
        "global_std_weight": methods["global_control"]["weight_std"],
        "mean_absolute_weight_difference": float(adaptive_global_difference.mean()),
        "maximum_absolute_weight_difference": float(adaptive_global_difference.max()),
        "fraction_rows_with_meaningful_difference": float(
            np.mean(np.max(adaptive_global_difference, axis=1) > WEIGHT_COMPARISON_TOLERANCE)
        ),
    }
    sensitivity: list[dict[str, Any]] = []
    for factor in MIXUP_SENSITIVITY_FACTORS:
        sensitivity_weights = renormalize_mixup_weights(ridge_weights, factor)
        sensitivity_predictions = combine_weighted_logits(
            dataset.logits,
            _weights_for_existing_combiner(
                sensitivity_weights, num_samples=dataset.num_samples
            ),
        ).argmax(axis=1)
        metrics = _metric_report(dataset.labels, sensitivity_predictions, class_counts)
        sensitivity.append(
            {
                "factor": float(factor),
                "metrics": metrics,
                "mean_weight": {
                    expert: float(sensitivity_weights[:, index].mean())
                    for index, expert in enumerate(EXPERT_ORDER)
                },
                "prediction_equal_saved_highlighted_ridge": bool(
                    np.array_equal(sensitivity_predictions, ridge_predictions)
                ),
            }
        )
    _require(
        sensitivity[0]["prediction_equal_saved_highlighted_ridge"] is True,
        "Mixup factor 1.0 does not reproduce the saved Ridge predictions",
    )
    diagnostic_d = {
        "methods": methods,
        "adaptive_vs_global": adaptive_vs_global,
        "mixup_sensitivity": sensitivity,
    }

    task3f_hashes_after = {
        name: {"path": str(path), "sha256": _sha256_file(path)}
        for name, path in task3f_paths.items()
    }
    _require(
        task3f_hashes_before == task3f_hashes_after,
        "a Task 3F-A input artifact changed during diagnostics",
    )
    source_hashes = {
        name: {
            "path": str(details["path"]),
            "sha256": str(details["sha256"]),
        }
        for name, details in source_files.items()
    }
    results: dict[str, Any] = {
        "task_identifier": TASK3FB_TASK_IDENTIFIER,
        "schema_version": TASK3FB_SCHEMA_VERSION,
        "source_git_commit": _git_commit(project_root_path),
        "provenance": {
            "dataset": DATASET_NAME,
            "outer_fold": OUTER_FOLD,
            "permitted_inner_folds": list(PERMITTED_ANALYSIS_INNER_FOLDS),
            "reserved_inner_folds": [0],
            "analyzed_sample_count": dataset.num_samples,
            "expert_order": list(EXPERT_ORDER),
            "labels_used_only_for_retrospective_diagnostics": True,
            "oracle_weights_used": False,
            "training_or_checkpoints_modified": False,
            "original_cifar_test_used": False,
            "reserved_outer_evaluation_used": False,
            "task3f_a_artifacts": task3f_hashes_before,
            "task3c_oof_artifacts": source_hashes,
            "task3f_a_config_input_artifacts": experiment_config.get("input_artifacts", {}),
        },
        "highlighted_configuration_id": HIGHLIGHTED_CONFIGURATION_ID,
        "global_control_configuration_id": GLOBAL_CONFIGURATION_ID,
        "diagnostic_a": {
            "formula": (
                "score_e(x) = intercept_e + sum_j coefficient[e,j] * "
                "((feature_j(x) - scaler_mean_j) / scaler_scale_j)"
            ),
            "folds": diagnostic_a_folds,
        },
        "diagnostic_b": {
            "group_definitions_source": "scripts.base_trainer.compute_class_groups",
            "canonical_class_counts": [int(value) for value in class_counts.tolist()],
            "groups": diagnostic_b_groups,
        },
        "diagnostic_c": diagnostic_c,
        "diagnostic_d": diagnostic_d,
    }
    summary = _render_summary(results)
    result_path = output_path / "diagnostic_results.json"
    summary_path = output_path / "summary.md"
    _write_json_once(result_path, results)
    _write_text_once(summary_path, summary)
    return {
        "diagnostic_results": result_path,
        "summary": summary_path,
    }


__all__ = [
    "GLOBAL_CONFIGURATION_ID",
    "HIGHLIGHTED_CONFIGURATION_ID",
    "MIXUP_SENSITIVITY_FACTORS",
    "Task3FBDiagnosticError",
    "compute_group_statistics",
    "decompose_ridge_scores",
    "renormalize_mixup_weights",
    "run_task3f_mixup_diagnostics",
    "tail_gain_loss_accounting",
    "validate_artifact_alignment",
]
