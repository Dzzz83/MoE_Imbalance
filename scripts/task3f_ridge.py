"""Task 3F-A: predictable Ridge routing on the restricted OOF population.

This module is deliberately analysis-only.  It consumes the already validated
Task 3C aligned OOF artifact through the restricted Task 3E-A loader, fits
small deterministic multi-output Ridge models on the three permitted inner
folds, and writes development-only diagnostics.  It never loads images,
checkpoints, the reserved outer population, or the CIFAR-100 test set.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from data.nested_oof import NestedOOFFoldManager, OOFProtocolError
from scripts.analysis.artifacts import (
    git_commit,
    load_json_object,
    repository_relative,
    sha256_array,
    sha256_file,
    write_json_once,
    write_npz_once,
)
from scripts.analysis.combination import (
    combine_weighted_logits as _combine_weighted_logits,
    combine_weighted_probabilities as _combine_weighted_probabilities,
    stable_softmax,
)
from scripts.analysis.validation import (
    validate_expert_weights,
    validate_integer_vector,
    validate_numeric_array,
)
from scripts.base_trainer import compute_class_groups
from scripts.evaluation import balanced_accuracy
from scripts.task3e_fixed import (
    EXPERT_ORDER,
    RestrictedAnalysisDataset,
    load_restricted_analysis_dataset,
    verify_uniform_baseline,
)


TASK3F_SCHEMA_VERSION = "task3f_ridge.v1"
TASK3F_RESULTS_SCHEMA_VERSION = "task3f_ridge_results.v1"
TASK3F_TASK_IDENTIFIER = "Task 3F-A"

DATASET_NAME = "CIFAR-100-LT"
IMBALANCE_RATIO = 100.0
CANONICAL_POPULATION_SIZE = 10_847
FULL_ALIGNED_SAMPLE_COUNT = 8_677
ANALYSIS_SAMPLE_COUNT = 6_507
NUM_CLASSES = 100
TRAINING_SEED = 78
FOLD_GENERATION_SEED = 42
OUTER_FOLD = 0
PERMITTED_ANALYSIS_INNER_FOLDS = (1, 2, 3)
RESERVED_ROUTER_SELECTION_INNER_FOLDS = (0,)
OUTER_EVALUATION_SIZE = 2_170

FEATURE_SET_CONFIDENCE = "confidence_only"
FEATURE_SET_FULL = "full_13"
FEATURE_SETS = (FEATURE_SET_CONFIDENCE, FEATURE_SET_FULL)
FEATURE_SET_ALIASES = {
    "A": FEATURE_SET_CONFIDENCE,
    "a": FEATURE_SET_CONFIDENCE,
    "confidence": FEATURE_SET_CONFIDENCE,
    "confidence_only": FEATURE_SET_CONFIDENCE,
    "B": FEATURE_SET_FULL,
    "b": FEATURE_SET_FULL,
    "full": FEATURE_SET_FULL,
    "full_13": FEATURE_SET_FULL,
}
FEATURE_NAMES = {
    FEATURE_SET_CONFIDENCE: (
        "confidence_CE",
        "confidence_LAL",
        "confidence_BalancedSoftmax",
        "confidence_Mixup",
    ),
    FEATURE_SET_FULL: (
        "confidence_CE",
        "confidence_LAL",
        "confidence_BalancedSoftmax",
        "confidence_Mixup",
        "margin_CE",
        "margin_LAL",
        "margin_BalancedSoftmax",
        "margin_Mixup",
        "entropy_CE",
        "entropy_LAL",
        "entropy_BalancedSoftmax",
        "entropy_Mixup",
        "distinct_top1_predictions",
    ),
}

ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
GAMMAS = (0.0, 0.5, 1.0)
TEMPERATURES = (0.5, 1.0, 2.0, 5.0)
SHRINKAGES = (0.0, 0.25, 0.5, 0.75, 1.0)
WEIGHT_CAP = 5.0
WEIGHT_DIFFERENCE_TOLERANCE = 1e-6
UNIFORM_BASELINE_TOLERANCE = 1e-12
RIDGE_SOLVER = "lsqr"
RIDGE_TOLERANCE = 1e-12

FIXED_REFERENCE_UNITS = {
    "fixed_006": (0, 1, 1, 2),
    "fixed_007": (0, 1, 2, 1),
    "fixed_010": (0, 2, 1, 1),
    "fixed_011": (0, 2, 2, 0),
}
HISTORICAL_NO_CE_WEIGHTS = (0.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)


class Task3FError(OOFProtocolError):
    """Raised when Task 3F-A inputs or calculations violate the contract."""


def _as_float_array(value: Any, *, name: str) -> np.ndarray:
    return validate_numeric_array(
        value,
        name=name,
        finite=True,
        cast_dtype=np.float64,
        error_type=Task3FError,
    )


def _validate_logits(logits: Any, *, name: str = "logits") -> np.ndarray:
    array = _as_float_array(logits, name=name)
    if array.ndim != 3:
        raise Task3FError(f"{name} must have shape (samples, 4, classes), got {array.shape}")
    if array.shape[0] == 0:
        raise Task3FError(f"{name} must contain at least one sample")
    if array.shape[1] != len(EXPERT_ORDER):
        raise Task3FError(
            f"{name} must contain exactly {len(EXPERT_ORDER)} experts in the frozen order"
        )
    if array.shape[2] < 2:
        raise Task3FError(f"{name} must contain at least two classes")
    return array


def _validate_labels(labels: Any, *, num_samples: int, num_classes: int) -> np.ndarray:
    return validate_integer_vector(
        labels,
        name="labels",
        shape=(num_samples,),
        lower_bound=0,
        upper_bound=num_classes,
        error_type=Task3FError,
    )


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Return a finite softmax without changing the supplied logits."""
    array = _validate_logits(logits)
    probabilities = stable_softmax(array)
    if not np.isfinite(probabilities).all():
        raise Task3FError("stable softmax produced non-finite probabilities")
    return probabilities


def _canonical_feature_set(feature_set: str) -> str:
    try:
        canonical = FEATURE_SET_ALIASES[str(feature_set)]
    except KeyError as exc:
        raise Task3FError(
            f"feature_set must be {FEATURE_SET_CONFIDENCE!r} or {FEATURE_SET_FULL!r}"
        ) from exc
    return canonical


def compute_contribution_targets(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute marginal true-class log-probability contributions.

    For each expert this returns

    ``(z_e,y - z_bar,y) - sum_c softmax(z_bar)_c * (z_e,c - z_bar,c)``.

    Labels are accepted only because this is the supervised target-construction
    seam.  The function never returns or stores modified logits.
    """
    array = _validate_logits(logits)
    labels_array = _validate_labels(
        labels,
        num_samples=array.shape[0],
        num_classes=array.shape[2],
    )
    ensemble_logits = array.mean(axis=1)
    shifted_ensemble = ensemble_logits - ensemble_logits.max(axis=1, keepdims=True)
    ensemble_exponentials = np.exp(shifted_ensemble)
    ensemble_probabilities = ensemble_exponentials / ensemble_exponentials.sum(
        axis=1, keepdims=True
    )
    if not np.isfinite(ensemble_probabilities).all():
        raise Task3FError("uniform ensemble softmax produced non-finite probabilities")
    deltas = array - ensemble_logits[:, None, :]
    true_class_delta = np.take_along_axis(
        deltas,
        labels_array[:, None, None],
        axis=2,
    ).squeeze(axis=2)
    expected_delta = np.einsum("nc,nec->ne", ensemble_probabilities, deltas)
    targets = true_class_delta - expected_delta
    if not np.isfinite(targets).all():
        raise Task3FError("contribution targets contain non-finite values")
    if not np.allclose(targets.sum(axis=1), 0.0, rtol=1e-9, atol=1e-12):
        raise Task3FError("expert contribution targets do not sum to zero")
    return targets


def extract_features(logits: np.ndarray, feature_set: str) -> np.ndarray:
    """Extract one of the two frozen inference-time feature representations."""
    array = _validate_logits(logits)
    canonical = _canonical_feature_set(feature_set)
    probabilities = _stable_softmax(array)
    confidence = probabilities.max(axis=2)
    if canonical == FEATURE_SET_CONFIDENCE:
        features = confidence
    else:
        sorted_logits = np.sort(array, axis=2)
        margins = sorted_logits[:, :, -1] - sorted_logits[:, :, -2]
        log_probabilities = np.zeros_like(probabilities)
        np.log(probabilities, out=log_probabilities, where=probabilities > 0.0)
        entropy = -np.sum(
            np.where(probabilities > 0.0, probabilities * log_probabilities, 0.0),
            axis=2,
        )
        predictions = array.argmax(axis=2)
        disagreement = np.fromiter(
            (np.unique(row).size for row in predictions),
            dtype=np.float64,
            count=array.shape[0],
        )[:, None]
        features = np.concatenate((confidence, margins, entropy, disagreement), axis=1)
    if features.shape[1] != len(FEATURE_NAMES[canonical]):
        raise Task3FError("feature extraction produced the wrong number of columns")
    if not np.isfinite(features).all():
        raise Task3FError("inference-time features contain non-finite values")
    return features.astype(np.float64, copy=False)


def compute_sample_weights(
    labels: np.ndarray,
    gamma: float,
    *,
    cap: float = WEIGHT_CAP,
) -> np.ndarray:
    """Calculate normalized rare-class sample weights from one training split."""
    labels_array = np.asarray(labels)
    if labels_array.ndim != 1 or len(labels_array) == 0:
        raise Task3FError("training labels must be a non-empty one-dimensional array")
    if not np.issubdtype(labels_array.dtype, np.number) or np.iscomplexobj(labels_array):
        raise Task3FError("training labels must be integer-valued")
    if not np.isfinite(labels_array).all() or not np.equal(labels_array, np.floor(labels_array)).all():
        raise Task3FError("training labels must be finite integer-valued class indices")
    labels_array = labels_array.astype(np.int64, copy=False)
    if np.any(labels_array < 0):
        raise Task3FError("training labels must be non-negative")
    if isinstance(gamma, bool) or not np.isfinite(float(gamma)) or float(gamma) < 0.0:
        raise Task3FError("gamma must be a finite non-negative number")
    if isinstance(cap, bool) or not np.isfinite(float(cap)) or float(cap) <= 0.0:
        raise Task3FError("weight cap must be finite and positive")

    counts = np.bincount(labels_array)
    n_max = float(counts.max())
    per_sample_count = counts[labels_array].astype(np.float64)
    weights = np.minimum(
        float(cap),
        np.power(n_max / per_sample_count, float(gamma)),
    )
    weights /= weights.mean()
    if not np.isfinite(weights).all() or np.any(weights <= 0.0):
        raise Task3FError("sample weights must be finite and positive")
    if not np.isclose(weights.mean(), 1.0, rtol=0.0, atol=1e-12):
        raise Task3FError("sample weights were not normalized to mean one")
    return weights


def sample_weight_report(labels: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    """Serialize the observed training-fold weight distribution."""
    labels_array = np.asarray(labels, dtype=np.int64)
    weights_array = _as_float_array(weights, name="sample weights")
    if weights_array.ndim != 1 or weights_array.shape != labels_array.shape:
        raise Task3FError("sample weights and training labels are misaligned")
    if np.any(weights_array <= 0.0):
        raise Task3FError("sample weights must be positive")
    counts = np.bincount(labels_array)
    return {
        "count": int(len(weights_array)),
        "mean": float(weights_array.mean()),
        "std": float(weights_array.std()),
        "minimum": float(weights_array.min()),
        "maximum": float(weights_array.max()),
        "unique_values": sorted(float(value) for value in np.unique(weights_array)),
        "class_counts": [int(value) for value in counts.tolist()],
        "class_weight_by_class": {
            str(class_id): float(weights_array[labels_array == class_id][0])
            for class_id in range(len(counts))
        },
    }


def _validate_alpha(alpha: float) -> float:
    if isinstance(alpha, bool) or not np.isfinite(float(alpha)) or float(alpha) <= 0.0:
        raise Task3FError("alpha must be finite and positive")
    return float(alpha)


def _validate_temperature(temperature: float) -> float:
    if isinstance(temperature, bool) or not np.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise Task3FError("temperature must be finite and positive")
    return float(temperature)


def _validate_shrinkage(shrinkage: float) -> float:
    if (
        isinstance(shrinkage, bool)
        or not np.isfinite(float(shrinkage))
        or not 0.0 <= float(shrinkage) <= 1.0
    ):
        raise Task3FError("shrinkage must be finite and in [0, 1]")
    return float(shrinkage)


def scores_to_weights(
    scores: np.ndarray,
    temperature: float,
    shrinkage: float,
) -> np.ndarray:
    """Convert contribution scores into valid convex expert weights."""
    array = _as_float_array(scores, name="Ridge contribution scores")
    if array.ndim not in (1, 2) or array.shape[-1] != len(EXPERT_ORDER):
        raise Task3FError("scores must have shape (4,) or (samples, 4)")
    temperature = _validate_temperature(temperature)
    shrinkage = _validate_shrinkage(shrinkage)
    was_vector = array.ndim == 1
    matrix = array[None, :] if was_vector else array
    shifted = matrix / temperature
    shifted -= shifted.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    weights = (1.0 - shrinkage) * (1.0 / len(EXPERT_ORDER)) + shrinkage * probabilities
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise Task3FError("generated routing weights are not finite and non-negative")
    if not np.allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise Task3FError("generated routing weights do not sum to one")
    return weights[0] if was_vector else weights


def _validate_weights(weights: np.ndarray, *, num_samples: int) -> np.ndarray:
    return validate_expert_weights(
        weights,
        num_experts=len(EXPERT_ORDER),
        num_samples=num_samples,
        allow_vector=True,
        sum_atol=1e-12,
        name="routing weights",
        error_type=Task3FError,
    )


def combine_weighted_logits(logits: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Combine the original expert logits using supplied convex weights."""
    return _combine_weighted_logits(
        logits,
        weights,
        num_experts=len(EXPERT_ORDER),
        min_classes=2,
        logit_cast_dtype=np.float64,
        error_type=Task3FError,
    )


def _weighted_probability_predictions(logits: np.ndarray, weights: np.ndarray) -> np.ndarray:
    combined = _combine_weighted_probabilities(
        logits,
        weights,
        num_experts=len(EXPERT_ORDER),
        min_classes=2,
        logit_cast_dtype=np.float64,
        error_type=Task3FError,
    )
    return combined.argmax(axis=1).astype(np.int64)


@dataclass
class RidgeRouterFit:
    """A fitted deterministic Ridge router and its training-only diagnostics."""

    feature_set: str
    alpha: float
    gamma: float
    scaler: StandardScaler
    model: Ridge
    training_targets: np.ndarray
    training_sample_weights: np.ndarray
    training_scores: np.ndarray

    def predict_scores(self, logits: np.ndarray) -> np.ndarray:
        features = extract_features(logits, self.feature_set)
        scaled = self.scaler.transform(features)
        scores = np.asarray(self.model.predict(scaled), dtype=np.float64)
        if scores.ndim != 2 or scores.shape[1] != len(EXPERT_ORDER):
            raise Task3FError("Ridge prediction has the wrong contribution-score shape")
        if not np.isfinite(scores).all():
            raise Task3FError("Ridge prediction contains non-finite contribution scores")
        return scores


def fit_ridge_router(
    logits: np.ndarray,
    labels: np.ndarray,
    feature_set: str,
    alpha: float,
    gamma: float,
    *,
    solver: str = RIDGE_SOLVER,
    tolerance: float = RIDGE_TOLERANCE,
) -> RidgeRouterFit:
    """Fit one multi-output Ridge model using only the supplied training rows."""
    canonical = _canonical_feature_set(feature_set)
    alpha = _validate_alpha(alpha)
    if solver != RIDGE_SOLVER:
        raise Task3FError(f"deterministic Ridge solver is fixed at {RIDGE_SOLVER!r}")
    if isinstance(tolerance, bool) or not np.isfinite(float(tolerance)) or float(tolerance) <= 0.0:
        raise Task3FError("Ridge tolerance must be finite and positive")
    array = _validate_logits(logits)
    labels_array = _validate_labels(
        labels,
        num_samples=array.shape[0],
        num_classes=array.shape[2],
    )
    features = extract_features(array, canonical)
    targets = compute_contribution_targets(array, labels_array)
    sample_weights = compute_sample_weights(labels_array, gamma)
    return _fit_ridge_from_precomputed(
        features,
        targets,
        sample_weights,
        feature_set=canonical,
        alpha=alpha,
        gamma=gamma,
        solver=solver,
        tolerance=tolerance,
    )


def _fit_ridge_from_precomputed(
    features: np.ndarray,
    targets: np.ndarray,
    sample_weights: np.ndarray,
    *,
    feature_set: str,
    alpha: float,
    gamma: float,
    solver: str = RIDGE_SOLVER,
    tolerance: float = RIDGE_TOLERANCE,
) -> RidgeRouterFit:
    """Fit Ridge after CV orchestration has cached features and targets."""
    canonical = _canonical_feature_set(feature_set)
    alpha = _validate_alpha(alpha)
    if solver != RIDGE_SOLVER:
        raise Task3FError(f"deterministic Ridge solver is fixed at {RIDGE_SOLVER!r}")
    if isinstance(tolerance, bool) or not np.isfinite(float(tolerance)) or float(tolerance) <= 0.0:
        raise Task3FError("Ridge tolerance must be finite and positive")
    features_array = _as_float_array(features, name="training features")
    targets_array = _as_float_array(targets, name="training contribution targets")
    weights_array = _as_float_array(sample_weights, name="training sample weights")
    expected_features = len(FEATURE_NAMES[canonical])
    if features_array.ndim != 2 or features_array.shape[1] != expected_features:
        raise Task3FError(
            f"training features must have shape (samples, {expected_features})"
        )
    if targets_array.shape != (len(features_array), len(EXPERT_ORDER)):
        raise Task3FError("training contribution targets have the wrong shape")
    if weights_array.shape != (len(features_array),):
        raise Task3FError("training sample weights have the wrong shape")
    if np.any(weights_array <= 0.0) or not np.isclose(
        weights_array.mean(), 1.0, rtol=0.0, atol=1e-12
    ):
        raise Task3FError("training sample weights must be finite, positive, and mean one")
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaled_features = scaler.fit_transform(features_array)
    if not np.isfinite(scaled_features).all():
        raise Task3FError("standardized training features contain non-finite values")
    model = Ridge(
        alpha=alpha,
        fit_intercept=True,
        solver=solver,
        tol=float(tolerance),
        random_state=None,
    )
    model.fit(scaled_features, targets_array, sample_weight=weights_array)
    training_scores = np.asarray(model.predict(scaled_features), dtype=np.float64)
    if not np.isfinite(training_scores).all():
        raise Task3FError("Ridge training predictions contain non-finite values")
    if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
        raise Task3FError("Ridge parameters contain non-finite values")
    return RidgeRouterFit(
        feature_set=canonical,
        alpha=alpha,
        gamma=float(gamma),
        scaler=scaler,
        model=model,
        training_targets=targets_array,
        training_sample_weights=weights_array,
        training_scores=training_scores,
    )


@dataclass
class GlobalScoreControl:
    """Intercept-only, training-only contribution-score control."""

    gamma: float
    scores: np.ndarray
    training_targets: np.ndarray
    training_sample_weights: np.ndarray

    def predict_scores(self, num_samples: int) -> np.ndarray:
        if isinstance(num_samples, bool) or int(num_samples) < 1:
            raise Task3FError("num_samples must be a positive integer")
        return np.repeat(self.scores[None, :], int(num_samples), axis=0)


def fit_global_score_control(
    logits: np.ndarray,
    labels: np.ndarray,
    gamma: float,
) -> GlobalScoreControl:
    """Fit the weighted intercept-only contribution-score model."""
    array = _validate_logits(logits)
    labels_array = _validate_labels(
        labels,
        num_samples=array.shape[0],
        num_classes=array.shape[2],
    )
    targets = compute_contribution_targets(array, labels_array)
    sample_weights = compute_sample_weights(labels_array, gamma)
    return _fit_global_from_precomputed(targets, sample_weights, gamma)


def _fit_global_from_precomputed(
    targets: np.ndarray,
    sample_weights: np.ndarray,
    gamma: float,
) -> GlobalScoreControl:
    """Fit an intercept-only control from cached targets and weights."""
    targets = _as_float_array(targets, name="training contribution targets")
    sample_weights = _as_float_array(sample_weights, name="training sample weights")
    if targets.ndim != 2 or targets.shape[1] != len(EXPERT_ORDER):
        raise Task3FError("global-control targets have the wrong shape")
    if sample_weights.shape != (len(targets),):
        raise Task3FError("global-control sample weights have the wrong shape")
    if np.any(sample_weights <= 0.0) or not np.isclose(
        sample_weights.mean(), 1.0, rtol=0.0, atol=1e-12
    ):
        raise Task3FError("global-control sample weights must be positive and mean one")
    scores = np.average(targets, axis=0, weights=sample_weights)
    scores = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(scores).all():
        raise Task3FError("global contribution scores are non-finite")
    return GlobalScoreControl(
        gamma=float(gamma),
        scores=scores,
        training_targets=targets,
        training_sample_weights=sample_weights,
    )


def _mean_squared_error(predictions: np.ndarray, targets: np.ndarray) -> float:
    predictions = _as_float_array(predictions, name="predicted contribution scores")
    targets = _as_float_array(targets, name="contribution targets")
    if predictions.shape != targets.shape:
        raise Task3FError("predicted contribution scores and targets are misaligned")
    return float(np.mean((predictions - targets) ** 2))


def _validate_class_counts(class_counts: np.ndarray, *, num_classes: int) -> np.ndarray:
    counts = np.asarray(class_counts)
    if counts.ndim != 1 or len(counts) != num_classes:
        raise Task3FError(f"class_counts must have shape ({num_classes},)")
    if not np.issubdtype(counts.dtype, np.number) or np.iscomplexobj(counts):
        raise Task3FError("class_counts must contain real counts")
    if (
        not np.isfinite(counts).all()
        or np.any(counts < 1)
        or not np.equal(counts, np.floor(counts)).all()
    ):
        raise Task3FError("class_counts must contain positive integer counts")
    return counts.astype(np.int64, copy=False)


def _class_recalls(
    labels: np.ndarray,
    predictions: np.ndarray,
    *,
    num_classes: int,
) -> np.ndarray:
    recalls = np.full(num_classes, np.nan, dtype=np.float64)
    for class_id in np.unique(labels):
        mask = labels == class_id
        recalls[int(class_id)] = float(np.mean(predictions[mask] == class_id))
    return recalls


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_counts: np.ndarray,
) -> dict[str, Any]:
    """Report the project's canonical sample and macro-recall metrics."""
    labels_array = np.asarray(labels)
    predictions_array = np.asarray(predictions)
    if labels_array.ndim != 1 or predictions_array.shape != labels_array.shape:
        raise Task3FError("labels and predictions are not aligned")
    counts = _validate_class_counts(class_counts, num_classes=len(class_counts))
    labels_array = _validate_labels(
        labels_array,
        num_samples=len(labels_array),
        num_classes=len(counts),
    )
    predictions_array = _validate_labels(
        predictions_array,
        num_samples=len(labels_array),
        num_classes=len(counts),
    )
    recalls = _class_recalls(
        labels_array,
        predictions_array,
        num_classes=len(counts),
    )
    groups = compute_class_groups(counts)
    for group_name, classes in groups.items():
        if len(classes) == 0:
            raise Task3FError(f"canonical class group {group_name} is empty")
        if np.isnan(recalls[classes]).any():
            missing = classes[np.isnan(recalls[classes])].tolist()
            raise Task3FError(
                f"labels are missing classes in {group_name} group: {missing}"
            )
    if np.isnan(recalls).any():
        missing = np.flatnonzero(np.isnan(recalls)).tolist()
        raise Task3FError(f"labels are missing classes required for BA: {missing}")
    metrics = {
        "ordinary_accuracy": float(np.mean(predictions_array == labels_array)),
        "balanced_accuracy": float(balanced_accuracy(labels_array, predictions_array)),
        "head_accuracy": float(np.mean(recalls[groups["head"]])),
        "medium_accuracy": float(np.mean(recalls[groups["medium"]])),
        "tail_accuracy": float(np.mean(recalls[groups["tail"]])),
        "per_class_recall": {
            str(class_id): float(value) for class_id, value in enumerate(recalls)
        },
    }
    if not all(np.isfinite(value) for key, value in metrics.items() if key != "per_class_recall"):
        raise Task3FError("classification metrics contain non-finite values")
    return metrics


def _tail_diagnostics(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_counts: np.ndarray,
) -> dict[str, Any]:
    counts = _validate_class_counts(class_counts, num_classes=len(class_counts))
    groups = compute_class_groups(counts)
    tail = groups["tail"]
    mask = np.isin(labels, tail)
    per_class = {}
    zero_classes = []
    for class_id in tail:
        class_mask = labels == class_id
        if not class_mask.any():
            raise Task3FError(f"Tail class {int(class_id)} is absent from the evaluated rows")
        correct = int(np.sum(predictions[class_mask] == class_id))
        per_class[str(int(class_id))] = {
            "sample_count": int(class_mask.sum()),
            "correct_count": correct,
            "recall": float(correct / class_mask.sum()),
        }
        if correct == 0:
            zero_classes.append(int(class_id))
    return {
        "sample_count": int(mask.sum()),
        "correct_count": int(np.sum(predictions[mask] == labels[mask])),
        "per_class_recall": per_class,
        "zero_correct_classes": zero_classes,
        "zero_correct_class_count": len(zero_classes),
    }


def _metric_report(
    labels: np.ndarray,
    predictions: np.ndarray,
    class_counts: np.ndarray,
) -> dict[str, Any]:
    metrics = classification_metrics(labels, predictions, class_counts)
    return {
        "metrics": metrics,
        "tail_diagnostics": _tail_diagnostics(
            np.asarray(labels), np.asarray(predictions), np.asarray(class_counts)
        ),
    }


def _weight_diagnostics(
    weights: np.ndarray,
    global_weights: np.ndarray,
    *,
    tolerance: float = WEIGHT_DIFFERENCE_TOLERANCE,
) -> dict[str, Any]:
    array = _validate_weights(weights, num_samples=len(weights))
    global_array = _validate_weights(global_weights, num_samples=len(weights))
    means = array.mean(axis=0)
    standard_deviations = array.std(axis=0)
    minima = array.min(axis=0)
    maxima = array.max(axis=0)
    max_difference = np.max(np.abs(array - global_array), axis=1)
    argmax_experts = array.argmax(axis=1)
    dominant_counts = np.bincount(argmax_experts, minlength=len(EXPERT_ORDER))
    dominant_index = int(dominant_counts.argmax())
    dominant_fraction = float(dominant_counts[dominant_index] / len(array))
    mean_dominant_index = int(means.argmax())
    mean_dominant_fraction = float(means[mean_dominant_index])
    collapse = dominant_fraction >= 0.9 and mean_dominant_fraction >= 0.9
    return {
        "mean_weight": means.tolist(),
        "std_weight": standard_deviations.tolist(),
        "minimum_weight": minima.tolist(),
        "maximum_weight": maxima.tolist(),
        "fraction_meaningfully_different_from_global": float(np.mean(max_difference > tolerance)),
        "weight_difference_tolerance": tolerance,
        "dominant_expert_index": dominant_index,
        "dominant_expert_name": EXPERT_ORDER[dominant_index],
        "dominant_argmax_fraction": dominant_fraction,
        "mean_weight_dominant_expert_index": mean_dominant_index,
        "mean_weight_dominant_expert_name": EXPERT_ORDER[mean_dominant_index],
        "mean_weight_dominant_fraction": mean_dominant_fraction,
        "collapses_toward_one_expert": bool(collapse),
        "collapse_rule": "dominant argmax fraction and mean weight both >= 0.9",
        "effectively_uniform": bool(
            np.max(np.abs(array - 0.25)) <= tolerance
        ),
    }


def _sha256_int_array(values: np.ndarray) -> str:
    return sha256_array(np.asarray(values, dtype=np.int64))


def _sha256_array(values: np.ndarray) -> str:
    return sha256_array(values)


def _token(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def _adaptive_model_key(validation_fold: int, feature_set: str, alpha: float, gamma: float) -> str:
    return (
        f"ridge_fold{int(validation_fold)}_{feature_set}"
        f"_alpha{_token(alpha)}_gamma{_token(gamma)}"
    )


def _adaptive_configuration_id(
    feature_set: str,
    alpha: float,
    gamma: float,
    temperature: float,
    shrinkage: float,
) -> str:
    return (
        f"adaptive_{feature_set}_alpha{_token(alpha)}_gamma{_token(gamma)}"
        f"_temperature{_token(temperature)}_lambda{_token(shrinkage)}"
    )


def _global_configuration_id(gamma: float, temperature: float, shrinkage: float) -> str:
    return (
        f"global_gamma{_token(gamma)}_temperature{_token(temperature)}"
        f"_lambda{_token(shrinkage)}"
    )


def _grid_validate() -> None:
    if len(FEATURE_SETS) != 2 or len(ALPHAS) != 5 or len(GAMMAS) != 3:
        raise Task3FError("Task 3F-A adaptive Ridge grid dimensions are not frozen")
    if len(TEMPERATURES) != 4 or len(SHRINKAGES) != 5:
        raise Task3FError("Task 3F-A score-to-weight grid dimensions are not frozen")


def _validate_dataset_for_cv(dataset: RestrictedAnalysisDataset) -> None:
    if not isinstance(dataset, RestrictedAnalysisDataset):
        raise Task3FError(
            "RidgeCVAnalyzer accepts only the restricted inner-folds-1–3 dataset"
        )
    if set(np.unique(dataset.inner_fold_ids).tolist()) != set(PERMITTED_ANALYSIS_INNER_FOLDS):
        raise Task3FError("Ridge CV requires exactly inner folds 1–3")
    if np.any(dataset.outer_fold_ids != OUTER_FOLD):
        raise Task3FError("Ridge CV received a nonzero outer fold")


def _fit_record(
    *,
    model_key: str,
    validation_fold: int,
    feature_set: str,
    alpha: float,
    gamma: float,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    train_sample_ids: np.ndarray,
    validation_sample_ids: np.ndarray,
    training_labels: np.ndarray,
    fit: RidgeRouterFit,
    validation_scores: np.ndarray,
    validation_targets: np.ndarray,
    global_control: GlobalScoreControl,
) -> dict[str, Any]:
    global_training_scores = global_control.predict_scores(len(train_indices))
    global_validation_scores = global_control.predict_scores(len(validation_indices))
    return {
        "model_key": model_key,
        "model_type": "adaptive_ridge",
        "validation_inner_fold": int(validation_fold),
        "router_training_sample_count": int(len(train_indices)),
        "router_validation_sample_count": int(len(validation_indices)),
        "router_training_sample_id_sha256": _sha256_int_array(train_sample_ids),
        "router_validation_sample_id_sha256": _sha256_int_array(validation_sample_ids),
        "feature_set": feature_set,
        "feature_names": list(FEATURE_NAMES[feature_set]),
        "alpha": float(alpha),
        "gamma": float(gamma),
        "solver": RIDGE_SOLVER,
        "solver_random_state": None,
        "solver_tolerance": RIDGE_TOLERANCE,
        "training_mse": _mean_squared_error(fit.training_scores, fit.training_targets),
        "held_out_mse": _mean_squared_error(validation_scores, validation_targets),
        "global_control_training_mse": _mean_squared_error(
            global_training_scores, fit.training_targets
        ),
        "global_control_held_out_mse": _mean_squared_error(
            global_validation_scores, validation_targets
        ),
        "adaptive_improves_training_mse_over_global": bool(
            _mean_squared_error(fit.training_scores, fit.training_targets)
            < _mean_squared_error(global_training_scores, fit.training_targets)
        ),
        "adaptive_improves_held_out_mse_over_global": bool(
            _mean_squared_error(validation_scores, validation_targets)
            < _mean_squared_error(global_validation_scores, validation_targets)
        ),
        "sample_weight_distribution": sample_weight_report(
            training_labels, fit.training_sample_weights
        ),
        "scaler_mean": fit.scaler.mean_.astype(float).tolist(),
        "scaler_scale": fit.scaler.scale_.astype(float).tolist(),
        "intercept": np.asarray(fit.model.intercept_, dtype=np.float64).tolist(),
        "coefficients": np.asarray(fit.model.coef_, dtype=np.float64).tolist(),
    }


class RidgeCVAnalyzer:
    """Run the frozen three-fold, 600-configuration Ridge analysis."""

    def __init__(self, dataset: RestrictedAnalysisDataset, class_counts: np.ndarray) -> None:
        _validate_dataset_for_cv(dataset)
        self.dataset = dataset
        self.class_counts = _validate_class_counts(
            np.asarray(class_counts), num_classes=dataset.logits.shape[2]
        )
        _grid_validate()

    def _baseline_row(
        self,
        baseline_id: str,
        predictions: np.ndarray,
        *,
        combination: str,
        weight_vector: Sequence[float] | None,
    ) -> dict[str, Any]:
        report = _metric_report(self.dataset.labels, predictions, self.class_counts)
        fold_metrics = {}
        fold_tail = {}
        for fold in PERMITTED_ANALYSIS_INNER_FOLDS:
            mask = self.dataset.inner_fold_ids == fold
            fold_report = _metric_report(
                self.dataset.labels[mask], predictions[mask], self.class_counts
            )
            fold_metrics[str(fold)] = fold_report["metrics"]
            fold_tail[str(fold)] = fold_report["tail_diagnostics"]
        return {
            "baseline_id": baseline_id,
            "model_type": "fixed_reference",
            "combination": combination,
            "weight_vector": None if weight_vector is None else [float(value) for value in weight_vector],
            "metrics": report["metrics"],
            "tail_diagnostics": report["tail_diagnostics"],
            "fold_metrics": fold_metrics,
            "fold_tail_diagnostics": fold_tail,
        }

    def _build_baselines(self) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
        logits = self.dataset.logits
        uniform_weights = np.full(len(EXPERT_ORDER), 1.0 / len(EXPERT_ORDER))
        baseline_predictions: dict[str, np.ndarray] = {}
        rows: list[dict[str, Any]] = []

        uniform_logits = combine_weighted_logits(logits, uniform_weights).argmax(axis=1)
        baseline_predictions["uniform_logit"] = uniform_logits
        rows.append(
            self._baseline_row(
                "uniform_logit",
                uniform_logits,
                combination="uniform_logit_average",
                weight_vector=uniform_weights,
            )
        )

        uniform_probabilities = _weighted_probability_predictions(logits, uniform_weights)
        baseline_predictions["uniform_probability"] = uniform_probabilities
        rows.append(
            self._baseline_row(
                "uniform_probability",
                uniform_probabilities,
                combination="uniform_probability_average",
                weight_vector=uniform_weights,
            )
        )

        for baseline_id, units in FIXED_REFERENCE_UNITS.items():
            weights = np.asarray(units, dtype=np.float64) / 4.0
            predictions = combine_weighted_logits(logits, weights).argmax(axis=1)
            baseline_predictions[baseline_id] = predictions
            rows.append(
                self._baseline_row(
                    baseline_id,
                    predictions,
                    combination="fixed_logit_reference",
                    weight_vector=weights,
                )
            )

        historical_weights = np.asarray(HISTORICAL_NO_CE_WEIGHTS, dtype=np.float64)
        historical_predictions = combine_weighted_logits(logits, historical_weights).argmax(axis=1)
        baseline_predictions["uniform_without_ce"] = historical_predictions
        rows.append(
            self._baseline_row(
                "uniform_without_ce",
                historical_predictions,
                combination="fixed_logit_historical_reference",
                weight_vector=historical_weights,
            )
        )
        return rows, baseline_predictions

    def run(self) -> dict[str, Any]:
        _grid_validate()
        dataset = self.dataset
        n_samples = dataset.num_samples
        adaptive_scores: dict[str, np.ndarray] = {}
        global_scores: dict[float, np.ndarray] = {
            float(gamma): np.full(
                (n_samples, len(EXPERT_ORDER)), np.nan, dtype=np.float64
            )
            for gamma in GAMMAS
        }
        model_records: list[dict[str, Any]] = []
        fold_model_keys: dict[tuple[int, str, float, float], str] = {}

        held_out_counts = np.zeros(n_samples, dtype=np.int64)
        fold_runs: list[dict[str, Any]] = []

        for validation_fold in PERMITTED_ANALYSIS_INNER_FOLDS:
            validation_mask = dataset.inner_fold_ids == validation_fold
            training_mask = ~validation_mask
            if np.any(training_mask & validation_mask) or not np.all(training_mask | validation_mask):
                raise Task3FError("router-training and router-validation rows overlap")
            held_out_counts[validation_mask] += 1
            train_positions = np.flatnonzero(training_mask)
            validation_positions = np.flatnonzero(validation_mask)
            train_logits = dataset.logits[training_mask]
            train_labels = dataset.labels[training_mask]
            validation_logits = dataset.logits[validation_mask]
            validation_labels = dataset.labels[validation_mask]
            training_targets = compute_contribution_targets(train_logits, train_labels)
            training_features = {
                feature_set: extract_features(train_logits, feature_set)
                for feature_set in FEATURE_SETS
            }
            training_sample_weights = {
                float(gamma): compute_sample_weights(train_labels, gamma)
                for gamma in GAMMAS
            }
            # Validation labels are intentionally held back until after a
            # held-out prediction has been generated below.  They are then
            # used only for contribution-error diagnostics and metrics.
            validation_targets: np.ndarray | None = None
            fold_runs.append(
                {
                    "validation_inner_fold": int(validation_fold),
                    "training_inner_folds": [
                        int(value)
                        for value in PERMITTED_ANALYSIS_INNER_FOLDS
                        if value != validation_fold
                    ],
                    "training_sample_count": int(training_mask.sum()),
                    "validation_sample_count": int(validation_mask.sum()),
                    "training_sample_id_sha256": _sha256_int_array(
                        dataset.sample_indices[training_mask]
                    ),
                    "validation_sample_id_sha256": _sha256_int_array(
                        dataset.sample_indices[validation_mask]
                    ),
                }
            )

            for gamma in GAMMAS:
                global_control = _fit_global_from_precomputed(
                    training_targets,
                    training_sample_weights[float(gamma)],
                    gamma,
                )
                global_scores[float(gamma)][validation_mask] = global_control.predict_scores(
                    int(validation_mask.sum())
                )
                global_training_targets = global_control.training_targets
                global_control_recorded = False

                for feature_set in FEATURE_SETS:
                    for alpha in ALPHAS:
                        model_key = _adaptive_model_key(
                            validation_fold, feature_set, alpha, gamma
                        )
                        fit = _fit_ridge_from_precomputed(
                            training_features[feature_set],
                            training_targets,
                            training_sample_weights[float(gamma)],
                            feature_set=feature_set,
                            alpha=alpha,
                            gamma=gamma,
                        )
                        validation_scores = fit.predict_scores(validation_logits)
                        if not np.isfinite(validation_scores).all():
                            raise Task3FError("held-out Ridge scores are non-finite")
                        if validation_targets is None:
                            # This is the first held-out Ridge prediction for
                            # the fold.  Only now may the validation labels be
                            # used for target-error diagnostics.
                            validation_targets = compute_contribution_targets(
                                validation_logits, validation_labels
                            )
                        if not global_control_recorded:
                            # Preserve the control's fold-specific diagnostic
                            # record after a held-out model prediction exists.
                            fold_runs[-1].setdefault("global_controls", []).append(
                                {
                                    "gamma": float(gamma),
                                    "scores": global_control.scores.tolist(),
                                    "training_mse": _mean_squared_error(
                                        global_control.predict_scores(len(train_logits)),
                                        global_training_targets,
                                    ),
                                    "held_out_mse": _mean_squared_error(
                                        global_control.predict_scores(len(validation_logits)),
                                        validation_targets,
                                    ),
                                    "sample_weight_distribution": sample_weight_report(
                                        train_labels, global_control.training_sample_weights
                                    ),
                                }
                            )
                            global_control_recorded = True
                        adaptive_scores[model_key] = np.full(
                            (n_samples, len(EXPERT_ORDER)), np.nan, dtype=np.float64
                        )
                        adaptive_scores[model_key][validation_mask] = validation_scores
                        fold_model_keys[(validation_fold, feature_set, float(alpha), float(gamma))] = model_key
                        record = _fit_record(
                            model_key=model_key,
                            validation_fold=validation_fold,
                            feature_set=feature_set,
                            alpha=alpha,
                            gamma=gamma,
                            train_indices=train_positions,
                            validation_indices=validation_positions,
                            train_sample_ids=dataset.sample_indices[training_mask],
                            validation_sample_ids=dataset.sample_indices[validation_mask],
                            training_labels=train_labels,
                            fit=fit,
                            validation_scores=validation_scores,
                            validation_targets=validation_targets,
                            global_control=global_control,
                        )
                        model_records.append(record)

        if not np.array_equal(held_out_counts, np.ones(n_samples, dtype=np.int64)):
            raise Task3FError("every permitted sample must receive exactly one held-out prediction")
        for scores in adaptive_scores.values():
            valid_rows = ~np.isnan(scores).any(axis=1)
            if not valid_rows.any() or not np.isfinite(scores[valid_rows]).all():
                raise Task3FError("adaptive held-out score assembly is incomplete")
        if any(np.isnan(scores).any() for scores in global_scores.values()):
            raise Task3FError("global held-out score assembly is incomplete")

        baseline_rows, baseline_predictions = self._build_baselines()
        adaptive_rows: list[dict[str, Any]] = []
        global_rows: list[dict[str, Any]] = []
        adaptive_prediction_arrays: list[np.ndarray] = []
        adaptive_weight_arrays: list[np.ndarray] = []
        adaptive_ids: list[str] = []
        global_prediction_arrays: list[np.ndarray] = []
        global_weight_arrays: list[np.ndarray] = []
        global_ids: list[str] = []

        model_record_by_key = {record["model_key"]: record for record in model_records}
        for feature_set in FEATURE_SETS:
            for alpha in ALPHAS:
                for gamma in GAMMAS:
                    scores = np.zeros((n_samples, len(EXPERT_ORDER)), dtype=np.float64)
                    model_keys = []
                    for validation_fold in PERMITTED_ANALYSIS_INNER_FOLDS:
                        key = fold_model_keys[(validation_fold, feature_set, float(alpha), float(gamma))]
                        scores[dataset.inner_fold_ids == validation_fold] = adaptive_scores[key][
                            dataset.inner_fold_ids == validation_fold
                        ]
                        model_keys.append(key)
                    for temperature in TEMPERATURES:
                        for shrinkage in SHRINKAGES:
                            configuration_id = _adaptive_configuration_id(
                                feature_set, alpha, gamma, temperature, shrinkage
                            )
                            if shrinkage == 0.0:
                                # All temperatures, feature sets, alphas, and
                                # gammas are exactly the same uniform ensemble
                                # at lambda=0.  Reuse that one calculation.
                                weights = np.full(
                                    (n_samples, len(EXPERT_ORDER)),
                                    1.0 / len(EXPERT_ORDER),
                                    dtype=np.float64,
                                )
                                global_weights = weights.copy()
                                predictions = baseline_predictions["uniform_logit"]
                            else:
                                weights = scores_to_weights(scores, temperature, shrinkage)
                                global_weights = scores_to_weights(
                                    global_scores[float(gamma)], temperature, shrinkage
                                )
                                predictions = combine_weighted_logits(
                                    dataset.logits, weights
                                ).argmax(axis=1)
                            report = _metric_report(dataset.labels, predictions, self.class_counts)
                            fold_metrics = {}
                            fold_tail = {}
                            for fold in PERMITTED_ANALYSIS_INNER_FOLDS:
                                mask = dataset.inner_fold_ids == fold
                                fold_report = _metric_report(
                                    dataset.labels[mask], predictions[mask], self.class_counts
                                )
                                fold_metrics[str(fold)] = fold_report["metrics"]
                                fold_tail[str(fold)] = fold_report["tail_diagnostics"]
                            fit_records = [model_record_by_key[key] for key in model_keys]
                            train_count = sum(
                                record["router_training_sample_count"] for record in fit_records
                            )
                            contribution = {
                                "training_mse": float(
                                    sum(
                                        record["training_mse"] * record["router_training_sample_count"]
                                        for record in fit_records
                                    )
                                    / train_count
                                ),
                                "held_out_mse": float(
                                    np.mean([record["held_out_mse"] for record in fit_records])
                                ),
                                "global_control_training_mse": float(
                                    sum(
                                        record["global_control_training_mse"]
                                        * record["router_training_sample_count"]
                                        for record in fit_records
                                    )
                                    / train_count
                                ),
                                "global_control_held_out_mse": float(
                                    np.mean(
                                        [record["global_control_held_out_mse"] for record in fit_records]
                                    )
                                ),
                                "adaptive_improves_over_global": bool(
                                    np.mean([record["held_out_mse"] for record in fit_records])
                                    < np.mean(
                                        [record["global_control_held_out_mse"] for record in fit_records]
                                    )
                                ),
                                "model_fit_keys": model_keys,
                            }
                            adaptive_rows.append(
                                {
                                    "configuration_id": configuration_id,
                                    "model_type": "adaptive_ridge",
                                    "feature_set": feature_set,
                                    "alpha": float(alpha),
                                    "gamma": float(gamma),
                                    "temperature": float(temperature),
                                    "shrinkage": float(shrinkage),
                                    "metrics": report["metrics"],
                                    "tail_diagnostics": report["tail_diagnostics"],
                                    "fold_metrics": fold_metrics,
                                    "fold_tail_diagnostics": fold_tail,
                                    "contribution_prediction": contribution,
                                    "weight_diagnostics": _weight_diagnostics(weights, global_weights),
                                    "prediction_is_uniform_configuration": bool(shrinkage == 0.0),
                                    "equivalent_output_reference": (
                                        "uniform_logit" if shrinkage == 0.0 else None
                                    ),
                                    "prediction_sha256": _sha256_array(predictions.astype(np.int16)),
                                    "weight_sha256": _sha256_array(weights.astype(np.float32)),
                                }
                            )
                            adaptive_ids.append(configuration_id)
                            adaptive_prediction_arrays.append(predictions.astype(np.int16))
                            adaptive_weight_arrays.append(weights.astype(np.float32))

        for gamma in GAMMAS:
            global_score_matrix = global_scores[float(gamma)]
            for temperature in TEMPERATURES:
                for shrinkage in SHRINKAGES:
                    configuration_id = _global_configuration_id(gamma, temperature, shrinkage)
                    if shrinkage == 0.0:
                        weights = np.full(
                            (n_samples, len(EXPERT_ORDER)),
                            1.0 / len(EXPERT_ORDER),
                            dtype=np.float64,
                        )
                        predictions = baseline_predictions["uniform_logit"]
                    else:
                        weights = scores_to_weights(global_score_matrix, temperature, shrinkage)
                        predictions = combine_weighted_logits(
                            dataset.logits, weights
                        ).argmax(axis=1)
                    report = _metric_report(dataset.labels, predictions, self.class_counts)
                    fold_metrics = {}
                    fold_tail = {}
                    for fold in PERMITTED_ANALYSIS_INNER_FOLDS:
                        mask = dataset.inner_fold_ids == fold
                        fold_report = _metric_report(
                            dataset.labels[mask], predictions[mask], self.class_counts
                        )
                        fold_metrics[str(fold)] = fold_report["metrics"]
                        fold_tail[str(fold)] = fold_report["tail_diagnostics"]
                    control_records = [
                        fold["global_controls"]
                        for fold in fold_runs
                    ]
                    per_gamma_records = [
                        next(record for record in records if record["gamma"] == float(gamma))
                        for records in control_records
                    ]
                    global_train_mse = float(
                        np.mean([record["training_mse"] for record in per_gamma_records])
                    )
                    global_val_mse = float(
                        np.mean([record["held_out_mse"] for record in per_gamma_records])
                    )
                    global_rows.append(
                        {
                            "configuration_id": configuration_id,
                            "model_type": "global_score_control",
                            "feature_set": None,
                            "alpha": None,
                            "gamma": float(gamma),
                            "temperature": float(temperature),
                            "shrinkage": float(shrinkage),
                            "metrics": report["metrics"],
                            "tail_diagnostics": report["tail_diagnostics"],
                            "fold_metrics": fold_metrics,
                            "fold_tail_diagnostics": fold_tail,
                            "contribution_prediction": {
                                "training_mse": global_train_mse,
                                "held_out_mse": global_val_mse,
                                "model_fit_keys": [
                                    f"global_fold{fold}_gamma{_token(gamma)}"
                                    for fold in PERMITTED_ANALYSIS_INNER_FOLDS
                                ],
                            },
                            "weight_diagnostics": _weight_diagnostics(weights, weights),
                            "prediction_is_uniform_configuration": bool(shrinkage == 0.0),
                            "equivalent_output_reference": (
                                "uniform_logit" if shrinkage == 0.0 else None
                            ),
                            "prediction_sha256": _sha256_array(predictions.astype(np.int16)),
                            "weight_sha256": _sha256_array(weights.astype(np.float32)),
                        }
                    )
                    global_ids.append(configuration_id)
                    global_prediction_arrays.append(predictions.astype(np.int16))
                    global_weight_arrays.append(weights.astype(np.float32))

        global_model_records = []
        for fold in fold_runs:
            for control in fold.get("global_controls", []):
                global_model_records.append(
                    {
                        "model_key": f"global_fold{fold['validation_inner_fold']}_gamma{_token(control['gamma'])}",
                        "model_type": "global_score_control",
                        "validation_inner_fold": fold["validation_inner_fold"],
                        "gamma": control["gamma"],
                        "training_mse": control["training_mse"],
                        "held_out_mse": control["held_out_mse"],
                        "sample_weight_distribution": control["sample_weight_distribution"],
                        "scores": control["scores"],
                    }
                )

        def hash_groups(ids: Sequence[str], arrays: Sequence[np.ndarray]) -> dict[str, list[str]]:
            groups: dict[str, list[str]] = {}
            for identifier, array in zip(ids, arrays):
                groups.setdefault(_sha256_array(array), []).append(str(identifier))
            return {key: value for key, value in groups.items() if len(value) > 1}

        return {
            "cv_contract": {
                "validation_inner_folds": list(PERMITTED_ANALYSIS_INNER_FOLDS),
                "training_folds_by_validation_fold": {
                    str(fold): [
                        int(value)
                        for value in PERMITTED_ANALYSIS_INNER_FOLDS
                        if value != fold
                    ]
                    for fold in PERMITTED_ANALYSIS_INNER_FOLDS
                },
                "every_sample_held_out_exactly_once": True,
                "router_training_and_validation_disjoint": True,
                "inner_fold_zero_used": False,
                "outer_evaluation_used": False,
            },
            "fold_runs": fold_runs,
            "held_out_sample_indices": dataset.sample_indices.copy(),
            "held_out_inner_fold_ids": dataset.inner_fold_ids.copy(),
            "held_out_labels": dataset.labels.copy(),
            "model_fits": model_records,
            "global_model_fits": global_model_records,
            "adaptive_results": adaptive_rows,
            "global_results": global_rows,
            "baseline_results": baseline_rows,
            "baseline_predictions": baseline_predictions,
            "adaptive_configuration_ids": adaptive_ids,
            "adaptive_predictions": np.asarray(adaptive_prediction_arrays, dtype=np.int16),
            "adaptive_weights": np.asarray(adaptive_weight_arrays, dtype=np.float32),
            "global_configuration_ids": global_ids,
            "global_predictions": np.asarray(global_prediction_arrays, dtype=np.int16),
            "global_weights": np.asarray(global_weight_arrays, dtype=np.float32),
            "output_equivalence": {
                "adaptive_prediction_hash_groups": hash_groups(
                    adaptive_ids, adaptive_prediction_arrays
                ),
                "adaptive_weight_hash_groups": hash_groups(
                    adaptive_ids, adaptive_weight_arrays
                ),
                "global_prediction_hash_groups": hash_groups(
                    global_ids, global_prediction_arrays
                ),
                "global_weight_hash_groups": hash_groups(
                    global_ids, global_weight_arrays
                ),
                "lambda_zero_reference": "uniform_logit",
            },
            "held_out_adaptive_scores": np.asarray(
                [adaptive_scores[key] for key in sorted(adaptive_scores)], dtype=np.float32
            ),
            "held_out_adaptive_score_model_ids": np.asarray(
                sorted(adaptive_scores), dtype="U128"
            ),
            "global_scores": np.asarray(
                [global_scores[float(gamma)] for gamma in GAMMAS], dtype=np.float32
            ),
            "global_score_gammas": np.asarray(GAMMAS, dtype=np.float64),
        }


def _sha256_file(path: Path) -> str:
    return sha256_file(path, error_type=Task3FError, description="file")


def _repository_relative(path: Path, project_root: Path) -> str:
    return repository_relative(path, project_root)


def _git_commit(project_root: Path) -> str:
    return git_commit(project_root, error_type=Task3FError)


def _load_reference_uniform_metrics(path: Path) -> dict[str, Any]:
    payload = load_json_object(
        path,
        name="Task 3C diagnostics",
        error_type=Task3FError,
    )
    primary = payload.get("primary_router_fit_partition")
    if not isinstance(primary, Mapping):
        raise Task3FError("Task 3C diagnostics lack the primary router-fit partition")
    expected = {
        "schema_version": "task3c_diagnostics.v1",
        "dataset": DATASET_NAME,
        "imbalance_ratio": IMBALANCE_RATIO,
        "training_seed": TRAINING_SEED,
        "fold_generation_seed": FOLD_GENERATION_SEED,
        "outer_fold_id": OUTER_FOLD,
        "expert_order": list(EXPERT_ORDER),
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise Task3FError(f"Task 3C diagnostics.{key} is incompatible")
    if primary.get("sample_count") != ANALYSIS_SAMPLE_COUNT:
        raise Task3FError("Task 3C diagnostics primary sample count is incompatible")
    if primary.get("inner_fold_ids") != list(PERMITTED_ANALYSIS_INNER_FOLDS):
        raise Task3FError("Task 3C diagnostics primary folds are incompatible")
    uniform = primary.get("uniform_logit_ensemble")
    if not isinstance(uniform, Mapping) or uniform.get("combination") != "uniform logit average":
        raise Task3FError("Task 3C diagnostics lack the stored uniform baseline")
    metrics = uniform.get("metrics")
    if not isinstance(metrics, Mapping):
        raise Task3FError("stored Task 3C uniform metrics are malformed")
    return dict(metrics)


def _validate_source_files(
    source_files: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    validated = {}
    for name, details in source_files.items():
        path = Path(details["path"])
        if not path.exists():
            raise Task3FError(f"source artifact disappeared during validation: {path}")
        actual_hash = _sha256_file(path)
        if actual_hash != str(details["sha256"]):
            raise Task3FError(f"source artifact changed during Task 3F-A setup: {path}")
        validated[name] = {"path": path, "sha256": actual_hash}
    return validated


def build_experiment_config(
    *,
    project_root: Path,
    source_files: Mapping[str, Mapping[str, Any]],
    reference_diagnostics: Path,
    manager: NestedOOFFoldManager,
    dataset: RestrictedAnalysisDataset,
) -> dict[str, Any]:
    groups = compute_class_groups(np.asarray(manager.canonical_class_counts, dtype=np.int64))
    return {
        "task_identifier": TASK3F_TASK_IDENTIFIER,
        "schema_version": TASK3F_SCHEMA_VERSION,
        "source_git_commit": _git_commit(project_root),
        "input_artifacts": {
            name: {
                "path": _repository_relative(Path(details["path"]), project_root),
                "sha256": str(details["sha256"]),
            }
            for name, details in source_files.items()
        }
        | {
            "task3c_diagnostics": {
                "path": _repository_relative(reference_diagnostics, project_root),
                "sha256": _sha256_file(reference_diagnostics),
            }
        },
        "dataset": DATASET_NAME,
        "imbalance_ratio": IMBALANCE_RATIO,
        "canonical_training_population_size": CANONICAL_POPULATION_SIZE,
        "full_aligned_sample_count": FULL_ALIGNED_SAMPLE_COUNT,
        "analyzed_sample_count": dataset.num_samples,
        "training_seed": TRAINING_SEED,
        "fold_generation_seed": FOLD_GENERATION_SEED,
        "outer_fold": OUTER_FOLD,
        "outer_evaluation_population_size": OUTER_EVALUATION_SIZE,
        "permitted_analysis_inner_folds": list(PERMITTED_ANALYSIS_INNER_FOLDS),
        "reserved_router_selection_inner_folds": list(RESERVED_ROUTER_SELECTION_INNER_FOLDS),
        "expert_order": list(EXPERT_ORDER),
        "number_of_classes": NUM_CLASSES,
        "canonical_training_index_sha256": manager.canonical_training_index_sha256,
        "target": {
            "name": "marginal_uniform_ensemble_true_class_log_probability",
            "formula": "(z_e_y-zbar_y)-sum_c softmax(zbar)_c*(z_e_c-zbar_c)",
            "target_dimension": len(EXPERT_ORDER),
            "zero_sum_verified": True,
            "oracle_weights_used": False,
        },
        "features": {
            "confidence_only": {
                "name": FEATURE_SET_CONFIDENCE,
                "dimension": 4,
                "features": list(FEATURE_NAMES[FEATURE_SET_CONFIDENCE]),
            },
            "full_13": {
                "name": FEATURE_SET_FULL,
                "dimension": 13,
                "features": list(FEATURE_NAMES[FEATURE_SET_FULL]),
            },
            "source": "original OOF logits only; labels and fold IDs excluded",
            "standardization": "StandardScaler fit separately on each router-training fold",
        },
        "ridge": {
            "model": "sklearn.linear_model.Ridge",
            "multi_output": True,
            "alphas": list(ALPHAS),
            "fit_intercept": True,
            "intercept_penalized": False,
            "solver": RIDGE_SOLVER,
            "solver_random_state": None,
            "solver_tolerance": RIDGE_TOLERANCE,
        },
        "class_weighting": {
            "gammas": list(GAMMAS),
            "formula": "min(5,(n_max/n_y)**gamma)",
            "cap": WEIGHT_CAP,
            "normalization": "mean one within each router-training fold",
            "labels_source": "router-training rows only",
        },
        "score_to_weight": {
            "temperatures": list(TEMPERATURES),
            "shrinkages": list(SHRINKAGES),
            "formula": "w=(1-lambda)/4+lambda*softmax(score/T)",
            "combination": "weighted original logits",
        },
        "grid_size": {
            "adaptive_ridge_configurations": len(FEATURE_SETS) * len(ALPHAS) * len(GAMMAS) * len(TEMPERATURES) * len(SHRINKAGES),
            "ridge_fits_per_validation_fold": len(FEATURE_SETS) * len(ALPHAS) * len(GAMMAS),
            "ridge_fits_total": len(PERMITTED_ANALYSIS_INNER_FOLDS) * len(FEATURE_SETS) * len(ALPHAS) * len(GAMMAS),
            "global_control_configurations": len(GAMMAS) * len(TEMPERATURES) * len(SHRINKAGES),
        },
        "class_group_definitions": {
            "source": "scripts.base_trainer.compute_class_groups",
            "head": [int(value) for value in groups["head"].tolist()],
            "medium": [int(value) for value in groups["medium"].tolist()],
            "tail": [int(value) for value in groups["tail"].tolist()],
        },
        "metric_definitions": {
            "ordinary_accuracy": "fraction of samples with predicted class equal to label",
            "balanced_accuracy": "mean per-class recall over the pooled held-out predictions",
            "head_accuracy": "macro recall over canonical Head classes",
            "medium_accuracy": "macro recall over canonical Medium classes",
            "tail_accuracy": "macro recall over canonical Tail classes",
        },
        "data_restrictions": {
            "inner_fold_zero_used": False,
            "reserved_outer_evaluation_used": False,
            "original_cifar_test_used": False,
            "expert_training_or_checkpoints_modified": False,
            "source_oof_logits_modified": False,
            "independent_validation_claim": False,
        },
        "scientific_limitations": [
            "This is exploratory development-data cross-validation on one seed and one outer fold.",
            "Existing OOF experts have overlapping training populations across router folds.",
            "The permitted population contains only 183 Tail images.",
            "Inner fold 0 was inspected descriptively by Task 3C before this study.",
            "The result does not establish generalization to outer-fold or full-data experts.",
        ],
    }


def _jsonable_analysis(result: Mapping[str, Any]) -> dict[str, Any]:
    """Remove NumPy arrays from the JSON result while retaining all records."""
    return {
        "task_identifier": TASK3F_TASK_IDENTIFIER,
        "schema_version": TASK3F_RESULTS_SCHEMA_VERSION,
        "cv_contract": dict(result["cv_contract"]),
        "fold_runs": list(result["fold_runs"]),
        "model_fits": list(result["model_fits"]),
        "global_model_fits": list(result["global_model_fits"]),
        "output_equivalence": dict(result["output_equivalence"]),
        "baseline_results": list(result["baseline_results"]),
        "adaptive_results": list(result["adaptive_results"]),
        "global_results": list(result["global_results"]),
        "configuration_counts": {
            "baseline": len(result["baseline_results"]),
            "adaptive_ridge": len(result["adaptive_results"]),
            "global_score_control": len(result["global_results"]),
        },
    }


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    write_json_once(path, payload, error_type=Task3FError)


def _write_npz_once(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    write_npz_once(path, arrays, error_type=Task3FError)


def _render_summary(
    experiment_config: Mapping[str, Any],
    results: Mapping[str, Any],
    baseline_verification: Mapping[str, Any],
) -> str:
    def pct(value: Any) -> str:
        return f"{100.0 * float(value):.4f}%"

    def metric_line(row: Mapping[str, Any]) -> str:
        metrics = row["metrics"]
        return (
            f"accuracy={pct(metrics['ordinary_accuracy'])}, "
            f"BA={pct(metrics['balanced_accuracy'])}, "
            f"Head={pct(metrics['head_accuracy'])}, "
            f"Medium={pct(metrics['medium_accuracy'])}, "
            f"Tail={pct(metrics['tail_accuracy'])}"
        )

    baselines = {row["baseline_id"]: row for row in results["baseline_results"]}
    adaptive = results["adaptive_results"]
    joint = [
        row
        for row in adaptive
        if row["metrics"]["balanced_accuracy"] > baselines["uniform_logit"]["metrics"]["balanced_accuracy"]
        and row["metrics"]["tail_accuracy"] > baselines["uniform_logit"]["metrics"]["tail_accuracy"]
    ]
    best_ba = max(adaptive, key=lambda row: row["metrics"]["balanced_accuracy"])
    best_tail = max(adaptive, key=lambda row: row["metrics"]["tail_accuracy"])
    best_joint = max(
        joint,
        key=lambda row: (
            row["metrics"]["balanced_accuracy"] + row["metrics"]["tail_accuracy"]
        ),
        default=None,
    )
    lines = [
        "# Task 3F-A — Ridge-Based Predictable Routing Feasibility",
        "",
        "Exploratory three-fold cross-validation of multi-output Ridge contribution prediction on the permitted OOF development population.",
        "",
        "## Protocol and safeguards",
        "",
        f"- Source commit: `{experiment_config['source_git_commit']}`",
        f"- Population: {experiment_config['analyzed_sample_count']} images; outer fold {experiment_config['outer_fold']}; inner folds {experiment_config['permitted_analysis_inner_folds']}",
        "- Router folds: train on (2,3), (1,3), and (1,2); validate on 1, 2, and 3 respectively.",
        "- Inner fold 0, the reserved outer evaluation population, and the original CIFAR-100 test set were not used.",
        "- Original OOF logits, expert order, checkpoints, and training pipeline were not modified.",
        "- Results are exploratory development measurements; no independent validation claim is made.",
        "",
        "## Uniform baseline reproduction",
        "",
        f"Verification match: **{baseline_verification['matches']}** at tolerance `{baseline_verification['tolerance']}`.",
        "",
        "| Method | Accuracy | BA | Head | Medium | Tail |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for baseline_id in (
        "uniform_logit",
        "uniform_probability",
        "fixed_006",
        "fixed_007",
        "fixed_010",
        "fixed_011",
        "uniform_without_ce",
    ):
        row = baselines[baseline_id]
        lines.append(f"| {baseline_id} | {metric_line(row).replace(', ', ' | ').replace('accuracy=', '').replace('BA=', '').replace('Head=', '').replace('Medium=', '').replace('Tail=', '')} |")
    lines.extend(
        [
            "",
            "## Ridge results",
            "",
            f"- Adaptive configurations evaluated: **{len(adaptive)}**; fitted Ridge models: **{len([row for row in results['model_fits'] if row['model_type'] == 'adaptive_ridge'])}**.",
            f"- Configurations improving both BA and Tail over uniform logit: **{len(joint)}**.",
            f"- Global score-control configurations evaluated: **{len(results['global_results'])}**.",
            "",
            "| Highlight | Configuration | BA | Tail | Held-out contribution MSE | Global-control MSE |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for label, row in (
        ("Best BA", best_ba),
        ("Best Tail", best_tail),
        ("Best BA+Tail sum among joint improvements", best_joint),
    ):
        if row is None:
            lines.append(f"| {label} | none | n/a | n/a | n/a | n/a |")
        else:
            contribution = row["contribution_prediction"]
            lines.append(
                f"| {label} | `{row['configuration_id']}` | {pct(row['metrics']['balanced_accuracy'])} | "
                f"{pct(row['metrics']['tail_accuracy'])} | {contribution['held_out_mse']:.6g} | "
                f"{contribution['global_control_held_out_mse']:.6g} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "All 600 adaptive rows, all 60 global-control rows, all 90 fitted-model diagnostics, fold-level metrics, and weight diagnostics are in `ridge_results.json`.",
            "The global control uses the same training folds, targets, class weighting, temperatures, and shrinkage values but receives no image-dependent features.",
            "Any apparent development improvement remains subject to the documented Tail sparsity, overlapping OOF expert-training populations, prior development exposure, and untested generalization to outer/full-data experts.",
            "",
        ]
    )
    return "\n".join(lines)


def run_task3f_ridge(
    *,
    data_root: str | Path = "./data",
    oof_directory: str | Path = "artifacts/oof/task3c_oof",
    reference_diagnostics: str | Path | None = None,
    output_directory: str | Path = "artifacts/oof/task3f_ridge",
    project_root: str | Path | None = None,
) -> dict[str, Path]:
    """Execute Task 3F-A and persist write-once machine-readable artifacts."""
    project_root_path = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    source_directory = Path(oof_directory).resolve()
    reference_path = Path(reference_diagnostics or source_directory / "diagnostics.json").resolve()
    output_path = Path(output_directory).resolve()
    manager = NestedOOFFoldManager.from_canonical_training_data(
        data_root,
        seed=FOLD_GENERATION_SEED,
        outer_folds=5,
        inner_folds=4,
        expert_order=EXPERT_ORDER,
    )
    dataset, source_files = load_restricted_analysis_dataset(source_directory, manager)
    source_files = _validate_source_files(source_files)
    if not reference_path.exists():
        raise Task3FError(f"reference diagnostics do not exist: {reference_path}")
    reference_hash_before = _sha256_file(reference_path)
    stored_uniform_metrics = _load_reference_uniform_metrics(reference_path)

    analysis = RidgeCVAnalyzer(
        dataset,
        np.asarray(manager.canonical_class_counts, dtype=np.int64),
    ).run()
    uniform_row = next(
        row for row in analysis["baseline_results"] if row["baseline_id"] == "uniform_logit"
    )
    baseline_verification = verify_uniform_baseline(
        {
            "ordinary_accuracy": uniform_row["metrics"]["ordinary_accuracy"],
            "balanced_accuracy": uniform_row["metrics"]["balanced_accuracy"],
            "head_accuracy": uniform_row["metrics"]["head_accuracy"],
            "medium_accuracy": uniform_row["metrics"]["medium_accuracy"],
            "tail_accuracy": uniform_row["metrics"]["tail_accuracy"],
        },
        stored_uniform_metrics,
        tolerance=UNIFORM_BASELINE_TOLERANCE,
    )
    if _sha256_file(reference_path) != reference_hash_before:
        raise Task3FError("reference diagnostics changed during Task 3F-A")

    config = build_experiment_config(
        project_root=project_root_path,
        source_files=source_files,
        reference_diagnostics=reference_path,
        manager=manager,
        dataset=dataset,
    )
    config["uniform_baseline_verification"] = baseline_verification
    json_results = _jsonable_analysis(analysis)
    json_results["uniform_baseline_verification"] = baseline_verification
    arrays = {
        "sample_indices": np.asarray(analysis["held_out_sample_indices"], dtype=np.int64),
        "inner_fold_ids": np.asarray(analysis["held_out_inner_fold_ids"], dtype=np.int64),
        "labels": np.asarray(analysis["held_out_labels"], dtype=np.int64),
        "baseline_ids": np.asarray(list(analysis["baseline_predictions"]), dtype="U64"),
        "baseline_predictions": np.asarray(
            [analysis["baseline_predictions"][key] for key in analysis["baseline_predictions"]],
            dtype=np.int16,
        ),
        "adaptive_configuration_ids": np.asarray(analysis["adaptive_configuration_ids"], dtype="U160"),
        "adaptive_predictions": analysis["adaptive_predictions"],
        "adaptive_weights": analysis["adaptive_weights"],
        "global_configuration_ids": np.asarray(analysis["global_configuration_ids"], dtype="U128"),
        "global_predictions": analysis["global_predictions"],
        "global_weights": analysis["global_weights"],
    }
    router_scores = {
        "sample_indices": arrays["sample_indices"],
        "inner_fold_ids": arrays["inner_fold_ids"],
        "labels": arrays["labels"],
        "model_ids": analysis["held_out_adaptive_score_model_ids"],
        "held_out_scores": analysis["held_out_adaptive_scores"],
        "global_score_gammas": analysis["global_score_gammas"],
        "global_scores": analysis["global_scores"],
    }
    fold_assignments = {
        "schema_version": "task3f_fold_assignments.v1",
        "outer_fold": OUTER_FOLD,
        "permitted_inner_fold_ids": list(PERMITTED_ANALYSIS_INNER_FOLDS),
        "every_sample_held_out_exactly_once": True,
        "folds": {
            str(fold): {
                "validation_sample_ids": [
                    int(value)
                    for value in dataset.sample_indices[dataset.inner_fold_ids == fold]
                ],
                "training_sample_ids": [
                    int(value)
                    for value in dataset.sample_indices[dataset.inner_fold_ids != fold]
                ],
            }
            for fold in PERMITTED_ANALYSIS_INNER_FOLDS
        },
    }
    summary = _render_summary(config, json_results, baseline_verification)
    output_files = {
        output_path / "experiment_config.json": config,
        output_path / "ridge_results.json": json_results,
        output_path / "fold_assignments.json": fold_assignments,
    }
    for path, payload in output_files.items():
        _write_json_once(path, payload)
    _write_npz_once(output_path / "held_out_predictions.npz", arrays)
    _write_npz_once(output_path / "router_scores.npz", router_scores)
    summary_path = output_path / "summary.md"
    if summary_path.exists():
        if summary_path.read_text() != summary:
            raise Task3FError(f"refusing to overwrite an incompatible output: {summary_path}")
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary)
    _validate_source_files(source_files)
    return {
        "experiment_config": output_path / "experiment_config.json",
        "ridge_results": output_path / "ridge_results.json",
        "fold_assignments": output_path / "fold_assignments.json",
        "held_out_predictions": output_path / "held_out_predictions.npz",
        "router_scores": output_path / "router_scores.npz",
        "summary": summary_path,
    }


__all__ = [
    "ALPHAS",
    "FEATURE_NAMES",
    "FEATURE_SET_CONFIDENCE",
    "FEATURE_SET_FULL",
    "FEATURE_SETS",
    "GAMMAS",
    "PERMITTED_ANALYSIS_INNER_FOLDS",
    "RIDGE_SOLVER",
    "SHRINKAGES",
    "TEMPERATURES",
    "Task3FError",
    "GlobalScoreControl",
    "RidgeCVAnalyzer",
    "RidgeRouterFit",
    "classification_metrics",
    "combine_weighted_logits",
    "compute_contribution_targets",
    "compute_sample_weights",
    "extract_features",
    "fit_global_score_control",
    "fit_ridge_router",
    "run_task3f_ridge",
    "scores_to_weights",
    "sample_weight_report",
]
