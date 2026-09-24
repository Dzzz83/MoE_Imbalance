"""Fitted, inference-only Ridge maps for the Ridge/Sinkhorn study.

Targets and sample weights are constructed by the caller from router-fit rows.
The fitted object needs only expert logits at prediction time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from scripts.task3f_ridge import (
    EXPERT_ORDER,
    FEATURE_NAMES,
    RIDGE_SOLVER,
    RIDGE_TOLERANCE,
    _as_float_array,
    _canonical_feature_set,
    _validate_alpha,
    extract_features,
)


@dataclass(frozen=True)
class ResidualRidgeFit:
    """A multi-output Ridge fit with training-only feature standardization."""

    feature_set: str
    alpha: float
    gamma: float
    scaler: StandardScaler
    model: Ridge

    def predict_residual(self, logits: np.ndarray) -> np.ndarray:
        features = extract_features(logits, self.feature_set)
        values = np.asarray(self.model.predict(self.scaler.transform(features)), dtype=np.float64)
        if values.shape != (len(features), len(EXPERT_ORDER)) or not np.isfinite(values).all():
            raise ValueError("Ridge predicted invalid expert residuals")
        return values


def fit_residual_ridge(
    logits: np.ndarray,
    targets: np.ndarray,
    sample_weights: np.ndarray,
    *,
    feature_set: str,
    alpha: float,
    gamma: float,
) -> ResidualRidgeFit:
    """Fit the frozen Task 3F-A scaler/Ridge convention to oracle residuals."""
    canonical = _canonical_feature_set(feature_set)
    alpha = _validate_alpha(alpha)
    features = extract_features(logits, canonical)
    target = _as_float_array(targets, name="oracle residual targets")
    weights = _as_float_array(sample_weights, name="fit sample weights")
    if features.shape[1] != len(FEATURE_NAMES[canonical]):
        raise ValueError("feature count disagrees with the frozen representation")
    if target.shape != (len(features), len(EXPERT_ORDER)):
        raise ValueError("oracle residual targets have the wrong shape")
    if weights.shape != (len(features),) or np.any(weights <= 0):
        raise ValueError("fit sample weights must be positive and aligned")
    if not np.isclose(weights.mean(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("fit sample weights must have mean one")
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaled = scaler.fit_transform(features)
    model = Ridge(
        alpha=alpha,
        fit_intercept=True,
        solver=RIDGE_SOLVER,
        tol=RIDGE_TOLERANCE,
        random_state=None,
    )
    model.fit(scaled, target, sample_weight=weights)
    predicted = model.predict(scaled)
    if not all(np.isfinite(x).all() for x in (scaled, predicted, model.coef_, model.intercept_)):
        raise ValueError("Ridge fit produced non-finite parameters or predictions")
    return ResidualRidgeFit(canonical, alpha, float(gamma), scaler, model)
