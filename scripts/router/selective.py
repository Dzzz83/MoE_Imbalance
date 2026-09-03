"""
Selective routing — abstain on low-confidence samples and fall back to
a safe default (uniform averaging or abstention).

Allows the router to "pass" on hard samples where confidence is low,
improving reliability at the cost of coverage.
"""

from __future__ import annotations

from typing import Self

import numpy as np

from scripts.router.base import BaseRouter
from scripts.utils.features import softmax


class SelectiveRouter(BaseRouter):
    """Route selectively: only use learned routing when confidence is high.

    For samples where the router's confidence is below a threshold,
    falls back to uniform averaging (or another safe default).

    Args:
        base_router: The underlying router to use for confident samples.
        confidence_threshold: Minimum routing confidence to use learned routing.
        fallback_router: Router to use for low-confidence samples.
                         If None, uses uniform averaging.
    """

    def __init__(
        self,
        base_router: BaseRouter,
        expert_names: list[str] | None = None,
        confidence_threshold: float = 0.5,
        fallback_router: BaseRouter | None = None,
    ):
        if expert_names is None:
            expert_names = base_router.expert_names
        super().__init__(expert_names)
        self.base_router = base_router
        self.confidence_threshold = confidence_threshold
        self.fallback_router = fallback_router

    def train(
        self,
        val_logits: np.ndarray,
        val_labels: np.ndarray,
        val_features: dict | None = None,
    ) -> Self:
        """Train the base router and optionally tune threshold."""
        self.base_router.train(val_logits, val_labels, val_features)
        if self.fallback_router is not None:
            self.fallback_router.train(val_logits, val_labels, val_features)

        # Tune threshold on validation set to maximize coverage at target BA
        self._tune_threshold(val_logits, val_labels, val_features)
        self._is_trained = True
        return self

    def predict(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Route confidently or fall back. Return expert index."""
        # Get base router predictions and confidence
        base_expert = self.base_router.predict(logits, features)
        base_weights = self.base_router.predict_proba(logits, features)
        routing_confidence = base_weights.max(axis=1)

        # Get fallback expert
        if self.fallback_router is not None:
            fallback_expert = self.fallback_router.predict(logits, features)
        else:
            fallback_expert = np.zeros(logits.shape[0], dtype=np.int64)

        # Selective: use base router where confident, fallback elsewhere
        confident = routing_confidence >= self.confidence_threshold
        return np.where(confident, base_expert, fallback_expert)

    def predict_class(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Return class prediction with selective fallback."""
        base_preds = self.base_router.predict_class(logits, features)
        base_weights = self.base_router.predict_proba(logits, features)
        routing_confidence = base_weights.max(axis=1)

        if self.fallback_router is not None:
            fallback_preds = self.fallback_router.predict_class(logits, features)
        else:
            avg_logits = logits.mean(axis=1)
            fallback_preds = avg_logits.argmax(axis=1)

        confident = routing_confidence >= self.confidence_threshold
        return np.where(confident, base_preds, fallback_preds)

    def predict_proba(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Return routing weights, with low-confidence samples masked."""
        base_weights = self.base_router.predict_proba(logits, features)
        routing_confidence = base_weights.max(axis=1)
        confident = routing_confidence >= self.confidence_threshold

        # For non-confident samples, return uniform weights
        weights = base_weights.copy()
        weights[~confident] = 1.0 / self.num_experts
        return weights

    def _tune_threshold(
        self,
        val_logits: np.ndarray,
        val_labels: np.ndarray,
        val_features: dict | None = None,
    ) -> None:
        """Tune confidence threshold to maximize coverage at target BA."""
        base_preds = self.base_router.predict_class(val_logits, val_features)
        base_weights = self.base_router.predict_proba(val_logits, val_features)
        routing_confidence = base_weights.max(axis=1)

        # Fallback predictions
        if self.fallback_router is not None:
            fallback_preds = self.fallback_router.predict_class(val_logits, val_features)
        else:
            avg_logits = val_logits.mean(axis=1)
            fallback_preds = avg_logits.argmax(axis=1)

        from scripts.utils.metrics import balanced_accuracy

        # Try different thresholds
        best_coverage = 0.0
        best_threshold = self.confidence_threshold

        for threshold in np.linspace(0.0, 0.95, 20):
            confident = routing_confidence >= threshold
            if confident.sum() == 0:
                continue

            final_preds = np.where(confident, base_preds, fallback_preds)
            ba = balanced_accuracy(val_labels, final_preds)
            coverage = confident.mean()

            # Prefer high coverage while maintaining BA close to full-routing BA
            full_ba = balanced_accuracy(val_labels, base_preds)
            if ba >= full_ba * 0.98 and coverage > best_coverage:
                best_coverage = coverage
                best_threshold = threshold

        self.confidence_threshold = best_threshold
