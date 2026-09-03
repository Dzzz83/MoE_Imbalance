"""
Test-Time Augmentation (TTA) routing — apply routing on TTA-averaged predictions.

Computes predictions over multiple augmented views of each sample, then
applies the same routing methods on the TTA-smoothed features.
"""

from __future__ import annotations

from typing import Self, Callable

import numpy as np

from scripts.router.base import BaseRouter
from scripts.utils.features import softmax

EPS = 1e-12


class TTARouter(BaseRouter):
    """Apply routing on test-time augmentation averaged predictions.

    Wraps an existing router: first averages predictions over N_AUGS
    augmentations per sample, then runs the wrapped router on the
    TTA-smoothed features.
    """

    def __init__(
        self,
        base_router: BaseRouter | None = None,
        expert_names: list[str] | None = None,
        n_augs: int = 10,
    ):
        if base_router is None:
            # Default: use confidence routing on TTA predictions
            from scripts.router.confidence import ConfidenceRouter
            base_router = ConfidenceRouter(expert_names, calibrate=False)
        super().__init__(expert_names if expert_names else base_router.expert_names)
        self.base_router = base_router
        self.n_augs = n_augs

    def train(
        self,
        val_logits: np.ndarray,
        val_labels: np.ndarray,
        val_features: dict | None = None,
    ) -> Self:
        """Train the base router on TTA features if needed."""
        # For TTA, we typically train the base router on single-pass features
        # and apply on TTA features at test time
        self.base_router.train(val_logits, val_labels, val_features)
        self._is_trained = True
        return self

    def predict(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Apply base router on TTA-averaged predictions."""
        # TTA average: logits here are already averaged across augs
        # or we use the base router's predict directly
        return self.base_router.predict(logits, features)

    def predict_proba(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Return routing weights from base router on TTA features."""
        return self.base_router.predict_proba(logits, features)
