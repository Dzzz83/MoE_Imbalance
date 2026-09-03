"""
Uniform averaging router — the simplest baseline.

Averages logits across all experts, then takes argmax.
No training required.
"""

from __future__ import annotations

from typing import Self

import numpy as np

from scripts.router.base import BaseRouter
from scripts.utils.features import softmax


class UniformRouter(BaseRouter):
    """Route by averaging logits across all experts.

    This is the standard uniform ensemble baseline.
    """

    def train(
        self,
        val_logits: np.ndarray,
        val_labels: np.ndarray,
        val_features: dict | None = None,
    ) -> Self:
        """No training needed for uniform averaging."""
        self._is_trained = True
        return self

    def predict(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Uniform averaging: no single expert chosen. Return expert 0 (all equal)."""
        return np.zeros(logits.shape[0], dtype=np.int64)

    def predict_proba(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Uniform weights for all experts."""
        N = logits.shape[0]
        return np.ones((N, self.num_experts), dtype=np.float32) / self.num_experts

    def predict_class(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Average logits across experts, then argmax."""
        avg_logits = logits.mean(axis=1)
        return avg_logits.argmax(axis=1)
