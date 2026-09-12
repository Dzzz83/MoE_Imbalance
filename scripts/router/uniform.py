"""
Uniform averaging router — the simplest baseline.

Averages logits across all experts, then takes argmax. Parameter-free: nothing
is fitted, so no held-out data is required.
"""

from __future__ import annotations

import numpy as np

from scripts.router.base import BaseRouter


class UniformRouter(BaseRouter):
    """Route by averaging logits across all experts (the standard baseline)."""

    #: No single expert is selected, so ``predict`` returns a sentinel and
    #: ``evaluate`` must score ``predict_class`` instead.
    selects_single_expert = False

    def predict(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Uniform averaging selects no single expert; return expert 0 for all."""
        return np.zeros(logits.shape[0], dtype=np.int64)

    def predict_proba(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Equal weight for every expert."""
        return np.ones((logits.shape[0], self.num_experts), dtype=np.float32) / self.num_experts

    def predict_class(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Average logits across experts, then argmax."""
        return logits.mean(axis=1).argmax(axis=1)
