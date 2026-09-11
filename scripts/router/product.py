"""
Product-of-experts routing — multiply softmax probabilities across experts.

Parameter-free: combines expert predictions via the geometric mean, so nothing
is fitted and no held-out data is required.
"""

from __future__ import annotations

import numpy as np

from scripts.router.base import BaseRouter
from scripts.utils.features import softmax

EPS = 1e-12


class ProductRouter(BaseRouter):
    """Route by multiplying softmax probabilities across experts.

    The product ensemble computes ``P(y|x) ∝ ∏_e P_e(y|x)``, equivalent to
    summing log-probabilities, which sharpens predictions and down-weights
    uncertain experts.
    """

    def predict(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Return the expert contributing most to the product."""
        return self._contribution(logits).argmax(axis=1)

    def predict_class(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Multiply softmax probabilities, renormalise, argmax."""
        probs = softmax(logits)
        product = np.prod(probs + EPS, axis=1)
        product /= product.sum(axis=1, keepdims=True)
        return product.argmax(axis=1)

    def predict_proba(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Per-expert contribution to the product (normalised to sum to 1)."""
        return self._contribution(logits)

    @staticmethod
    def _contribution(logits: np.ndarray) -> np.ndarray:
        """How far each expert's distribution sits from uniform, normalised."""
        probs = softmax(logits)
        uniform = np.ones_like(probs) / probs.shape[-1]
        contribution = np.abs(probs - uniform).mean(axis=2)  # (N, num_experts)
        total = contribution.sum(axis=1, keepdims=True)
        return contribution / np.maximum(total, EPS)
