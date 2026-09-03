"""
Product-of-experts routing — multiply softmax probabilities across experts.

No training required. Combines expert predictions via geometric mean.
"""

from __future__ import annotations

from typing import Self

import numpy as np

from scripts.router.base import BaseRouter
from scripts.utils.features import softmax

EPS = 1e-12


class ProductRouter(BaseRouter):
    """Route by multiplying softmax probabilities across experts.

    The product ensemble computes::

        P(y|x) ∝ ∏_{e} P_e(y|x)

    which is equivalent to summing log-probs. This tends to sharpen
    predictions and down-weight uncertain experts.
    """

    def train(
        self,
        val_logits: np.ndarray,
        val_labels: np.ndarray,
        val_features: dict | None = None,
    ) -> Self:
        """No training needed for product combination."""
        self._is_trained = True
        return self

    def predict(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Return expert with highest contribution to product."""
        probs = softmax(logits)
        product = np.prod(probs + EPS, axis=1)
        product /= product.sum(axis=1, keepdims=True)
        # Estimate per-expert contribution
        uniform = np.ones_like(probs) / probs.shape[-1]
        contribution = np.abs(probs - uniform).mean(axis=2)
        return contribution.argmax(axis=1)

    def predict_class(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Multiply softmax probabilities, renormalize, argmax."""
        probs = softmax(logits)
        product = np.prod(probs + EPS, axis=1)
        product /= product.sum(axis=1, keepdims=True)
        return product.argmax(axis=1)

    def predict_proba(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Product combination is inherently soft — return the product weights
        as the effective contribution of each expert to the final prediction.
        This is approximated by the normalized product contribution.
        """
        probs = softmax(logits)
        product = np.prod(probs + EPS, axis=1)
        product /= product.sum(axis=1, keepdims=True)
        # Estimate per-expert contribution: how much does each expert
        # shift the product away from uniform?
        uniform = np.ones_like(probs) / probs.shape[-1]
        contribution = np.abs(probs - uniform).mean(axis=2)  # (N, num_experts)
        contribution /= contribution.sum(axis=1, keepdims=True)
        return contribution
