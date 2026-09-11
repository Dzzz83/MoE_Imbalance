"""
Confidence-based routing — pick the expert with the highest max-softmax confidence.

Parameter-free. The temperature-calibration path that previously fitted a
per-expert temperature on held-out labels was removed along with the other
fitted mechanisms: this project has no validation split to fit it on
(see `docs/routing-results-record.md`).
"""

from __future__ import annotations

import numpy as np

from scripts.router.base import BaseRouter
from scripts.utils.features import softmax


class ConfidenceRouter(BaseRouter):
    """Route to the expert with the highest raw max-softmax confidence."""

    def predict(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Pick the expert with the highest max-softmax confidence.

        Note (recorded, not fixed here): in savable samples the correct expert
        is systematically the *least* confident one — the lone-dissenter
        paradox, `research-findings.md` §1.3 — so this rule is expected to
        underperform uniform averaging.
        """
        return self.confidences(logits).argmax(axis=1)

    def predict_proba(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Softmax over raw confidences, used as routing weights."""
        return softmax(self.confidences(logits) * 5.0)

    def confidences(self, logits: np.ndarray) -> np.ndarray:
        """Raw max-softmax confidence per expert, shape (N, num_experts)."""
        probs = softmax(logits)          # (N, num_experts, num_classes)
        return probs.max(axis=2)
