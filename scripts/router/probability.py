"""
Probability-averaging ensemble — the standard "soft vote" baseline.

Averages each expert's softmax distribution and takes the argmax. This is the
conventional ensemble baseline in the literature, and it is **not** the same
rule as averaging logits:

    probability average :  argmax_c  mean_e  softmax(z_e)_c
    logit average       :  argmax_c  mean_e  z_{e,c}

The difference matters for unbalanced experts. A model with larger logit
magnitudes dominates a *logit* average, whereas the probability average is
invariant to each expert's logit scale. On this project's pool the three
logit-adjusted experts have ~2.3x the logit scale of Mixup, so the two rules
genuinely differ (a test pins this).

Which one to use is not arbitrary: Buchanan et al., "The Effects of Ensembling
on Long-Tailed Data" (NeurIPS 2023 Heavy Tails workshop) compared logit and
probability ensembling and found them equivalent on balanced data but *different*
on imbalanced data, depending on ensemble diversity. Both are parameter-free, so
neither needs held-out data and neither is a cheating baseline.
"""

from __future__ import annotations

import numpy as np

from scripts.router.base import BaseRouter
from scripts.utils.features import softmax


class ProbabilityAverageRouter(BaseRouter):
    """Average expert softmax probabilities, then take the argmax."""

    #: No single expert is selected, so ``predict`` returns a sentinel and
    #: ``evaluate`` must score ``predict_class`` instead.
    selects_single_expert = False

    def predict(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """No single expert is selected; return expert 0 for every sample."""
        return np.zeros(logits.shape[0], dtype=np.int64)

    def predict_proba(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Equal weight for every expert — the average is unweighted."""
        return np.ones((logits.shape[0], self.num_experts), dtype=np.float32) / self.num_experts

    def predict_class(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Argmax of the mean expert probability distribution."""
        return self.averaged_probabilities(logits).argmax(axis=1)

    def averaged_probabilities(self, logits: np.ndarray) -> np.ndarray:
        """Mean softmax probability over experts, shape (N, num_classes)."""
        return softmax(logits).mean(axis=1)
