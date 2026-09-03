"""
Confidence-based routing — pick the expert with highest max-softmax confidence.

No training required. Optionally calibrates with temperature scaling
on the validation set.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from scipy.optimize import minimize_scalar

from scripts.router.base import BaseRouter
from scripts.utils.features import softmax


class ConfidenceRouter(BaseRouter):
    """Route to the expert with the highest max-softmax confidence.

    Optionally applies temperature scaling per expert, learned on
    the validation set to improve calibration.
    """

    def __init__(
        self,
        expert_names: list[str] | None = None,
        calibrate: bool = True,
    ):
        super().__init__(expert_names)
        self.calibrate = calibrate
        self.temperatures: list[float] | None = None

    def train(
        self,
        val_logits: np.ndarray,
        val_labels: np.ndarray,
        val_features: dict | None = None,
    ) -> Self:
        """Learn per-expert temperature scaling if calibrate=True."""
        if self.calibrate:
            self.temperatures = []
            for e in range(self.num_experts):
                logits_e = val_logits[:, e]  # (N, 100)
                temp = self._find_temperature(logits_e, val_labels)
                self.temperatures.append(temp)
        self._is_trained = True
        return self

    def predict(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Pick expert with highest calibrated max-softmax confidence."""
        confidences = self._get_confidences(logits)  # (N, num_experts)
        return confidences.argmax(axis=1)

    def predict_proba(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Return calibrated confidences as routing weights."""
        confidences = self._get_confidences(logits)
        # Softmax over confidences to get valid routing weights
        return softmax(confidences * 5.0)  # temperature for sharpness

    def _get_confidences(self, logits: np.ndarray) -> np.ndarray:
        """Compute calibrated max-softmax confidence per expert.

        Returns:
            Array shape (N, num_experts).
        """
        N = logits.shape[0]
        confidences = np.zeros((N, self.num_experts))
        for e in range(self.num_experts):
            logits_e = logits[:, e]  # (N, 100)
            if self.temperatures is not None:
                logits_e = logits_e / self.temperatures[e]
            probs = softmax(logits_e)
            confidences[:, e] = probs.max(axis=1)
        return confidences

    @staticmethod
    def _find_temperature(
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Find optimal temperature via NLL minimization."""
        def nll(t: float) -> float:
            scaled = logits / max(t, 1e-12)
            probs = softmax(scaled)
            nll_val = -np.log(probs[np.arange(len(labels)), labels] + 1e-12).mean()
            return nll_val

        result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
        return result.x if result.success else 1.0
