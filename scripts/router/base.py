"""
Abstract base class for all routing methods.

Defines the standard interface:
  - train(val_logits, val_labels, val_features) → Self
  - predict(logits, features) → np.ndarray (expert index per sample)
  - evaluate(logits, labels, features, class_counts) → dict
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import numpy as np

from scripts.utils.metrics import compute_routing_metrics


class BaseRouter(ABC):
    """Abstract base for all expert routing methods.

    Subclasses must implement:
      - train()
      - predict()

    Subclasses may override:
      - evaluate()  (for custom evaluation logic)
      - name        (class property for reporting)
    """

    def __init__(self, expert_names: list[str] | None = None):
        if expert_names is None:
            expert_names = ["LAL", "Mixup", "PaCo"]
        self.expert_names = expert_names
        self.num_experts = len(expert_names)
        self._is_trained = False

    @property
    def name(self) -> str:
        """Human-readable name for this router."""
        return self.__class__.__name__.replace("Router", "")

    @abstractmethod
    def train(
        self,
        val_logits: np.ndarray,          # (N_val, num_experts, 100)
        val_labels: np.ndarray,           # (N_val,)
        val_features: dict | None = None, # optional extra features
    ) -> Self:
        """Train the router on validation data.

        Args:
            val_logits: Logits from all experts on validation set.
            val_labels: Ground-truth labels for validation set.
            val_features: Optional dict of precomputed features
                          (e.g., 24-d, 89-d, backbone features).
        Returns:
            self (trained router).
        """
        ...

    @abstractmethod
    def predict(
        self,
        logits: np.ndarray,               # (N, num_experts, 100)
        features: dict | None = None,     # optional features
    ) -> np.ndarray:
        """Return expert index (0..num_experts-1) for each sample.

        Args:
            logits: Logits from all experts on evaluation set.
            features: Optional dict of precomputed features.
        Returns:
            Array of shape (N,) with integer expert indices.
        """
        ...

    def predict_proba(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Return routing weights (num_experts,) per sample.

        Default implementation returns one-hot from predict().
        Override for soft routing methods.
        """
        indices = self.predict(logits, features)
        weights = np.zeros((logits.shape[0], self.num_experts))
        weights[np.arange(len(indices)), indices] = 1.0
        return weights

    def predict_class(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Return predicted class label (0..num_classes-1) for each sample.

        Default implementation: uses the chosen expert's logits → argmax.
        Override for soft routing methods (uniform, product) that combine
        experts before argmax.
        """
        expert_indices = self.predict(logits, features)
        chosen_logits = logits[np.arange(len(expert_indices)), expert_indices]
        return chosen_logits.argmax(axis=1)

    def evaluate(
        self,
        logits: np.ndarray,               # (N, num_experts, 100)
        labels: np.ndarray,               # (N,)
        class_counts: np.ndarray | None = None,
        features: dict | None = None,
    ) -> dict:
        """Evaluate routing performance.

        Args:
            logits: Logits from all experts on evaluation set.
            labels: Ground-truth labels.
            class_counts: Per-class sample counts (for head/med/tail).
            features: Optional features.
        Returns:
            Dict with keys: ba, accuracy, head_acc, med_acc, tail_acc,
                           oracle_ba, oracle_gap, all_wrong_pct,
                           expert_usage, expert_ba.
        """
        expert_indices = self.predict(logits, features)
        return compute_routing_metrics(
            expert_indices, labels, logits,
            self.expert_names, class_counts,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(experts={self.expert_names})"
