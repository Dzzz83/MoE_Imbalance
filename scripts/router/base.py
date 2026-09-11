"""
Abstract base class for all routing methods.

Design constraint
-----------------
This project has **no validation split**: experts train on the full long-tailed
training set, so no honest held-out labels exist. A router that fitted anything
would have to fit on the test set. The interface therefore exposes **no fitting
entry point at all** — there is no ``train()``, ``fit()`` or ``calibrate()``
anywhere in this package, and only parameter-free mechanisms are registered.

Interface:
    - predict(logits, features)        -> expert index per sample
    - predict_proba(logits, features)  -> routing weights per sample
    - predict_class(logits, features)  -> class label per sample
    - evaluate(logits, labels, ...)    -> metrics dict
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from scripts.utils.metrics import compute_routing_metrics


class BaseRouter(ABC):
    """Abstract base for all expert routing methods.

    Subclasses implement ``predict``; they may override ``predict_proba`` and
    ``predict_class`` for soft or combine-then-argmax behaviour.

    Subclasses must not introduce any fitting method: the absence of held-out
    data is what makes these mechanisms honest, not the absence of a flag.
    """

    def __init__(self, expert_names: list[str]):
        if not expert_names:
            raise ValueError("expert_names must be a non-empty list")
        self.expert_names = list(expert_names)
        self.num_experts = len(self.expert_names)

    @property
    def name(self) -> str:
        """Human-readable name for this router."""
        return self.__class__.__name__.replace("Router", "")

    @abstractmethod
    def predict(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Return the selected expert index (0..num_experts-1) per sample.

        Args:
            logits: (N, num_experts, num_classes) expert logits.
            features: optional precomputed features (unused by parameter-free
                routers; kept so call sites share one signature).
        Returns:
            Array of shape (N,) with integer expert indices.
        """
        ...

    def predict_proba(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Return routing weights of shape (N, num_experts).

        Default: one-hot from ``predict``. Override for soft routing.
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
        """Return the predicted class label per sample.

        Default: argmax of the selected expert's logits. Override for
        combine-then-argmax routers (uniform, product).
        """
        expert_indices = self.predict(logits, features)
        chosen_logits = logits[np.arange(len(expert_indices)), expert_indices]
        return chosen_logits.argmax(axis=1)

    def evaluate(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
        class_counts: np.ndarray | None = None,
        features: dict | None = None,
    ) -> dict:
        """Compute routing metrics (BA, group accuracies, oracle, usage)."""
        expert_indices = self.predict(logits, features)
        return compute_routing_metrics(
            expert_indices, labels, logits,
            self.expert_names, class_counts,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(experts={self.expert_names})"
