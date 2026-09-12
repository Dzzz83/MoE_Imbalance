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

    #: Whether this rule must be fed TTA-averaged logits rather than plain
    #: single-view logits. Declared here so the evaluator supplies the right
    #: input structurally: an earlier version fed every rule the same plain
    #: logits, which silently turned the TTA row into a duplicate of Confidence.
    requires_tta: bool = False

    #: Whether ``predict`` names one expert per sample. Combine-then-argmax rules
    #: (uniform, probability averaging) make no such choice: their ``predict``
    #: returns expert 0 as a sentinel, so ``evaluate`` must take the accuracy from
    #: ``predict_class`` and the usage from ``predict_proba`` instead of reading
    #: that sentinel as a decision.
    selects_single_expert: bool = True

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
        """Compute routing metrics (BA, group accuracies, oracle, usage).

        The accuracy comes from ``predict_class`` — the rule's actual decision —
        so a combine-then-argmax rule (whose ``predict`` returns a sentinel) is
        not scored as if every sample had been routed to expert 0. Usage is the
        hard selection histogram for rules that pick an expert, and the mean
        routing weight for rules that combine all of them.
        """
        expert_indices = self.predict(logits, features)
        class_predictions = self.predict_class(logits, features)
        weights = (None if self.selects_single_expert
                   else self.predict_proba(logits, features))
        return compute_routing_metrics(
            expert_indices, labels, logits,
            self.expert_names, class_counts,
            class_predictions=class_predictions,
            routing_weights=weights,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(experts={self.expert_names})"
