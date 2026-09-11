"""
Test-Time Augmentation (TTA) routing.

Averages expert logits over several augmented views of each sample, then applies
a **parameter-free** router to the averaged logits. Nothing is fitted, so no
held-out data is required: TTA changes the inputs, not the decision rule.
"""

from __future__ import annotations

import numpy as np

from scripts.router.base import BaseRouter
from scripts.router.confidence import ConfidenceRouter


class TTARouter(BaseRouter):
    """Apply a parameter-free base router to TTA-averaged logits.

    Args:
        base_router: the rule to apply. Defaults to raw confidence selection.
        expert_names: expert labels; required when `base_router` is omitted.
        n_augs: number of augmented views averaged upstream.
    """

    def __init__(
        self,
        base_router: BaseRouter | None = None,
        expert_names: list[str] | None = None,
        n_augs: int = 10,
    ) -> None:
        if base_router is None:
            if not expert_names:
                raise ValueError(
                    "TTARouter needs either a base_router or explicit expert_names"
                )
            base_router = ConfidenceRouter(expert_names)
        super().__init__(expert_names if expert_names else base_router.expert_names)
        if n_augs < 1:
            raise ValueError(f"n_augs must be >= 1, got {n_augs}")
        self.base_router = base_router
        self.n_augs = n_augs

    def predict(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Delegate to the base router on TTA-averaged logits."""
        return self.base_router.predict(logits, features)

    def predict_proba(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Routing weights from the base router."""
        return self.base_router.predict_proba(logits, features)

    def predict_class(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Class predictions from the base router."""
        return self.base_router.predict_class(logits, features)

    def __repr__(self) -> str:
        return (f"TTARouter(base={self.base_router.name}, n_augs={self.n_augs}, "
                f"experts={self.expert_names})")
