"""
Mixup augmentation — Zhang et al., ICLR 2018.

Kept in the data layer because mixup is an input/label transformation, not a
loss: the trainer still applies its own (possibly imbalance-aware) criterion to
the two mixed label sets.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Mixup:
    """Draws ``lam ~ Beta(alpha, alpha)`` and mixes a batch with its shuffle.

    Uses torch's RNG so that ``torch.manual_seed`` alone makes a run reproducible.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        self.alpha = float(alpha)
        self._beta = torch.distributions.Beta(self.alpha, self.alpha)

    def __call__(
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Returns (mixed_images, targets_a, targets_b, lam)."""
        lam = float(self._beta.sample().item())
        index = torch.randperm(images.size(0), device=images.device)
        mixed = lam * images + (1.0 - lam) * images[index]
        return mixed, targets, targets[index], lam

    @staticmethod
    def criterion(
        criterion: nn.Module,
        logits: torch.Tensor,
        targets_a: torch.Tensor,
        targets_b: torch.Tensor,
        lam: float,
    ) -> torch.Tensor:
        """Convex combination of the criterion on both label sets."""
        return lam * criterion(logits, targets_a) + (1.0 - lam) * criterion(logits, targets_b)
