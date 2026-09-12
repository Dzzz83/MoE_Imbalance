"""
Test-time augmentation views.

TTA improves predictions by averaging over several augmented versions of the
same input. The augmentation must match what the model saw during training —
otherwise the views are out of distribution and averaging degrades rather than
helps. Training uses ``RandomCrop(32, padding=4)`` plus ``RandomHorizontalFlip``
(see ``data/cifar_lt.py``), so this module reproduces exactly that.

Implemented with tensor ops rather than PIL transforms: the trainer already
normalised the images, and PIL round-trips would both slow the loop down and
risk a denormalise/renormalise mismatch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class AugmentationViews:
    """Generates ``n_views`` independently augmented copies of a batch.

    Args:
        n_views: number of views per sample (the pre-registered TTA uses 10).
        crop_padding: padding before the random 32x32 crop; 4 matches training.
        flip_prob: probability of a horizontal flip; 0.5 matches training.
        seed: fixes the view sampling, so TTA results are reproducible.

    Views are sampled from a dedicated ``torch.Generator``, so drawing views does
    not disturb the global RNG and a run stays reproducible regardless of how
    much randomness was consumed earlier.
    """

    def __init__(
        self,
        n_views: int = 10,
        crop_padding: int = 4,
        flip_prob: float = 0.5,
        seed: int = 0,
    ) -> None:
        if n_views < 1:
            raise ValueError(f"n_views must be >= 1, got {n_views}")
        if crop_padding < 0:
            raise ValueError(f"crop_padding must be >= 0, got {crop_padding}")
        if not 0.0 <= flip_prob <= 1.0:
            raise ValueError(f"flip_prob must be in [0, 1], got {flip_prob}")
        self.n_views = n_views
        self.crop_padding = crop_padding
        self.flip_prob = flip_prob
        self.seed = seed
        self._generator = torch.Generator().manual_seed(seed)

    def __call__(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Return ``n_views`` augmented copies of ``images`` (N, C, H, W)."""
        if images.dim() != 4:
            raise ValueError(f"images must be (N, C, H, W), got {tuple(images.shape)}")
        n, _, height, width = images.shape
        padded = F.pad(
            images,
            (self.crop_padding, self.crop_padding, self.crop_padding, self.crop_padding),
        )
        side = 2 * self.crop_padding
        span_h = side + 1
        span_w = side + 1

        views = []
        for _ in range(self.n_views):
            tops = torch.randint(0, span_h, (1,), generator=self._generator).item()
            lefts = torch.randint(0, span_w, (1,), generator=self._generator).item()
            view = padded[:, :, tops:tops + height, lefts:lefts + width]
            if self.flip_prob > 0.0:
                flips = torch.rand(n, generator=self._generator) < self.flip_prob
                view = torch.where(flips.view(n, 1, 1, 1), view.flip(-1), view)
            views.append(view.contiguous())
        return views

    def __repr__(self) -> str:
        return (f"AugmentationViews(n_views={self.n_views}, "
                f"crop_padding={self.crop_padding}, flip_prob={self.flip_prob}, "
                f"seed={self.seed})")
