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

from data.cifar_lt import CIFAR100_MEAN, CIFAR100_STD


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
        n, channels, height, width = images.shape
        if channels != len(CIFAR100_MEAN):
            raise ValueError(
                "normalized TTA expects CIFAR-100 RGB images with 3 channels, "
                f"got {channels}"
            )

        # ``LongTailCIFAR100`` applies RandomCrop to a PIL image before
        # ToTensor/Normalize.  RandomCrop's constant fill is black (0 in the
        # uint8 image), which becomes -mean/std after normalization.  Padding
        # an already-normalized tensor with zero would instead insert a
        # mean-colour pixel and changes the distribution seen by TTA.
        if self.crop_padding:
            padding = images.new_tensor(CIFAR100_MEAN) / images.new_tensor(CIFAR100_STD)
            black = -padding.view(1, channels, 1, 1)
            padded = images.new_empty(
                n,
                channels,
                height + 2 * self.crop_padding,
                width + 2 * self.crop_padding,
            )
            padded[:] = black
            padded[
                :,
                :,
                self.crop_padding:self.crop_padding + height,
                self.crop_padding:self.crop_padding + width,
            ] = images
        else:
            padded = images
        side = 2 * self.crop_padding
        span_h = side + 1
        span_w = side + 1

        views = []
        for _ in range(self.n_views):
            # Draw every crop independently, matching RandomCrop being called
            # once per dataset sample.  The dedicated CPU generator keeps the
            # random stream stable across CPU and CUDA inputs; only the index
            # tensors are transferred to the input device.
            tops = torch.randint(0, span_h, (n,), generator=self._generator)
            lefts = torch.randint(0, span_w, (n,), generator=self._generator)
            view = torch.stack([
                padded[index, :, int(tops[index]):int(tops[index]) + height,
                       int(lefts[index]):int(lefts[index]) + width]
                for index in range(n)
            ])
            if self.flip_prob > 0.0:
                flips = (
                    torch.rand(n, generator=self._generator) < self.flip_prob
                ).to(images.device)
                view = torch.where(flips.view(n, 1, 1, 1), view.flip(-1), view)
            views.append(view.contiguous())
        return views

    def __repr__(self) -> str:
        return (f"AugmentationViews(n_views={self.n_views}, "
                f"crop_padding={self.crop_padding}, flip_prob={self.flip_prob}, "
                f"seed={self.seed})")
