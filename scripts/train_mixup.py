#!/usr/bin/env python3
"""Train the Mixup + Cross-Entropy expert."""

import os
import sys

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.resnet32 import ResNet32
from losses.ce_loss import CELoss
from scripts.base_trainer import BaseTrainer
from data.cifar_lt import LongTailCIFAR100


# ---------------------------------------------------------------------------
# Mixup helpers
# ---------------------------------------------------------------------------

def mixup_data(
    images: torch.Tensor, targets: torch.Tensor, alpha: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Apply Mixup augmentation to a batch.

    Returns:
        mixed_images:   linearly interpolated images
        targets_a:      original labels
        targets_b:      permuted labels
        lam:            mixing coefficient (from Beta(alpha, alpha))
    """
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1.0 - lam) * images[index]
    targets_a = targets
    targets_b = targets[index]
    return mixed_images, targets_a, targets_b, lam


def mixup_criterion(
    criterion: torch.nn.Module,
    pred: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute the Mixup loss as a convex combination of two CE losses."""
    return lam * criterion(pred, targets_a) + (1.0 - lam) * criterion(pred, targets_b)


# ---------------------------------------------------------------------------
# MixupTrainer
# ---------------------------------------------------------------------------

class MixupTrainer(BaseTrainer):
    """
    Trainer for the Mixup + Cross-Entropy expert.

    Uses standard CE loss but applies Mixup augmentation (Zhang et al., ICLR 2018)
    at the batch level during training.  Validation uses plain forward pass.
    """

    def __init__(self, mixup_alpha: float = 1.0, **kwargs):
        model = ResNet32(num_classes=100)
        loss_fn = CELoss()
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            expert_name='Mixup',
            **kwargs,
        )
        self.mixup_alpha = mixup_alpha

    def _compute_loss(self, images, targets, weights=None):
        # Apply Mixup
        mixed_images, targets_a, targets_b, lam = mixup_data(
            images, targets, alpha=self.mixup_alpha,
        )
        logits = self.model(mixed_images)
        loss = mixup_criterion(self.loss_fn, logits, targets_a, targets_b, lam)
        if weights is not None:
            # Weight the loss per-sample
            loss = (loss * weights).mean()
        return loss, {}

    def _forward_for_eval(self, images):
        # No Mixup at validation / test time
        return self.model(images)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--checkpoint-dir', default='./checkpoints')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--mixup-alpha', type=float, default=1.0,
                        help='Alpha parameter for Beta distribution in Mixup')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # ── prepare data ──────────────────────────────────────────────────
    train_idx = np.load(f'{args.data_root}/processed/lt_train_indices.npy')
    val_idx = np.load(f'{args.data_root}/processed/lt_val_indices.npy')

    train_set = LongTailCIFAR100(
        root=args.data_root,
        base_train_indices=train_idx,
        imbalance_ratio=100.0,
        train=True, download=False,
        already_subsampled=True,
    )
    # Balanced validation set (50 samples per class, no long-tail)
    val_set = LongTailCIFAR100(
        root=args.data_root,
        base_train_indices=val_idx,
        imbalance_ratio=100.0,
        train=False, download=False,
        already_subsampled=True,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    # ── class counts for head/medium/tail grouping ────────────────────
    class_counts = train_set.get_class_counts()

    # ── train ─────────────────────────────────────────────────────────
    trainer = MixupTrainer(
        mixup_alpha=args.mixup_alpha,
        device=args.device,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
    )
    trainer.train(train_loader, val_loader, class_counts=class_counts)
    trainer.save_history()


if __name__ == '__main__':
    main()
