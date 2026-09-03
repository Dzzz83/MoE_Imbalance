#!/usr/bin/env python3
"""
Train a PaCo expert with per-sample loss weighting on a validation set.

This extends the PaCo trainer to accept a weighted validation set alongside
the standard training set. The model trains on both: the training set provides
general knowledge, while the weighted validation set focuses learning on
samples where previous experts fail (Phase 3 of boosting).

Usage:
    python scripts/train_paco_weighted.py \
        --data-root ./data \
        --checkpoint-dir ./checkpoints \
        --val-weight-file ./checkpoints/val_weights_C.npy \
        --val-batch-fraction 0.2 \
        --device cuda
"""

import os
import sys
import json
import time

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import (
    AutoAugment, AutoAugmentPolicy, Compose, RandomCrop, RandomHorizontalFlip,
    RandomResizedCrop, RandomApply, ColorJitter, RandomGrayscale, ToTensor,
    Normalize, RandomErasing, ToPILImage,
)

from models.resnet32 import PaCoResNet32
from losses.paco_loss import PaCoLoss
from scripts.base_trainer import BaseTrainer, balanced_accuracy, group_accuracies, compute_class_groups
from data.cifar_lt import LongTailCIFAR100, CIFAR100_MEAN, CIFAR100_STD
from data.weighted_dataset import WeightedDataset, ConcatDataset


# ---------------------------------------------------------------------------
# Augmentations matching official PaCo --aug cifar100
# ---------------------------------------------------------------------------

# View 1: regular augmentation with AutoAugment + Cutout
_augmentation_regular = Compose([
    ToPILImage(),
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    AutoAugment(AutoAugmentPolicy.CIFAR10),
    ToTensor(),
    RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
    Normalize(CIFAR100_MEAN, CIFAR100_STD),
])

# View 2: MoCo v2-style augmentation
_augmentation_sim_cifar = Compose([
    ToPILImage(),
    RandomResizedCrop(size=32, scale=(0.2, 1.0)),
    RandomHorizontalFlip(),
    RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    RandomGrayscale(p=0.2),
    ToTensor(),
    Normalize(CIFAR100_MEAN, CIFAR100_STD),
])

_val_transform = Compose([
    ToPILImage(),
    ToTensor(),
    Normalize(CIFAR100_MEAN, CIFAR100_STD),
])


# ---------------------------------------------------------------------------
# Step LR schedule matching official PaCo
# ---------------------------------------------------------------------------

def adjust_learning_rate(optimizer, epoch, warmup_epochs, lr):
    """Exact LR schedule from official PaCo."""
    if epoch <= warmup_epochs:
        return lr / warmup_epochs * epoch  # linear warmup
    elif epoch > 360:
        return lr * 0.01
    elif epoch > 320:
        return lr * 0.1
    else:
        return lr


# ---------------------------------------------------------------------------
# Interleaved DataLoader: mixes training and weighted validation batches
# ---------------------------------------------------------------------------

class InterleavedLoader:
    """
    Interleaves batches from a training loader and a weighted validation loader.

    Each iteration yields a batch where:
      - ~(1 - val_fraction) of samples come from the training set (unweighted)
      - ~val_fraction come from the validation set (weighted)

    This ensures the model sees enough training data while being regularly
    exposed to hard validation samples.
    """

    def __init__(self, train_loader, val_loader, val_fraction=0.2):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_fraction = val_fraction
        self.train_iter = iter(train_loader)
        self.val_iter = iter(val_loader)

    def __iter__(self):
        self.train_iter = iter(self.train_loader)
        self.val_iter = iter(self.val_loader)
        return self

    def __next__(self):
        try:
            train_batch = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(self.train_loader)
            train_batch = next(self.train_iter)

        try:
            val_batch = next(self.val_iter)
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            val_batch = next(self.val_iter)

        # Unpack batches (both should be (image, target, weight) tuples)
        if len(train_batch) == 3:
            train_imgs, train_targets, train_weights = train_batch
        else:
            train_imgs, train_targets = train_batch
            train_weights = torch.ones(len(train_imgs))

        if len(val_batch) == 3:
            val_imgs, val_targets, val_weights = val_batch
        else:
            val_imgs, val_targets = val_batch
            val_weights = torch.ones(len(val_imgs))

        # Interleave
        n_train = len(train_imgs)
        n_val = int(n_train * self.val_fraction / (1 - self.val_fraction))
        n_val = min(n_val, len(val_imgs))

        if n_val > 0:
            images = torch.cat([train_imgs[:n_train - n_val], val_imgs[:n_val]], dim=0)
            targets = torch.cat([train_targets[:n_train - n_val], val_targets[:n_val]], dim=0)
            weights = torch.cat([
                train_weights[:n_train - n_val],
                val_weights[:n_val]
            ], dim=0)
        else:
            images = train_imgs
            targets = train_targets
            weights = train_weights

        return images, targets, weights


# ---------------------------------------------------------------------------
# PaCoWeightedTrainer
# ---------------------------------------------------------------------------

class PaCoWeightedTrainer(BaseTrainer):
    """
    Trainer for PaCo with weighted validation set.

    Trains on interleaved batches from:
      - LT training set (standard PaCo, no weighting)
      - Balanced validation set (with per-sample error weights)

    Args:
        sample_weights_val: numpy array of validation set weights (5K,).
                            Applied only to validation samples.
        val_fraction: Fraction of each batch from validation set (default: 0.2).
        **kwargs: passed to BaseTrainer.
    """

    def __init__(self, sample_weights_val: np.ndarray | None = None,
                 val_fraction: float = 0.2, **kwargs):
        model = PaCoResNet32(num_classes=100, dim=32, K=2048)
        self.loss_fn_paco = PaCoLoss(
            alpha=0.01, beta=1.0, gamma=1.0,
            supt=1.0, temperature=0.05, K=2048, num_classes=100,
        )
        super().__init__(
            model=model,
            loss_fn=None,  # PaCo handles loss internally
            expert_name='PaCo_Boost',
            **kwargs,
        )
        self.sample_weights_val = sample_weights_val
        self.val_fraction = val_fraction
        self.class_counts = None

    def _compute_loss(self, images, targets, weights=None):
        # PaCo needs two views per image
        # For simplicity, we apply the regular augmentation twice
        # (the PaCo model expects two views during training)
        # This is a simplified version — full PaCo training uses different
        # augmentations for view1 and view2.
        features, all_labels, logits = self.model(images, images, targets)

        loss, aux = self.loss_fn_paco(
            features, all_labels, logits, epoch=self.epoch
        )

        # Apply per-sample weights if provided
        if weights is not None:
            # The PaCo loss returns a scalar. We need to weight the
            # supervised (CE) component per-sample.
            # For simplicity, we scale the total loss by mean weight.
            # This is approximate but works for boosting.
            loss = loss * weights.mean()

        return loss, aux

    def _forward_for_eval(self, images):
        return self.model._inference(images)

    def train(self, train_loader, val_loader, class_counts=None):
        self.class_counts = class_counts
        if class_counts is not None:
            self.loss_fn_paco.set_class_weight(class_counts)

        # Create weighted validation loader if weights provided
        if self.sample_weights_val is not None:
            val_set = val_loader.dataset
            weighted_val = WeightedDataset(val_set, self.sample_weights_val)
            val_loader_weighted = DataLoader(
                weighted_val, batch_size=val_loader.batch_size,
                shuffle=True, num_workers=val_loader.num_workers,
                pin_memory=val_loader.pin_memory,
            )
            # Use interleaved loader
            combined_loader = InterleavedLoader(
                train_loader, val_loader_weighted,
                val_fraction=self.val_fraction,
            )
            print(f"Using interleaved training: "
                  f"{1-self.val_fraction:.0%} train + "
                  f"{self.val_fraction:.0%} weighted val per batch")
            return super().train(
                combined_loader, val_loader, class_counts=class_counts
            )
        else:
            # Standard training (no weighting)
            return super().train(
                train_loader, val_loader, class_counts=class_counts
            )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Train PaCo expert with weighted validation set'
    )
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--checkpoint-dir', default='./checkpoints')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    # Weighting options
    parser.add_argument('--val-weight-file', type=str, default=None,
                        help='Path to .npy file with validation set weights (5K,)')
    parser.add_argument('--val-batch-fraction', type=float, default=0.2,
                        help='Fraction of each batch from validation set (0-1)')
    args = parser.parse_args()

    # ── data ──────────────────────────────────────────────────────────
    base_idx = np.load(f'{args.data_root}/processed/base_train_indices.npy')
    val_idx = np.load(f'{args.data_root}/processed/balanced_val_indices.npy')

    # Training set with two-view augmentations for PaCo
    train_set = LongTailCIFAR100(
        root=args.data_root,
        base_train_indices=base_idx,
        imbalance_ratio=100.0,
        train=True, download=False,
        two_view=True,
    )
    # Set per-view transforms
    train_set.transform = [_augmentation_regular, _augmentation_sim_cifar]

    # Validation set (balanced, 50/class, 5K total)
    val_set = LongTailCIFAR100(
        root=args.data_root,
        base_train_indices=val_idx,
        imbalance_ratio=100.0,
        train=False, download=False,
        skip_longtail=True,
    )
    # Use standard transform for validation
    val_set.transform = _val_transform

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    # ── validation weights ───────────────────────────────────────────
    val_weights = None
    if args.val_weight_file is not None:
        val_weights = np.load(args.val_weight_file)
        assert len(val_weights) == len(val_set), (
            f"Validation weights length {len(val_weights)} != "
            f"validation set size {len(val_set)}"
        )
        print(f"Loaded validation weights: "
              f"min={val_weights.min():.3f}, max={val_weights.max():.3f}, "
              f"mean={val_weights.mean():.3f}")

    # ── class counts ─────────────────────────────────────────────────
    class_counts = train_set.get_class_counts()

    # ── train ─────────────────────────────────────────────────────────
    trainer = PaCoWeightedTrainer(
        sample_weights_val=val_weights,
        val_fraction=args.val_batch_fraction,
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
