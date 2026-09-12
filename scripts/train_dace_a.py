#!/usr/bin/env python3
"""
DACE Step 1: Train Expert A — Head-class specialist with LAL loss.

Expert A uses standard ResNet32 (no routing head) trained with:
  - LAL loss (τ=1.0) for logit-adjusted classification
  - PartitionedDataset: 80% Head classes + 20% all classes
  - 200 epochs, SGD, cosine LR schedule

Usage:
    python scripts/train_dace_a.py [--data-root DATA] [--checkpoint-dir CKPT]
                                   [--batch-size 128] [--epochs 200]
"""

import os
import sys

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.resnet32 import ResNet32
from losses.lal_loss import LALLoss
from scripts.base_trainer import BaseTrainer
from data.cifar_lt import LongTailCIFAR100
from data.partitioned_dataset import PartitionedDataset, PartitionedSampler


# ── Head class indices (training count >= 100) ───────────────────────────
# These are computed from the actual training data, but the CIFAR-100-LT
# IR=100 split has well-defined head/med/tail boundaries.
HEAD_THRESH = 100  # classes with >= 100 training samples


def get_head_classes(class_counts: np.ndarray) -> dict[str, np.ndarray]:
    """Return indices of Head classes (>= HEAD_THRESH training samples)."""
    head_idx = np.where(class_counts >= HEAD_THRESH)[0]
    return {'head': head_idx}


# ---------------------------------------------------------------------------
# DACE-A Trainer  (standard ResNet32 + LAL, no routing head)
# ---------------------------------------------------------------------------

class DACEATrainer(BaseTrainer):
    """Trainer for DACE Expert A (Head specialist, LAL)."""

    def __init__(self, class_priors: torch.Tensor | None = None, **kwargs):
        self.model = ResNet32(num_classes=100)
        self._loss_fn = None
        self._class_priors = class_priors
        super().__init__(
            model=self.model,
            loss_fn=None,
            expert_name='DACE_A',
            **kwargs,
        )

    def _init_loss(self, class_priors: torch.Tensor):
        self._loss_fn = LALLoss(class_priors=class_priors, tau=1.0).to(self.device)
        self.loss_fn = self._loss_fn

    def _compute_loss(self, images, targets, weights=None):
        logits = self.model(images)
        loss = self._loss_fn(logits, targets)
        if weights is not None:
            adjusted = logits + self._loss_fn.tau * self._loss_fn.log_prior.unsqueeze(0)
            unreduced = torch.nn.functional.cross_entropy(
                adjusted, targets, reduction='none'
            )
            loss = (unreduced * weights).mean()
        return loss, {}

    def _forward_for_eval(self, images):
        return self.model(images)

    def train(self, train_loader, val_loader, class_counts=None):
        if class_counts is not None:
            priors = torch.tensor(
                class_counts / class_counts.sum(), dtype=torch.float32
            )
        else:
            priors = torch.ones(100, dtype=torch.float32) / 100.0
        self._init_loss(priors)
        return super().train(train_loader, val_loader, class_counts=class_counts)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description='DACE Step 1: Train Expert A (Head)')
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--checkpoint-dir', default='./checkpoints')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--full-ratio', type=float, default=0.2,
                        help='Fraction of batches from ALL classes (default: 0.2)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── data ──────────────────────────────────────────────────────────
    train_idx = np.load(f'{args.data_root}/processed/lt_train_indices.npy')
    val_idx = np.load(f'{args.data_root}/processed/lt_val_indices.npy')

    base_train_set = LongTailCIFAR100(
        root=args.data_root,
        base_train_indices=train_idx,
        imbalance_ratio=100.0,
        train=True, download=False,
        already_subsampled=True,
    )
    val_set = LongTailCIFAR100(
        root=args.data_root,
        base_train_indices=val_idx,
        imbalance_ratio=100.0,
        train=False, download=False,
        already_subsampled=True,
    )

    class_counts = base_train_set.get_class_counts()
    head_groups = get_head_classes(class_counts)

    # Partitioned training set: bias toward Head classes
    train_set = PartitionedDataset(base_train_set)
    train_sampler = PartitionedSampler(
        base_train_set,
        primary_groups=head_groups,
        full_ratio=args.full_ratio,
        num_samples=len(base_train_set),
        primary_name='Head',
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    priors = torch.tensor(class_counts / class_counts.sum(), dtype=torch.float32)

    # ── train ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DACE Step 1: Training Expert A (Head specialist)")
    print(f"  Head classes: {head_groups['head'].tolist()}")
    print(f"  Full ratio: {args.full_ratio}")
    print(f"{'='*60}\n")

    trainer = DACEATrainer(
        class_priors=priors,
        device=args.device,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        seed=args.seed,   # BaseTrainer.train() reseeds; without this every
                          # --seed run was identical (it defaulted to 0)
    )
    trainer.train(train_loader, val_loader, class_counts=class_counts)
    trainer.save_history()
    print(f"  Expert A checkpoint saved to {args.checkpoint_dir}/DACE_A_best.pt")


if __name__ == '__main__':
    main()
