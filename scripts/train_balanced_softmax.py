#!/usr/bin/env python3
"""Train a Balanced Softmax expert (Ren et al., ECCV 2020)."""

import os
import sys

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.resnet32 import ResNet32
from losses.balanced_softmax_loss import BalancedSoftmaxLoss
from scripts.base_trainer import BaseTrainer
from data.cifar_lt import LongTailCIFAR100


class BalancedSoftmaxTrainer(BaseTrainer):
    """
    Trainer for the Balanced Softmax expert.

    Adjusts logits by adding log(class_counts) before softmax to
    compensate for class imbalance. Equivalent to LAL with τ=1
    when using raw counts instead of normalized priors.
    """

    def __init__(self, class_counts: torch.Tensor | None = None, **kwargs):
        self.model = ResNet32(num_classes=100)
        self._loss_fn = None
        self._class_counts = class_counts
        super().__init__(
            model=self.model,
            loss_fn=None,
            expert_name='BalancedSoftmax',
            **kwargs,
        )

    def _init_loss(self, class_counts: torch.Tensor):
        self._loss_fn = BalancedSoftmaxLoss(class_counts=class_counts).to(self.device)
        self.loss_fn = self._loss_fn

    def _compute_loss(self, images, targets, weights=None):
        logits = self.model(images)
        if weights is not None:
            # Recompute with weighting
            adjusted = logits + self._loss_fn.log_counts.unsqueeze(0)
            unreduced = torch.nn.functional.cross_entropy(
                adjusted, targets, reduction='none'
            )
            loss = (unreduced * weights).mean()
        else:
            loss = self._loss_fn(logits, targets)
        return loss, {}

    def _forward_for_eval(self, images):
        return self.model(images)

    def train(self, train_loader, val_loader, class_counts=None):
        if class_counts is not None:
            counts = torch.tensor(class_counts, dtype=torch.float32)
        else:
            counts = torch.ones(100, dtype=torch.float32)
        self._init_loss(counts)
        return super().train(train_loader, val_loader, class_counts=class_counts)


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
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # ── data ──
    train_idx = np.load(f'{args.data_root}/processed/lt_train_indices.npy')
    val_idx = np.load(f'{args.data_root}/processed/lt_val_indices.npy')

    train_set = LongTailCIFAR100(
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

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    class_counts = train_set.get_class_counts()

    # ── train ──
    trainer = BalancedSoftmaxTrainer(
        class_counts=torch.tensor(class_counts, dtype=torch.float32),
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
