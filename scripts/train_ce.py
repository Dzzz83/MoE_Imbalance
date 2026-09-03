#!/usr/bin/env python3
"""Train a standard Cross-Entropy expert (no imbalance handling)."""

import os
import sys

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.resnet32 import ResNet32
from losses.ce_loss import CELoss
from scripts.base_trainer import BaseTrainer
from data.cifar_lt import LongTailCIFAR100


class CETrainer(BaseTrainer):
    """
    Trainer for standard Cross-Entropy expert.

    No imbalance handling — serves as a baseline to measure the effect
    of long-tail losses (LAL, Balanced Softmax).
    """

    def __init__(self, **kwargs):
        self.model = ResNet32(num_classes=100)
        self._loss_fn = CELoss()
        super().__init__(
            model=self.model,
            loss_fn=self._loss_fn,
            expert_name='CE',
            **kwargs,
        )

    def _compute_loss(self, images, targets, weights=None):
        logits = self.model(images)
        if weights is not None:
            unreduced = torch.nn.functional.cross_entropy(
                logits, targets, reduction='none'
            )
            loss = (unreduced * weights).mean()
        else:
            loss = self._loss_fn(logits, targets)
        return loss, {}

    def _forward_for_eval(self, images):
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
    trainer = CETrainer(
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
