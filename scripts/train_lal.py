#!/usr/bin/env python3
"""Train the Logit-Adjusted Loss (LAL) expert."""

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


class LALTrainer(BaseTrainer):
    """
    Trainer for the Logit-Adjusted Loss expert.

    The class priors are computed from the long-tailed training set
    and used to initialise the LAL loss.
    """

    def __init__(self, class_priors: torch.Tensor | None = None, **kwargs):
        self.model = ResNet32(num_classes=100)
        # loss is created after we know priors; set to None for now
        self._loss_fn = None
        self._class_priors = class_priors
        super().__init__(
            model=self.model,
            loss_fn=None,  # will be set in train()
            expert_name='LAL',
            **kwargs,
        )

    def _init_loss(self, class_priors: torch.Tensor):
        self._loss_fn = LALLoss(class_priors=class_priors, tau=1.0).to(self.device)
        # also update the base class loss_fn reference
        self.loss_fn = self._loss_fn

    def _compute_loss(self, images, targets):
        logits = self.model(images)
        loss = self._loss_fn(logits, targets)
        return loss, {}

    def _forward_for_eval(self, images):
        return self.model(images)

    def train(self, train_loader, val_loader, class_counts=None):
        # Compute class priors from the training set
        if class_counts is not None:
            priors = torch.tensor(
                class_counts / class_counts.sum(), dtype=torch.float32
            )
        else:
            # fallback: uniform priors
            priors = torch.ones(100, dtype=torch.float32) / 100.0
        self._init_loss(priors)
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

    # ── data ──────────────────────────────────────────────────────────
    base_idx = np.load(f'{args.data_root}/processed/base_train_indices.npy')
    val_idx = np.load(f'{args.data_root}/processed/balanced_val_indices.npy')

    from data.cifar_lt import LongTailCIFAR100

    train_set = LongTailCIFAR100(
        root=args.data_root,
        base_train_indices=base_idx,
        imbalance_ratio=100.0,
        train=True, download=False,
    )
    val_set = LongTailCIFAR100(
        root=args.data_root,
        base_train_indices=val_idx,
        imbalance_ratio=100.0,
        train=False, download=False,
        skip_longtail=True,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    class_counts = train_set.get_class_counts()
    priors = torch.tensor(class_counts / class_counts.sum(), dtype=torch.float32)

    # ── train ─────────────────────────────────────────────────────────
    trainer = LALTrainer(
        class_priors=priors,
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
