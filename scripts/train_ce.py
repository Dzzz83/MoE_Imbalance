#!/usr/bin/env python3
"""Train the Cross-Entropy expert."""

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.resnet32 import ResNet32
from losses.ce_loss import CELoss
from .base_trainer import BaseTrainer


class CETrainer(BaseTrainer):
    """Trainer for the standard Cross-Entropy expert."""

    def __init__(self, **kwargs):
        model = ResNet32(num_classes=100)
        loss_fn = CELoss()
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            expert_name='CE',
            **kwargs,
        )

    def _compute_loss(self, images, targets):
        logits = self.model(images)
        loss = self.loss_fn(logits, targets)
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

    # ── prepare data ──────────────────────────────────────────────────
    base_idx = np.load(f'{args.data_root}/processed/base_train_indices.npy')
    val_idx = np.load(f'{args.data_root}/processed/balanced_val_indices.npy')

    from data.cifar_lt import LongTailCIFAR100

    train_set = LongTailCIFAR100(
        root=args.data_root,
        base_train_indices=base_idx,
        imbalance_ratio=100.0,
        train=True, download=False,
    )
    # Balanced validation set (50 samples per class, no long-tail)
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

    # ── class counts for head/medium/tail grouping ────────────────────
    class_counts = train_set.get_class_counts()

    # ── train ─────────────────────────────────────────────────────────
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
