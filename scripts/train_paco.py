#!/usr/bin/env python3
"""
Train the PaCo (Parametric Contrastive Learning) expert.

Uses a MoCo-style framework (query encoder + momentum-updated key encoder
+ queue memory bank) and the PaCo loss that concatenates logit-adjusted
supervised logits with contrastive similarities.

References:
  - Cui et al., "Parametric Contrastive Learning", ICCV 2021.
  - He et al., "Momentum Contrast for Unsupervised Visual Representation
    Learning", CVPR 2020.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.resnet32 import PaCoResNet32
from losses.paco_loss import PaCoLoss
from .base_trainer import BaseTrainer, balanced_accuracy, group_accuracies


class PaCoTrainer(BaseTrainer):
    """
    Trainer for the PaCo expert.

    Overrides _train_one_epoch and validate because PaCo uses:
      - Two augmented views per sample
      - MoCo-style momentum encoder + queue
      - A specialised loss combining supervised and contrastive terms
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 1.0,
        gamma: float = 1.0,
        supt: float = 1.0,
        temperature: float = 0.07,
        K: int = 4096,
        **kwargs,
    ):
        model = PaCoResNet32(num_classes=100, dim=128, K=K, m=0.999, mlp=True)
        loss_fn = PaCoLoss(
            alpha=alpha, beta=beta, gamma=gamma, supt=supt,
            temperature=temperature, K=K, num_classes=100,
        )

        super().__init__(
            model=model,
            loss_fn=loss_fn,
            expert_name='PaCo',
            **kwargs,
        )
        self.alpha = alpha
        self.temperature = temperature
        self.K = K

    def _compute_loss(self, images, targets):
        # images is a list [view1, view2] for PaCo
        features, all_labels, logits = self.model(
            im_q=images[0], im_k=images[1], labels=targets
        )
        loss, aux = self.loss_fn(features, all_labels, logits)
        return loss, aux

    def _forward_for_eval(self, images):
        # Validation: single view, returns logits directly
        return self.model(im_q=images)

    def _train_one_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        total_loss = 0.0
        total_contrastive = 0.0
        total_ce = 0.0
        grad_norm_sum = 0.0
        n_batches = 0

        for images, targets in loader:
            # images is a list [view1, view2] each of shape (B, 3, 32, 32)
            images[0] = images[0].to(self.device)
            images[1] = images[1].to(self.device)
            targets = targets.to(self.device)

            loss, aux = self._compute_loss(images, targets)

            self.optimiser.zero_grad()
            loss.backward()

            # gradient norm
            total_norm_sq = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm_sq += p.grad.norm().item() ** 2
            grad_norm = total_norm_sq ** 0.5

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimiser.step()

            total_loss += loss.item()
            total_contrastive += aux['contrastive_loss'].item()
            total_ce += aux['ce_loss'].item()
            grad_norm_sum += grad_norm
            n_batches += 1

        return {
            'loss': total_loss / n_batches,
            'contrastive_loss': total_contrastive / n_batches,
            'ce_loss': total_ce / n_batches,
            'grad_norm': grad_norm_sum / n_batches,
        }

    def validate(self, loader):
        """Override because PaCo model returns only logits during eval."""
        self.model.eval()
        total_loss = 0.0
        all_targets, all_preds = [], []
        n_batches = 0

        with torch.no_grad():
            for images, targets in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                logits = self._forward_for_eval(images)
                # simple CE for validation loss tracking
                loss = nn.functional.cross_entropy(logits, targets)

                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                all_targets.append(targets.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                n_batches += 1

        all_targets = np.concatenate(all_targets)
        all_preds = np.concatenate(all_preds)
        ba, _ = balanced_accuracy(all_targets, all_preds)

        metrics = {'loss': total_loss / n_batches, 'ba': ba}
        if self.class_groups is not None:
            grp = group_accuracies(all_targets, all_preds, self.class_groups)
            for name, acc in grp.items():
                metrics[f'acc_{name}'] = acc
        return metrics


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
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weight for sample-sample contrastive positives')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Contrastive temperature')
    parser.add_argument('--K', type=int, default=4096,
                        help='Queue size for MoCo memory bank')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # ── data ──
    base_idx = np.load(f'{args.data_root}/processed/base_train_indices.npy')
    val_idx = np.load(f'{args.data_root}/processed/balanced_val_indices.npy')

    from data.cifar_lt import LongTailCIFAR100

    # Training set: two augmented views (for MoCo contrastive learning)
    train_set = LongTailCIFAR100(
        root=args.data_root,
        base_train_indices=base_idx,
        imbalance_ratio=100.0,
        train=True, download=False,
        two_view=True,
    )
    # Validation set: single view, balanced
    val_set = LongTailCIFAR100(
        root=args.data_root,
        base_train_indices=val_idx,
        imbalance_ratio=100.0,
        train=False, download=False,
        skip_longtail=True,
        two_view=False,
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, num_workers=2, pin_memory=True,
        drop_last=True,  # needed for queue alignment
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    class_counts = train_set.get_class_counts()

    # ── train ──
    trainer = PaCoTrainer(
        alpha=args.alpha,
        temperature=args.temperature,
        K=args.K,
        device=args.device,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
    )
    # Set class-frequency weights for logit adjustment
    trainer.loss_fn.set_class_weight(class_counts.tolist())

    trainer.train(train_loader, val_loader, class_counts=class_counts)
    trainer.save_history()


if __name__ == '__main__':
    main()
