#!/usr/bin/env python3
"""
Train the PaCo (Parametric Contrastive Learning) expert.

Uses the EXACT hyperparameters from the official PaCo shell script:
    PaCo/LT/sh/CIFAR100_train_imb0.01.sh

Key settings from the paper:
  --alpha 0.01 --beta 1.0 --gamma 1.0
  --moco-t 0.05 --moco-k 1024 --moco-dim 32
  --lr 0.05 --epochs 400 --wd 5e-4
  --schedule [160, 180]  → step at epoch 320 and 360 (hardcoded in official code)
  --warmup-epochs 10
  --aug cifar100 (view1: AutoAugment+Cutout, view2: MoCo v2)
"""

import os
import sys

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import json
import time
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
from data.cifar_lt import LongTailCIFAR100


# ---------------------------------------------------------------------------
# Augmentations matching official PaCo --aug cifar100
# ---------------------------------------------------------------------------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

# View 1: regular augmentation with AutoAugment + Cutout
_augmentation_regular = Compose([
    ToPILImage(),
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    AutoAugment(AutoAugmentPolicy.CIFAR10),
    ToTensor(),
    RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
    Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

# View 2: MoCo v2-style augmentation
_augmentation_sim_cifar = Compose([
    ToPILImage(),
    RandomResizedCrop(size=32, scale=(0.2, 1.0)),
    RandomHorizontalFlip(),
    RandomApply([ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    RandomGrayscale(p=0.2),
    ToTensor(),
    Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

_val_transform = Compose([
    ToPILImage(),
    ToTensor(),
    Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


# ---------------------------------------------------------------------------
# Step LR schedule matching official PaCo
# ---------------------------------------------------------------------------

def adjust_learning_rate(optimizer, epoch, warmup_epochs, lr):
    """
    Exact LR schedule from official PaCo:
      - Warmup: linear 0 → lr over warmup_epochs
      - Epochs 1-320: lr
      - Epochs 321-360: lr * 0.1
      - Epochs 361-400: lr * 0.01
    """
    if epoch <= warmup_epochs:
        return lr / warmup_epochs * epoch  # linear warmup
    elif epoch > 360:
        return lr * 0.01
    elif epoch > 320:
        return lr * 0.1
    else:
        return lr


# ---------------------------------------------------------------------------
# PaCoTrainer
# ---------------------------------------------------------------------------

class PaCoTrainer(BaseTrainer):
    """
    Trainer for the PaCo expert with official hyperparameters.
    Overrides the training loop to use step LR schedule (not cosine).
    """

    def __init__(
        self,
        alpha: float = 0.01,
        beta: float = 1.0,
        gamma: float = 1.0,
        supt: float = 1.0,
        temperature: float = 0.05,
        K: int = 1024,
        warmup_epochs: int = 10,
        **kwargs,
    ):
        model = PaCoResNet32(num_classes=100, dim=32, K=K, m=0.999, mlp=True)
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
        self.paco_warmup_epochs = warmup_epochs

    # ── forward helpers ──────────────────────────────────────────────

    def _compute_loss(self, images, targets):
        features, all_labels, logits = self.model(
            im_q=images[0], im_k=images[1], labels=targets
        )
        loss, aux = self.loss_fn(features, all_labels, logits)
        return loss, aux

    def _forward_for_eval(self, images):
        return self.model(im_q=images)

    # ── training epoch ──────────────────────────────────────────────

    def _train_one_epoch(self, loader: DataLoader) -> dict:
        self.model.train()
        total_loss = 0.0
        total_contrastive = 0.0
        total_ce = 0.0
        grad_norm_sum = 0.0
        n_batches = 0

        for images, targets in loader:
            images[0] = images[0].to(self.device)
            images[1] = images[1].to(self.device)
            targets = targets.to(self.device)

            loss, aux = self._compute_loss(images, targets)

            self.optimiser.zero_grad()
            loss.backward()

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

    # ── validation ──────────────────────────────────────────────────

    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        all_targets, all_preds = [], []
        n_batches = 0

        with torch.no_grad():
            for images, targets in loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                logits = self._forward_for_eval(images)
                loss = nn.functional.cross_entropy(logits, targets)

                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                all_targets.append(targets.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                n_batches += 1

        all_targets = np.concatenate(all_targets)
        all_preds = np.concatenate(all_preds)
        ba, per_class_ba = balanced_accuracy(all_targets, all_preds)

        metrics = {'loss': total_loss / n_batches, 'ba': ba}
        if self.class_groups is not None:
            grp = group_accuracies(all_targets, all_preds, self.class_groups)
            for name, acc in grp.items():
                metrics[f'acc_{name}'] = acc
        return metrics

    # ── full training loop (overrides base to use step LR) ──────────

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        class_counts: np.ndarray | None = None,
    ) -> list[dict]:
        if class_counts is not None and self.class_groups is None:
            self.class_groups = compute_class_groups(class_counts)

        total_start = time.time()

        for epoch in range(1, self.epochs + 1):
            self.epoch = epoch
            epoch_start = time.time()

            # Step LR schedule (official PaCo)
            current_lr = adjust_learning_rate(
                self.optimiser, epoch,
                self.paco_warmup_epochs, self.lr,
            )
            for pg in self.optimiser.param_groups:
                pg['lr'] = current_lr

            train_metrics = self._train_one_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            log = {
                'epoch': epoch,
                'lr': current_lr,
                'time_s': time.time() - epoch_start,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'val_ba': val_metrics['ba'],
            }
            for prefix, src in [('train', train_metrics), ('val', val_metrics)]:
                for key in ('acc_head', 'acc_medium', 'acc_tail'):
                    if key in src:
                        log[f'{prefix}_{key}'] = src[key]
            log['grad_norm'] = train_metrics.get('grad_norm', 0.0)
            for k, v in train_metrics.items():
                if k not in ('loss', 'grad_norm', 'ba'):
                    log[f'train_{k}'] = v

            self.history.append(log)

            # Checkpoint by val BA
            current_val_ba = val_metrics.get('ba', 0.0)
            if current_val_ba > self.best_metric_val + 1e-3:
                self.best_metric_val = current_val_ba
                self._save_checkpoint(log, is_best=True)

            # Print
            if epoch == 1 or epoch % 50 == 0 or epoch == self.epochs:
                h = val_metrics.get('acc_head', 0.0)
                m = val_metrics.get('acc_medium', 0.0)
                t = val_metrics.get('acc_tail', 0.0)
                print(
                    f"[PaCo] Epoch {epoch:3d}/{self.epochs} | LR {current_lr:.5f} | "
                    f"Train Loss {log['train_loss']:.4f} | Val Loss {log['val_loss']:.4f} | "
                    f"Val BA {log['val_ba']:.2%} | "
                    f"H {h:.1%} M {m:.1%} T {t:.1%} | "
                    f"GradNorm {log['grad_norm']:.2f}"
                )

        total_time = time.time() - total_start
        print(f"[PaCo] ✓ Done in {total_time:.0f}s. "
              f"Best Val BA = {self.best_metric_val:.2%}")
        return self.history


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--checkpoint-dir', default='./checkpoints')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--alpha', type=float, default=0.01,
                        help='Weight for sample-sample contrastive positives')
    parser.add_argument('--temperature', type=float, default=0.05,
                        help='Contrastive temperature')
    parser.add_argument('--K', type=int, default=1024,
                        help='Queue size for MoCo memory bank')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # ── data ──
    base_idx = np.load(f'{args.data_root}/processed/base_train_indices.npy')
    val_idx = np.load(f'{args.data_root}/processed/balanced_val_indices.npy')

    # Override transforms on the dataset for PaCo-specific augmentations
    # We monkey-patch the transform after dataset creation
    train_set = LongTailCIFAR100(
        root=args.data_root,
        base_train_indices=base_idx,
        imbalance_ratio=100.0,
        train=True, download=False,
        two_view=True,
    )
    val_set = LongTailCIFAR100(
        root=args.data_root,
        base_train_indices=val_idx,
        imbalance_ratio=100.0,
        train=False, download=False,
        skip_longtail=True,
        two_view=False,
    )

    # Replace transforms with official PaCo augmentations
    train_set.transform = [_augmentation_regular, _augmentation_sim_cifar]
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
        warmup_epochs=10,
        checkpoint_dir=args.checkpoint_dir,
    )
    trainer.loss_fn.set_class_weight(class_counts.tolist())

    trainer.train(train_loader, val_loader, class_counts=class_counts)
    trainer.save_history()


if __name__ == '__main__':
    main()
