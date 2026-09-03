#!/usr/bin/env python3
"""
Train a Logit-Adjusted Loss (LAL) expert with per-sample loss weighting.

This extends the standard LAL trainer to accept per-sample weights,
which are used to scale the loss for each sample. This enables
boosting-style training where samples from previous experts' errors
are upweighted.

Usage:
    # Train with Mixup confidence weighting (Phase 2 of boosting)
    python scripts/train_lal_weighted.py \
        --data-root ./data \
        --checkpoint-dir ./checkpoints \
        --weight-source mixup_confidence \
        --alpha 2.0 \
        --device cuda

    # Train with custom weight file
    python scripts/train_lal_weighted.py \
        --data-root ./data \
        --checkpoint-dir ./checkpoints \
        --weight-file ./checkpoints/my_weights.npy \
        --device cuda
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
from data.weighted_dataset import WeightedDataset


class LALWeightedTrainer(BaseTrainer):
    """
    Trainer for Logit-Adjusted Loss with per-sample loss weighting.

    Extends LALTrainer by accepting per-sample weights and applying
    them to scale the loss during training.

    Args:
        sample_weights: numpy array of shape (N,) with per-sample weights.
                        If None, trains without weighting (standard LAL).
        **kwargs: passed to BaseTrainer.
    """

    def __init__(self, sample_weights: np.ndarray | None = None, **kwargs):
        self.model = ResNet32(num_classes=100)
        self._loss_fn = None
        self.sample_weights = sample_weights
        super().__init__(
            model=self.model,
            loss_fn=None,
            expert_name='LAL_Boost',
            **kwargs,
        )

    def _init_loss(self, class_priors: torch.Tensor):
        self._loss_fn = LALLoss(class_priors=class_priors, tau=1.0).to(self.device)
        self.loss_fn = self._loss_fn

    def _compute_loss(self, images, targets, weights=None):
        logits = self.model(images)
        loss = self._loss_fn(logits, targets)

        # Apply per-sample weights if provided
        if weights is not None:
            # loss is scalar (reduced) — we need unreduced loss for weighting
            # Recompute unreduced, apply weights, then reduce
            adjusted = logits + self._loss_fn.tau * self._loss_fn.log_prior.unsqueeze(0)
            unreduced = torch.nn.functional.cross_entropy(
                adjusted, targets, reduction='none'
            )
            loss = (unreduced * weights).mean()

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
            priors = torch.ones(100, dtype=torch.float32) / 100.0
        self._init_loss(priors)
        return super().train(train_loader, val_loader, class_counts=class_counts)


def load_weights(weight_source: str, data_root: str, alpha: float) -> np.ndarray:
    """
    Load or compute per-sample weights for training.

    Args:
        weight_source: One of:
            - 'mixup_confidence': weight = 1 + alpha * (1 - Mixup_confidence)
            - Path to a .npy file: load weights directly
        data_root: Data directory (for loading Mixup confidence)
        alpha: Weight strength parameter.

    Returns:
        weights: numpy array of shape (N,)
    """
    if weight_source.endswith('.npy'):
        # Custom weight file
        weights = np.load(weight_source)
        print(f"Loaded weights from {weight_source}: "
              f"min={weights.min():.3f}, max={weights.max():.3f}, "
              f"mean={weights.mean():.3f}")
        return weights

    elif weight_source == 'mixup_confidence':
        # Use Mixup's confidence as inverse weight
        confidence_path = os.path.join(
            os.path.dirname(data_root) if os.path.isfile(data_root) else data_root,
            'checkpoints', 'Mixup_train_confidence.npy'
        )
        # Also try checkpoints/ relative to project root
        if not os.path.exists(confidence_path):
            confidence_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'checkpoints', 'Mixup_train_confidence.npy'
            )

        if not os.path.exists(confidence_path):
            # Fallback: look in data-root's parent for checkpoints
            confidence_path = os.path.join(data_root, '..', 'checkpoints',
                                           'Mixup_train_confidence.npy')

        if not os.path.exists(confidence_path):
            raise FileNotFoundError(
                f"Mixup confidence not found at {confidence_path}. "
                f"Run 'python scripts/evaluate_expert.py --expert Mixup ...' first."
            )

        confidence = np.load(confidence_path)
        weights = 1.0 + alpha * (1.0 - confidence)
        print(f"Mixup confidence weights (α={alpha}): "
              f"min={weights.min():.3f}, max={weights.max():.3f}, "
              f"mean={weights.mean():.3f}, "
              f"fraction > 1.0: {(weights > 1.0).mean():.2%}")
        return weights

    else:
        raise ValueError(f"Unknown weight source: {weight_source}")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Train LAL expert with per-sample loss weighting'
    )
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--checkpoint-dir', default='./checkpoints')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    # Weighting options
    parser.add_argument('--weight-source', type=str, default=None,
                        help='Source of per-sample weights. Options: '
                             '"mixup_confidence", or path to .npy file. '
                             'If None, trains standard LAL.')
    parser.add_argument('--weight-file', type=str, default=None,
                        help='Path to .npy weight file (alternative to --weight-source)')
    parser.add_argument('--alpha', type=float, default=2.0,
                        help='Weight strength for confidence-based weighting '
                             '(default: 2.0). weight = 1 + alpha * (1 - confidence)')
    args = parser.parse_args()

    # ── data ──────────────────────────────────────────────────────────
    base_idx = np.load(f'{args.data_root}/processed/base_train_indices.npy')
    val_idx = np.load(f'{args.data_root}/processed/balanced_val_indices.npy')

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

    # ── load or compute weights ──────────────────────────────────────
    sample_weights = None
    weight_desc = "standard (no weighting)"

    if args.weight_file is not None:
        sample_weights = load_weights(args.weight_file, args.data_root, args.alpha)
        weight_desc = f"file={args.weight_file}, α={args.alpha}"
    elif args.weight_source is not None:
        sample_weights = load_weights(args.weight_source, args.data_root, args.alpha)
        weight_desc = f"source={args.weight_source}, α={args.alpha}"

    if sample_weights is not None:
        # Wrap training set with weights
        train_set = WeightedDataset(train_set, sample_weights)
        print(f"Using weighted training with {weight_desc}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    class_counts = train_set.base_dataset.get_class_counts() if isinstance(
        train_set, WeightedDataset) else train_set.get_class_counts()

    # ── train ─────────────────────────────────────────────────────────
    trainer = LALWeightedTrainer(
        sample_weights=sample_weights,
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
