#!/usr/bin/env python3
"""
Unified training script for CIFAR-100-LT experts.

Usage:
    python scripts/train.py --method lal
    python scripts/train.py --method mixup
    python scripts/train.py --method paco --epochs 400 --lr 0.05
"""

import os
import sys

_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import argparse
import numpy as np
import torch

from scripts.utils.data import create_cifar_loader, print_data_info


def main():
    parser = argparse.ArgumentParser(
        description='Train a CIFAR-100-LT expert'
    )
    parser.add_argument('--method', type=str, required=True,
                        choices=['lal', 'mixup', 'paco', 'ce', 'balanced_softmax'],
                        help='Training method')
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load data ──
    train_loader, train_counts = create_cifar_loader(
        'train', args.data_root,
        batch_size=args.batch_size or (256 if args.method == 'paco' else 128),
        expert_name=args.method if args.method == 'paco' else None,
    )
    val_loader, val_counts = create_cifar_loader(
        'val', args.data_root,
        batch_size=256,
    )
    print_data_info(train_counts, val_counts, 10000)

    # ── Configure training ──
    config = {
        'method': args.method,
        'data_root': args.data_root,
        'checkpoint_dir': args.checkpoint_dir,
        'device': args.device,
        'epochs': args.epochs,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'weight_decay': args.weight_decay,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'class_counts': train_counts,
    }

    # ── Dispatch to method-specific trainer ──
    if args.method == 'lal':
        _train_lal(config)
    elif args.method == 'mixup':
        _train_mixup(config)
    elif args.method == 'paco':
        _train_paco(config)
    elif args.method == 'ce':
        _train_ce(config)
    elif args.method == 'balanced_softmax':
        _train_balanced_softmax(config)
    else:
        raise ValueError(f"Unknown method: {args.method}")


def _train_lal(config):
    """Train with Logit-Adjusted Loss."""
    from scripts.train_lal import LALTrainer

    epochs = config['epochs'] or 200
    lr = config['lr'] or 0.1
    batch_size = config['batch_size'] or 128

    trainer = LALTrainer(
        device=config['device'],
        lr=lr,
        weight_decay=config['weight_decay'],
        batch_size=batch_size,
        epochs=epochs,
        checkpoint_dir=config['checkpoint_dir'],
        class_counts=config['class_counts'],
    )
    trainer.train(config['train_loader'], config['val_loader'])


def _train_mixup(config):
    """Train with Mixup augmentation + CE loss."""
    epochs = config['epochs'] or 200
    lr = config['lr'] or 0.1
    batch_size = config['batch_size'] or 128

    from scripts.train_mixup import MixupTrainer
    trainer = MixupTrainer(
        device=config['device'],
        lr=lr,
        weight_decay=config['weight_decay'],
        batch_size=batch_size,
        epochs=epochs,
        checkpoint_dir=config['checkpoint_dir'],
        class_counts=config['class_counts'],
    )
    trainer.train(config['train_loader'], config['val_loader'])


def _train_paco(config):
    """Train with PaCo contrastive learning."""
    epochs = config['epochs'] or 400
    lr = config['lr'] or 0.05
    batch_size = config['batch_size'] or 256

    from scripts.train_paco import PaCoTrainer
    trainer = PaCoTrainer(
        device=config['device'],
        lr=lr,
        weight_decay=config['weight_decay'],
        batch_size=batch_size,
        epochs=epochs,
        checkpoint_dir=config['checkpoint_dir'],
        class_counts=config['class_counts'],
    )
    trainer.train(config['train_loader'], config['val_loader'])


# ── CE (standard Cross-Entropy) ─────────────────────────────────────────


def _train_ce(config):
    """Train with standard Cross-Entropy loss (no imbalance handling)."""
    from scripts.train_ce import CETrainer

    epochs = config['epochs'] or 200
    lr = config['lr'] or 0.1
    batch_size = config['batch_size'] or 128

    trainer = CETrainer(
        device=config['device'],
        lr=lr,
        weight_decay=config['weight_decay'],
        batch_size=batch_size,
        epochs=epochs,
        checkpoint_dir=config['checkpoint_dir'],
        class_counts=config['class_counts'],
    )
    trainer.train(config['train_loader'], config['val_loader'])


# ── Balanced Softmax ────────────────────────────────────────────────────


def _train_balanced_softmax(config):
    """Train with Balanced Softmax Loss (Ren et al., ECCV 2020)."""
    from scripts.train_balanced_softmax import BalancedSoftmaxTrainer

    epochs = config['epochs'] or 200
    lr = config['lr'] or 0.1
    batch_size = config['batch_size'] or 128

    trainer = BalancedSoftmaxTrainer(
        device=config['device'],
        lr=lr,
        weight_decay=config['weight_decay'],
        batch_size=batch_size,
        epochs=epochs,
        checkpoint_dir=config['checkpoint_dir'],
        class_counts=config['class_counts'],
    )
    trainer.train(config['train_loader'], config['val_loader'])


if __name__ == '__main__':
    main()
