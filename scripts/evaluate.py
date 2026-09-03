#!/usr/bin/env python3
"""
Unified evaluation script for trained experts.

Usage:
    python scripts/evaluate.py --expert LAL --dataset test
    python scripts/evaluate.py --expert PaCo --dataset train --save-logits
    python scripts/evaluate.py --expert Mixup --dataset val --batch-size 64
"""

import os
import sys

_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import argparse
import json
import time

import numpy as np
import torch

from scripts.utils.data import (
    load_expert_checkpoint, create_cifar_loader, get_class_groups,
)
from scripts.utils.metrics import (
    balanced_accuracy, per_class_accuracy, group_accuracies,
    confidence_metrics,
)
from scripts.utils.features import extract_logits, softmax


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate a trained expert on any dataset split'
    )
    parser.add_argument('--expert', type=str, required=True,
                        choices=['LAL', 'Mixup', 'PaCo', 'CE', 'BalancedSoftmax'])
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to .pt file (default: checkpoints/{expert}_best.pt)')
    parser.add_argument('--dataset', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--output-dir', type=str, default='./checkpoints')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-logits', action='store_true')
    parser.add_argument('--save-metrics', action='store_true',
                        help='Save metrics JSON to output dir')
    args = parser.parse_args()

    # ── Load model ──
    print(f"Loading {args.expert}...")
    model = load_expert_checkpoint(args.expert, args.checkpoint, args.device)

    # ── Load data ──
    loader, class_counts = create_cifar_loader(
        args.dataset, args.data_root, batch_size=args.batch_size,
    )
    print(f"Evaluating on {args.dataset} set ({len(loader.dataset):,} samples)...")

    # ── Load training counts for meaningful head/med/tail grouping ──
    if args.dataset in ('test', 'val'):
        # Use training class counts for group definitions
        _, train_counts = create_cifar_loader(
            'train', args.data_root, batch_size=args.batch_size,
        )
        group_counts = train_counts
    else:
        group_counts = class_counts

    # ── Run evaluation ──
    t0 = time.time()
    logits = extract_logits(model, loader, args.device, args.expert)
    probs = softmax(logits)
    preds = logits.argmax(axis=1)
    targets = np.concatenate([t.numpy() for _, t in loader])
    elapsed = time.time() - t0

    # ── Compute metrics ──
    ba = balanced_accuracy(targets, preds)
    per_class = per_class_accuracy(targets, preds)
    groups = get_class_groups(group_counts)
    group_acc = group_accuracies(targets, preds, groups)
    conf_metrics = confidence_metrics(
        probs.max(axis=1),
        preds == targets,
    )

    accuracy = (preds == targets).mean()

    # ── Print results ──
    print(f"\nResults for {args.expert} on {args.dataset} set ({elapsed:.1f}s):")
    print(f"  Balanced Accuracy:  {ba:.2%}")
    print(f"  Accuracy:           {accuracy:.2%}  ({int(accuracy * len(targets))}/{len(targets)})")
    print(f"  Head Acc:           {group_acc.get('Head', 0):.2%}")
    print(f"  Med Acc:            {group_acc.get('Med', 0):.2%}")
    print(f"  Tail Acc:           {group_acc.get('Tail', 0):.2%}")
    print(f"  Avg conf (correct): {conf_metrics['avg_conf_correct']:.4f}")
    print(f"  Avg conf (wrong):   {conf_metrics['avg_conf_wrong']:.4f}")
    print(f"  ECE:                {conf_metrics['ece']:.4f}")

    # ── Save outputs ──
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = f"{args.expert}_{args.dataset}"

    np.save(os.path.join(args.output_dir, f"{prefix}_correctness.npy"), preds == targets)
    np.save(os.path.join(args.output_dir, f"{prefix}_confidence.npy"), probs.max(axis=1))
    np.save(os.path.join(args.output_dir, f"{prefix}_margin.npy"),
            np.sort(probs, axis=1)[:, -1] - np.sort(probs, axis=1)[:, -2])
    np.save(os.path.join(args.output_dir, f"{prefix}_preds.npy"), preds)

    if args.save_logits:
        np.save(os.path.join(args.output_dir, f"{prefix}_logits.npy"), logits)

    if args.save_metrics:
        metrics = {
            'expert': args.expert,
            'dataset': args.dataset,
            'ba': float(ba),
            'accuracy': float(accuracy),
            'head_acc': float(group_acc.get('Head', 0)),
            'med_acc': float(group_acc.get('Med', 0)),
            'tail_acc': float(group_acc.get('Tail', 0)),
            'ece': float(conf_metrics['ece']),
            'n_samples': len(targets),
        }
        with open(os.path.join(args.output_dir, f"{prefix}_metrics.json"), 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"  Metrics saved to {args.output_dir}/{prefix}_metrics.json")

    print(f"\n✅ Done.")


if __name__ == '__main__':
    main()
