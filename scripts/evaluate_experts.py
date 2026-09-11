#!/usr/bin/env python3
"""
Evaluate the trained experts and measure routing headroom on the test set.

**This is the only entry point that reads the CIFAR-100 test set**, and it
appends an entry to `docs/test-access-log.md` before doing so. The candidate
routing rules were frozen in `docs/routing-preregistration.md`; run this only
after that file is fixed, and run it **once** so all rules are compared on the
same look.

Reports:
  * per-expert BA / Head / Med / Tail / ECE  (are the models trained correctly?)
  * routing headroom: all-wrong floor, oracle ceiling, pairwise Cohen's kappa
  * the four frozen parameter-free routing rules, for context

Usage:
    python scripts/evaluate_experts.py --seeds 78
    python scripts/evaluate_experts.py --seeds 78 88 1034 --device cuda
"""

from __future__ import annotations

import os
import sys

_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.cifar_lt import LongTailCIFAR100
from data.lt_datamodule import LongTailDataModule
from scripts.evaluation import (
    EvaluationError, ExpertPool, HeadroomAnalyzer, evaluate_predictions,
)
from scripts.router import ROUTERS
from scripts.utils.test_access import TestAccessLog

DEFAULT_EXPERTS = ['CE', 'LAL', 'BalancedSoftmax', 'Mixup']


def build_test_loader(data_root: str, batch_size: int = 256) -> DataLoader:
    """The balanced 10K CIFAR-100 test set. The only test-set reader."""
    dataset = LongTailCIFAR100(root=data_root, train=False, use_test_set=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', default='./checkpoints')
    parser.add_argument('--experts', nargs='+', default=DEFAULT_EXPERTS)
    parser.add_argument('--seeds', nargs='+', type=int, default=[78, 88, 1034])
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output', default=None,
                        help='JSON output path (default: checkpoints/test_evaluation.json)')
    args = parser.parse_args(argv)

    # ── log the access BEFORE reading the test set ──
    logged = TestAccessLog().record(
        ' '.join(sys.argv),
        note=f"experts={','.join(args.experts)} seeds={','.join(map(str, args.seeds))}",
    )
    if not logged:
        print("WARNING: could not write the test-access log; continuing anyway")

    pool = ExpertPool(args.experts, args.seeds,
                      checkpoint_dir=args.checkpoint_dir, device=args.device)
    absent = pool.missing()
    if absent:
        print(f"Note: no checkpoint for {absent} at seeds {args.seeds} — skipped")
    pool.load()
    print(f"Loaded experts: {pool.loaded}\n")

    loader = build_test_loader(args.data_root, args.batch_size)
    logits, targets = pool.logits(loader)          # (N, E, C)
    print(f"Test set: {len(targets)} samples, logits {logits.shape}\n")

    # class groups come from the TRAINING counts (the split is immutable)
    train_counts = LongTailDataModule(root=args.data_root).class_counts()

    results: dict = {'experts': {}, 'routing': {}, 'headroom': {}}

    # ── per-expert metrics ──
    print("Per-expert performance on the balanced test set")
    print(f"  {'expert':<18}{'BA':>8}{'Head':>8}{'Med':>8}{'Tail':>8}{'ECE':>8}")
    for i, name in enumerate(pool.loaded):
        probs = softmax(logits[:, i])
        preds = probs.argmax(axis=1)
        m = evaluate_predictions(targets, preds, probs, train_counts)
        results['experts'][name] = m
        print(f"  {name:<18}{m['ba']:>8.4f}{m['head']:>8.4f}{m['medium']:>8.4f}"
              f"{m['tail']:>8.4f}{m['ece']:>8.4f}")

    # ── headroom ──
    headroom = HeadroomAnalyzer(logits, targets, expert_names=pool.loaded)
    summary = headroom.summary()
    results['headroom'] = summary
    print("\nRouting headroom")
    print(f"  all-wrong floor      : {summary['all_wrong_fraction']:.4f}")
    print(f"  oracle ceiling       : {summary['oracle_accuracy']:.4f}")
    print(f"  correctness counts   : {summary['correctness_counts']}")
    print("  pairwise Cohen's kappa:")
    for pair, kappa in summary['pairwise_kappa'].items():
        print(f"    {pair:<34}{kappa:>8.3f}")

    # ── the four frozen parameter-free rules ──
    print("\nFrozen parameter-free routing rules (pre-registered)")
    print(f"  {'rule':<14}{'BA':>8}{'Head':>8}{'Med':>8}{'Tail':>8}")
    for name, klass in ROUTERS.items():
        router = klass(expert_names=pool.loaded)
        preds = router.predict_class(logits)
        # ECE needs a probability proxy; use the routed experts' mean probs
        weights = router.predict_proba(logits)
        probs = np.einsum('ne,nec->nc', weights, softmax(logits))
        m = evaluate_predictions(targets, preds, probs, train_counts)
        results['routing'][name] = m
        print(f"  {name:<14}{m['ba']:>8.4f}{m['head']:>8.4f}{m['medium']:>8.4f}"
              f"{m['tail']:>8.4f}")

    out_path = Path(args.output or Path(args.checkpoint_dir) / 'test_evaluation.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except EvaluationError as exc:
        print(f"evaluation failed: {exc}", file=sys.stderr)
        sys.exit(2)
