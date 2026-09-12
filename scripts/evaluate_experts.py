#!/usr/bin/env python3
"""
Evaluate the trained experts and measure routing headroom on the test set.

**This is the only entry point that reads the CIFAR-100 test set**, and it
appends an entry to `docs/test-access-log.md` before doing so. The candidate
routing rules were frozen in `records/routing-preregistration.md`; run this only
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
    EvaluationError, ExpertPool, HeadroomAnalyzer, aggregate_across_seeds as ev_aggregate,
    evaluate_predictions,
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


def select_logits(router_class, plain_logits, tta_logits):
    """Return the logits a rule must be evaluated on.

    A rule declaring ``requires_tta`` gets the view-averaged pass; every other
    rule gets the plain single-view pass. Centralised so no rule can silently be
    handed the wrong input — the bug that made the TTA row a duplicate of the
    Confidence row.
    """
    if getattr(router_class, 'requires_tta', False):
        if tta_logits is None:
            raise EvaluationError(
                f"{router_class.__name__} requires TTA logits but none were computed"
            )
        return tta_logits
    return plain_logits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', default='./checkpoints')
    parser.add_argument('--experts', nargs='+', default=DEFAULT_EXPERTS)
    parser.add_argument('--seeds', nargs='+', type=int, default=[78, 88, 1034])
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--tta-augs', type=int, default=10,
                        help='augmented views per sample for the TTA rule (pre-registered: 10)')
    parser.add_argument('--tta-seed', type=int, default=0,
                        help='seed for TTA view sampling, so the result is reproducible')
    parser.add_argument('--output', default=None,
                        help='JSON output path (default: checkpoints/test_evaluation.json)')
    parser.add_argument('--access-log', default='docs/test-access-log.md',
                        help='where to append the test-set access record')
    args = parser.parse_args(argv)

    # ── log the access BEFORE reading the test set ──
    logged = TestAccessLog(args.access_log).record(
        ' '.join(sys.argv),
        note=f"experts={','.join(args.experts)} seeds={','.join(map(str, args.seeds))}",
    )
    if not logged:
        print("WARNING: could not write the test-access log; continuing anyway")

    # The test set is read ONCE; every seed is evaluated on that same read, so
    # the access log records a single evaluation and all rules share one look.
    loader = build_test_loader(args.data_root, args.batch_size)
    train_counts = LongTailDataModule(root=args.data_root).class_counts()

    probe = ExpertPool(args.experts, args.seeds,
                       checkpoint_dir=args.checkpoint_dir, device=args.device)
    absent = probe.missing()
    if absent:
        print(f"Note: no checkpoint for {absent} — skipped")
    seeds = probe.seeds_present()
    if not seeds:
        raise EvaluationError(
            f"no checkpoints found in {args.checkpoint_dir} for {args.experts}"
        )
    print(f"Seeds found: {seeds}\n")

    per_seed: dict[int, dict] = {}
    targets = None

    for seed in seeds:
        pool = ExpertPool(args.experts, seeds=[seed],
                          checkpoint_dir=args.checkpoint_dir,
                          device=args.device).load(seed=seed)
        logits, targets = pool.logits(loader)          # (N, E, C)
        print(f"seed {seed}: {len(targets)} samples, logits {logits.shape}, "
              f"experts {pool.loaded}")

        entry: dict = {'experts': {}, 'routing': {}, 'headroom': {}}

        for i, name in enumerate(pool.loaded):
            probs = softmax(logits[:, i])
            entry['experts'][name] = evaluate_predictions(
                targets, probs.argmax(axis=1), probs, train_counts)

        headroom = HeadroomAnalyzer(logits, targets, expert_names=pool.loaded)
        entry['headroom'] = headroom.summary()

        # Rules defined over view-averaged logits need the TTA pass; it is
        # computed per seed so every rule sees a consistent input.
        tta_rules = [n for n, k in ROUTERS.items() if getattr(k, 'requires_tta', False)]
        logits_tta = None
        if tta_rules:
            logits_tta, _ = pool.logits(loader, n_augs=args.tta_augs,
                                        tta_seed=args.tta_seed)
        for name, klass in ROUTERS.items():
            router = klass(expert_names=pool.loaded)
            rule_logits = select_logits(klass, logits, logits_tta)
            preds = router.predict_class(rule_logits)
            weights = router.predict_proba(rule_logits)
            probs = np.einsum('ne,nec->nc', weights, softmax(rule_logits))
            entry['routing'][name] = evaluate_predictions(
                targets, preds, probs, train_counts)

        per_seed[seed] = entry
        m = entry['headroom']
        print(f"  all-wrong {m['all_wrong_fraction']:.4f}  oracle {m['oracle_accuracy']:.4f}")

    # ── aggregate across seeds (AGENTs.md section 6) ──
    def agg(section: str, name: str) -> dict:
        return ev_aggregate([per_seed[s][section][name] for s in seeds])

    print("\n" + "=" * 62)
    print(f"Per-expert performance (mean ± std over {len(seeds)} seed(s))")
    print(f"  {'expert':<18}{'BA':>16}{'Head':>16}{'Tail':>16}")
    expert_table = {}
    for name in per_seed[seeds[0]]['experts']:
        a = agg('experts', name)
        expert_table[name] = a
        print(f"  {name:<18}{a['ba']['mean']:>9.4f}±{a['ba']['std']:<6.4f}"
              f"{a['head']['mean']:>9.4f}±{a['head']['std']:<6.4f}"
              f"{a['tail']['mean']:>9.4f}±{a['tail']['std']:<6.4f}")

    print(f"\nFrozen parameter-free routing rules (pre-registered)")
    print(f"  {'rule':<14}{'BA':>16}{'Tail':>16}")
    routing_table = {}
    for name in per_seed[seeds[0]]['routing']:
        a = agg('routing', name)
        routing_table[name] = a
        print(f"  {name:<14}{a['ba']['mean']:>9.4f}±{a['ba']['std']:<6.4f}"
              f"{a['tail']['mean']:>9.4f}±{a['tail']['std']:<6.4f}")

    print("\nPre-registered decision rule (BA and Tail both above Uniform, "
          "consistently across seeds)")
    uniform = routing_table.get('Uniform')
    if uniform:
        for name, a in routing_table.items():
            if name == 'Uniform':
                continue
            ba_ok = a['ba']['mean'] > uniform['ba']['mean']
            tail_ok = a['tail']['mean'] > uniform['tail']['mean']
            verdict = 'PASSES' if (ba_ok and tail_ok) else 'no gain'
            print(f"  {name:<14} BA {'>' if ba_ok else '≤'} uniform, "
                  f"Tail {'>' if tail_ok else '≤'} uniform  -> {verdict}")

    results = {
        'seeds': seeds,
        'per_seed': per_seed,
        'experts_aggregate': expert_table,
        'routing_aggregate': routing_table,
    }

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
