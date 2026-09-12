#!/usr/bin/env python3
"""
How does ensemble accuracy depend on how many experts are included?

Evaluates a combination rule over **every subset** of the expert pool and reports
accuracy per ensemble size, which answers "why 2, why 3, why 4?" empirically.

Reads the CIFAR-100 test set, so it appends to the access log like every other
evaluation entry point. The test set is read once; all sizes are computed from
that single pass.

Usage:
    python scripts/analyze_subsets.py --seeds 78 88 1034
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

import torch

from data.lt_datamodule import LongTailDataModule
from scripts.evaluate_experts import build_test_loader
from scripts.evaluation import EvaluationError, ExpertPool
from scripts.router import ROUTERS
from scripts.subsets import (
    SELECTION_WARNING, SubsetEnsembleAnalysis, aggregate_size_tables,
)
from scripts.utils.test_access import TestAccessLog

DEFAULT_EXPERTS = ['CE', 'LAL', 'BalancedSoftmax', 'Mixup']


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', default='./checkpoints')
    parser.add_argument('--experts', nargs='+', default=DEFAULT_EXPERTS)
    parser.add_argument('--seeds', nargs='+', type=int, default=[78, 88, 1034])
    parser.add_argument('--data-root', default='./data')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--access-log', default='docs/test-access-log.md')
    parser.add_argument('--output', default='checkpoints/subset_analysis.json')
    args = parser.parse_args(argv)

    TestAccessLog(args.access_log).record(
        ' '.join(sys.argv),
        note=f"subset analysis experts={','.join(args.experts)} "
             f"seeds={','.join(map(str, args.seeds))}",
    )

    loader = build_test_loader(args.data_root, args.batch_size)
    train_counts = LongTailDataModule(root=args.data_root).class_counts()

    probe = ExpertPool(args.experts, args.seeds,
                       checkpoint_dir=args.checkpoint_dir, device=args.device)
    seeds = probe.seeds_present()
    if not seeds:
        raise EvaluationError(f"no checkpoints in {args.checkpoint_dir}")

    per_rule: dict[str, list[dict]] = {name: [] for name in ROUTERS}
    results: dict = {'seeds': seeds, 'rules': {}}

    for seed in seeds:
        pool = ExpertPool(args.experts, seeds=[seed],
                          checkpoint_dir=args.checkpoint_dir,
                          device=args.device).load(seed=seed)
        logits, targets = pool.logits(loader)
        print(f"seed {seed}: {len(targets)} samples, experts {pool.loaded}")
        analysis = SubsetEnsembleAnalysis(
            logits, targets, expert_names=pool.loaded, class_counts=train_counts)
        for name, klass in ROUTERS.items():
            per_rule[name].append(analysis.size_table(klass))

    print("\n" + "=" * 78)
    print("Ensemble accuracy vs number of experts (uniform logit averaging)")
    print(f"  {'k':>2}{'subsets':>9}{'mean BA':>18}{'best BA':>18}  best subset")

    uniform_tables = per_rule['Uniform']
    aggregated = aggregate_size_tables(uniform_tables)
    for size, row in aggregated.items():
        n = row['n_subsets']
        best_names = uniform_tables[0][size]['best_names']
        print(f"  {size:>2}{n:>9}"
              f"{row['mean_ba']['mean']:>11.4f}±{row['mean_ba']['std']:<6.4f}"
              f"{row['best_ba']['mean']:>11.4f}±{row['best_ba']['std']:<6.4f}"
              f"  {'+'.join(best_names)}")

    print("\nFor comparison, the same table under probability averaging")
    print(f"  {'k':>2}{'mean BA':>18}")
    prob_agg = aggregate_size_tables(per_rule['Probability'])
    for size, row in prob_agg.items():
        print(f"  {size:>2}{row['mean_ba']['mean']:>11.4f}±{row['mean_ba']['std']:<6.4f}")

    print(f"\nNOTE: {SELECTION_WARNING}")

    results['rules'] = {
        name: {str(k): v for k, v in aggregate_size_tables(tables).items()}
        for name, tables in per_rule.items()
    }
    results['best_subsets_per_seed'] = {
        name: {str(k): [t[k]['best_names'] for t in tables]
               for k in tables[0]}
        for name, tables in per_rule.items()
    }
    results['selection_warning'] = SELECTION_WARNING

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Wrote {out}")
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except EvaluationError as exc:
        print(f"analysis failed: {exc}", file=sys.stderr)
        sys.exit(2)
