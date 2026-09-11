#!/usr/bin/env python3
"""
Health-check finished training runs against the published protocol.

Checks each run's `*_history.json`: full epoch budget, the LR actually decaying
to x0.01 at epoch 161 and x0.0001 at 181, no non-finite loss, a loss that
descended, and the presence of the final checkpoint.

This never touches the test set.

Usage:
    python scripts/check_runs.py
    python scripts/check_runs.py --seeds 78 88 1034 --checkpoint-dir ./checkpoints
"""

from __future__ import annotations

import os
import sys

_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import argparse
from pathlib import Path

from scripts.evaluation import EvaluationError, RunHealthChecker, load_history

#: Expert labels as written by the trainers, in reporting order.
DEFAULT_EXPERTS = ['CE', 'LAL', 'BalancedSoftmax', 'Mixup']
DEFAULT_SEEDS = [78, 88, 1034]


def find_history_file(checkpoint_dir: Path, expert: str, seed: int) -> Path:
    """The history file a run writes: `{expert}_seed{seed}_history.json`."""
    direct = checkpoint_dir / f'{expert}_seed{seed}_history.json'
    if direct.exists():
        return direct
    matches = sorted(checkpoint_dir.glob(f'{expert}*seed{seed}*history.json'))
    if matches:
        return matches[0]
    raise EvaluationError(f"no history file for {expert} seed={seed} in {checkpoint_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-dir', default='./checkpoints')
    parser.add_argument('--experts', nargs='+', default=DEFAULT_EXPERTS)
    parser.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    parser.add_argument('--expected-epochs', type=int, default=200)
    args = parser.parse_args(argv)

    checkpoint_dir = Path(args.checkpoint_dir)
    print(f"Checking runs in {checkpoint_dir} "
          f"({len(args.experts)} experts x {len(args.seeds)} seeds)\n")

    checked = failed = missing = 0
    for expert in args.experts:
        for seed in args.seeds:
            try:
                history = load_history(find_history_file(checkpoint_dir, expert, seed))
            except EvaluationError as exc:
                print(f"  · {expert} seed={seed}: not found — {exc}")
                missing += 1
                continue

            report = RunHealthChecker(
                expert, seed,
                checkpoint_dir=checkpoint_dir,
                expected_epochs=args.expected_epochs,
            ).check(history)
            checked += 1
            mark = 'OK  ' if report.ok else 'FAIL'
            print(f"  [{mark}] {report}")
            for problem in report.problems:
                print(f"           - {problem}")
            if not report.ok:
                failed += 1

    print(f"\n{checked} run(s) checked, {failed} with problems, {missing} not found")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
