#!/usr/bin/env python3
"""Run one explicit nested-OOF expert job.

Examples
--------
Short local smoke test (non-reportable)::

    python scripts/run_oof.py \
        --config configs/experts/ce.yaml --expert ce --seed 78 \
        --outer-fold 0 --inner-fold 0 --experiment-id task3b_smoke \
        --device cpu --epochs 1 --max-batches 1

Full Kaggle pilot (requires explicit authorization at execution time)::

    python scripts/run_oof.py \
        --config configs/experts/ce.yaml --expert ce --seed 78 \
        --outer-fold 0 --inner-fold 0 \
        --experiment-id task3b_pilot_ce_s78_o0_i0 \
        --device cuda --epochs 200 --execute-full

The command never constructs a test dataset.  It reads only the canonical
training index artifact and CIFAR-100's ``train=True`` population.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.nested_oof import OOFProtocolError, NestedOOFFoldManager  # noqa: E402
from scripts.config import ConfigError, TrainingConfig  # noqa: E402
from expert_method.oof.pipeline import (  # noqa: E402
    OOFArtifactError,
    OOFArtifactStore,
    OOFPipeline,
    OOFRunSpec,
    collect_resource_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and collect predictions for one nested-OOF expert run"
    )
    parser.add_argument("--config", required=False, help="expert YAML config")
    parser.add_argument("--expert", choices=("ce", "logit_adjusted", "balanced_softmax", "mixup"))
    parser.add_argument("--seed", type=int, default=78)
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument(
        "--inner-fold",
        type=int,
        default=None,
        help="inner prediction fold; omit for an outer expert/evaluation run",
    )
    parser.add_argument("--fold-seed", type=int, default=42)
    parser.add_argument("--experiment-id", required=False)
    parser.add_argument("--artifact-root", default="artifacts/oof")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default=None, choices=("auto", "cpu", "cuda"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="limit training batches for a non-reportable smoke test",
    )
    parser.add_argument(
        "--execute-full",
        action="store_true",
        help="required before a non-smoke run using the 200-epoch recipe",
    )
    parser.add_argument(
        "--resource-report",
        action="store_true",
        help="print local hardware information and exit",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.resource_report:
        import json

        print(json.dumps(collect_resource_report().to_dict(), indent=2, sort_keys=True))
        return 0

    missing = [
        name for name, value in (
            ("--config", args.config),
            ("--expert", args.expert),
            ("--experiment-id", args.experiment_id),
        ) if value is None
    ]
    if missing:
        parser.error("the following arguments are required unless --resource-report is used: " + ", ".join(missing))

    try:
        base_config = TrainingConfig.from_file(args.config)
        if base_config.expert != args.expert:
            raise OOFArtifactError(
                f"config expert {base_config.expert!r} does not match --expert {args.expert!r}"
            )
        data_root = args.data_root or base_config.data.root
        if args.data_root is not None:
            base_config = dataclasses.replace(
                base_config,
                data=dataclasses.replace(base_config.data, root=args.data_root),
            )
        effective_epochs = base_config.schedule.epochs if args.epochs is None else args.epochs
        if args.max_batches is None and effective_epochs >= 200 and not args.execute_full:
            raise OOFArtifactError(
                "refusing to launch a non-smoke 200-epoch OOF run without "
                "--execute-full; prepare the command and obtain explicit approval first"
            )

        manager = NestedOOFFoldManager.from_canonical_training_data(
            data_root,
            seed=args.fold_seed,
        )
        store = OOFArtifactStore(
            root=args.artifact_root,
            manager=manager,
            experiment_id=args.experiment_id,
        )
        spec = OOFRunSpec(
            experiment_id=args.experiment_id,
            expert=args.expert,
            training_seed=args.seed,
            outer_fold_id=args.outer_fold,
            inner_fold_id=args.inner_fold,
            device=args.device,
            epochs=args.epochs,
            max_batches=args.max_batches,
        )
        result = OOFPipeline(manager=manager, store=store).run(spec, base_config)
    except (ConfigError, OOFProtocolError, OOFArtifactError, OSError, RuntimeError) as exc:
        print(f"OOF run error: {exc}", file=sys.stderr)
        return 2

    print("OOF run complete (development artifact; not a test-set result)")
    print(f"  role: {result.context.role}")
    print(f"  expert: {result.context.expert_name} seed={result.context.training_seed}")
    print(f"  outer fold: {result.context.outer_fold_id} inner fold: {result.context.inner_fold_id}")
    print(f"  training samples: {len(result.context.training_indices)}")
    print(f"  prediction samples: {len(result.context.prediction_indices)}")
    print(f"  checkpoint: {result.checkpoint_path}")
    print(f"  checkpoint bytes: {result.checkpoint_path.stat().st_size:,}")
    print(f"  predictions: {result.prediction_path}")
    print(f"  prediction bytes: {result.prediction_path.stat().st_size:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
