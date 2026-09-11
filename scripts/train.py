#!/usr/bin/env python3
"""
Config-driven training entry point for CIFAR-100-LT experts.

Every expert is described by a YAML file under ``configs/``; nothing about a run
is decided in this script. One entry point, one code path, four configs.

Usage:
    python scripts/train.py --config configs/ce.yaml
    python scripts/train.py --config configs/lal.yaml --seed 42
    python scripts/train.py --config configs/mixup.yaml --device cpu --max-batches 2  # dry run

The final-epoch checkpoint is the reported model; checkpoints at every 20th epoch
from epoch 160 exist for inspection only and must never be selected on test
accuracy. There is no validation split (see docs/specs/training-protocol.md).
"""

from __future__ import annotations

import os
import sys

_proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import argparse
from pathlib import Path

import yaml

from data.lt_datamodule import LongTailDataModule
from scripts.config import ConfigError, TrainingConfig
from scripts.trainers import build_trainer


def build_datamodule(
    config: TrainingConfig, max_batches: int | None = None
) -> LongTailDataModule:
    """Adapt a TrainingConfig's data section into a data module."""
    return LongTailDataModule(
        root=config.data.root,
        imbalance_ratio=config.data.imbalance_ratio,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        max_batches=max_batches,
    )


def record_config(config: TrainingConfig) -> Path:
    """Write the effective config next to the run's checkpoints."""
    out_dir = Path(config.checkpoint.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{config.expert}_seed{config.seed}_config.yaml"
    path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False))
    return path


def is_kaggle_environment() -> bool:
    """True when this process is running inside a Kaggle kernel.

    Kaggle is the *intended* place for a full training sweep, so the local-GPU
    warning below must not fire there. It fired on Kaggle once and told the user
    that full training belongs on Kaggle while they were already on Kaggle.
    """
    for var in ('KAGGLE_KERNEL_RUN_TYPE', 'KAGGLE_URL_BASE', 'KAGGLE_DATA_PROXY_TOKEN'):
        if os.environ.get(var):
            return True
    try:
        return str(Path.cwd()).startswith('/kaggle')
    except OSError:
        return False


def full_run_on_local_gpu(
    config: TrainingConfig, max_batches: int | None
) -> bool:
    """True when a *reportable* run would execute on the workspace machine's GPU.

    The workspace GPU exists for verification only — the 12-run sweep belongs on
    Kaggle — so a non-dry run that lands on it is worth flagging. This is
    advisory: it never blocks or delays a run. Kaggle itself is never flagged.
    """
    if max_batches is not None:
        return False
    if is_kaggle_environment():
        return False
    return config.resolved_device.startswith('cuda')


def describe(config: TrainingConfig) -> str:
    s, o, c, d = config.schedule, config.optimiser, config.checkpoint, config.data
    return (
        f"expert={config.expert} seed={config.seed} device={config.resolved_device}\n"
        f"  schedule: {s.epochs} epochs, warmup {s.warmup_epochs}, "
        f"decay {tuple(s.decay_epochs)} -> {tuple(s.decay_factors)}\n"
        f"  optimiser: {o.name} lr={o.lr} momentum={o.momentum} "
        f"weight_decay={o.weight_decay} nesterov={o.nesterov}\n"
        f"  data: {d.root} batch={d.batch_size} IR={d.imbalance_ratio}\n"
        f"  checkpoints: {c.dir} (final epoch only)"
    )


def run_training(
    config: TrainingConfig, max_batches: int | None = None
) -> list[dict]:
    """Train one expert from a resolved config. Returns the per-epoch history."""
    print(describe(config))

    datamodule = build_datamodule(config, max_batches=max_batches)
    class_counts = datamodule.class_counts()
    print(f"  data: {datamodule.describe()}")

    config_path = record_config(config)
    print(f"  recorded config -> {config_path}")

    trainer = build_trainer(config, class_counts, device=config.resolved_device)
    history = trainer.train(datamodule.train_loader())
    trainer.save_history()
    return history


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Train a CIFAR-100-LT expert from a config file'
    )
    parser.add_argument('--config', required=True,
                        help='path to a config file, e.g. configs/ce.yaml')
    parser.add_argument('--seed', type=int, default=None,
                        help='override the config seed (one LT split, 3 training seeds)')
    parser.add_argument('--device', default=None,
                        help="override device: 'auto', 'cpu' or 'cuda'")
    parser.add_argument('--epochs', type=int, default=None,
                        help='override the epoch budget')
    parser.add_argument('--checkpoint-dir', default=None)
    parser.add_argument('--max-batches', type=int, default=None,
                        help='DRY RUN only: expose the first N batches (AGENTs.md gate)')
    args = parser.parse_args(argv)

    try:
        config = TrainingConfig.from_file(args.config).replace(
            seed=args.seed,
            device=args.device,
            epochs=args.epochs,
            checkpoint_dir=args.checkpoint_dir,
        )
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.max_batches is not None:
        print(f"[DRY RUN] limiting to {args.max_batches} batches — the resulting "
              f"model is NOT a reportable result")
    elif full_run_on_local_gpu(config, args.max_batches):
        print("!" * 72)
        print("WARNING: this is a REPORTABLE run executing on the LOCAL GPU.")
        print("  The local GPU is for verification only; full training belongs on")
        print("  Kaggle. If you only meant to check that the code runs, add")
        print("  --max-batches 2.")
        print("!" * 72)

    run_training(config, max_batches=args.max_batches)
    return 0


if __name__ == '__main__':
    sys.exit(main())
