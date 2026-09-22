#!/usr/bin/env python3
"""Run the read-only Task 3F-D combined-signal diagnostics."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from scripts.task3f_combined_signal_diagnostics import (  # noqa: E402
    run_task3f_combined_signal_diagnostics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Investigate combined inference-time Tail signals in frozen OOF data"
    )
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--oof-directory", default="artifacts/oof/task3c_oof")
    parser.add_argument("--ridge-directory", default="artifacts/oof/task3f_ridge")
    parser.add_argument(
        "--mixup-diagnostics-directory",
        default="artifacts/oof/task3f_mixup_diagnostics",
    )
    parser.add_argument(
        "--tail-signal-diagnostics-directory",
        default="artifacts/oof/task3f_tail_signal_diagnostics",
    )
    parser.add_argument(
        "--output-directory",
        default="artifacts/oof/task3f_combined_signal_diagnostics",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = run_task3f_combined_signal_diagnostics(
            data_root=args.data_root,
            oof_directory=args.oof_directory,
            ridge_directory=args.ridge_directory,
            mixup_diagnostics_directory=args.mixup_diagnostics_directory,
            tail_signal_diagnostics_directory=args.tail_signal_diagnostics_directory,
            output_directory=args.output_directory,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Task 3F-D error: {exc}", file=sys.stderr)
        return 2
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
