#!/usr/bin/env python3
"""Run the read-only Task 3F-F Ridge feature comparison."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from scripts.task3f_feature_comparison import (  # noqa: E402
    run_task3f_feature_comparison,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare the existing Task 3F-A Ridge feature sets on OOF diagnostics"
    )
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--oof-directory", default="artifacts/oof/task3c_oof")
    parser.add_argument("--ridge-directory", default="artifacts/oof/task3f_ridge")
    parser.add_argument(
        "--target-directory",
        default="artifacts/oof/task3f_target_diagnostics",
    )
    parser.add_argument(
        "--output-directory",
        default="artifacts/oof/task3f_feature_comparison",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = run_task3f_feature_comparison(
            data_root=args.data_root,
            oof_directory=args.oof_directory,
            ridge_directory=args.ridge_directory,
            target_directory=args.target_directory,
            output_directory=args.output_directory,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Task 3F-F error: {exc}", file=sys.stderr)
        return 2
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
