#!/usr/bin/env python3
"""Run the Task 3E-A fixed-weight feasibility analysis."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.task3e_fixed import run_task3e_fixed  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen Task 3E-A fixed-weight grid on OOF inner folds 1–3"
    )
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--oof-directory", default="artifacts/oof/task3c_oof")
    parser.add_argument("--reference-diagnostics", default=None)
    parser.add_argument(
        "--output-directory",
        default="artifacts/oof/task3e_fixed_feasibility",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = run_task3e_fixed(
            data_root=args.data_root,
            oof_directory=args.oof_directory,
            reference_diagnostics=args.reference_diagnostics,
            output_directory=args.output_directory,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Task 3E-A error: {exc}", file=sys.stderr)
        return 2
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
