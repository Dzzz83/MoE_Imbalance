#!/usr/bin/env python3
"""Run the Task 3F-A restricted OOF Ridge feasibility analysis."""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
# Task 3F-A is a CPU-only array analysis.  Avoid initializing an optional CUDA
# runtime while importing the existing training/provenance modules.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from scripts.task3f_ridge import run_task3f_ridge  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Task 3F-A Ridge predictability study on OOF inner folds 1–3"
        )
    )
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--oof-directory", default="artifacts/oof/task3c_oof")
    parser.add_argument(
        "--reference-diagnostics",
        default=None,
        help="Task 3C diagnostics path; defaults to <oof-directory>/diagnostics.json.",
    )
    parser.add_argument(
        "--output-directory",
        default="artifacts/oof/task3f_ridge",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        paths = run_task3f_ridge(
            data_root=args.data_root,
            oof_directory=args.oof_directory,
            reference_diagnostics=args.reference_diagnostics,
            output_directory=args.output_directory,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Task 3F-A error: {exc}", file=sys.stderr)
        return 2
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
