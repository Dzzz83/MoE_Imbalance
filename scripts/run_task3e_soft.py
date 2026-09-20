"""CLI for the Task 3E-B adaptive soft-mixture oracle feasibility study."""

from __future__ import annotations

import argparse
import os
import shlex
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.task3e_soft import run_task3e_soft


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the label-dependent Task 3E-B soft-mixture oracle analysis."
    )
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--oof-directory", default="artifacts/oof/task3c_oof")
    parser.add_argument(
        "--fixed-results",
        default="artifacts/oof/task3e_fixed_feasibility",
        help="Existing validated Task 3E-A output directory.",
    )
    parser.add_argument(
        "--reference-diagnostics",
        default=None,
        help="Task 3C diagnostics path; defaults to <oof-directory>/diagnostics.json.",
    )
    parser.add_argument(
        "--output-directory",
        default="artifacts/oof/task3e_soft_feasibility",
    )
    args = parser.parse_args()
    command = " ".join(shlex.quote(value) for value in [sys.executable, *sys.argv])
    try:
        paths = run_task3e_soft(
            data_root=args.data_root,
            oof_directory=args.oof_directory,
            fixed_results_directory=args.fixed_results,
            reference_diagnostics=args.reference_diagnostics,
            output_directory=args.output_directory,
            execution_command=command,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Task 3E-B error: {exc}", file=sys.stderr)
        return 2
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
