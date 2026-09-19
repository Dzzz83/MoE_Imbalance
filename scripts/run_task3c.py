#!/usr/bin/env python3
"""Incremental Task 3C OOF execution and diagnostics.

The runner is intentionally resumable. A dry run or validation scan never
trains. ``--run-missing`` executes only missing/partial non-pilot jobs through
the existing :class:`scripts.oof_pipeline.OOFPipeline`; a validated completed
job is skipped. The CE Task 3B pilot is always referenced at its original path
and is never copied, rewritten, or retrained.

Typical Kaggle use from the cloned repository is::

    cd /kaggle/working/MoE_Imbalance
    python scripts/run_task3c.py \
        --dry-run \
        --device cuda \
        --data-root ./data \
        --artifact-root ./artifacts/oof \
        --pilot-root ./artifacts/oof/task3b_pilot_ce_s78_o0_i0
    python scripts/run_task3c.py --validate-only
    python scripts/run_task3c.py --run-missing --max-jobs 1 --execute-full \
        --device cuda
    python scripts/run_task3c.py --build-diagnostics --device cuda

The preflight deliberately reports missing jobs as a successful dry-run
inventory.  ``--validate-only`` returns a nonzero status while any required
job is missing; that status means incomplete work, not corrupted artifacts.
All Task 3C outputs stay under ``./artifacts/oof/task3c_oof`` so an operator
can validate them and explicitly commit/push them from the repository root.

The last two commands may be rerun after a session interruption. Results are
stored under ``artifacts/oof/task3c_oof``. This library never runs ``git push``;
validated artifacts must be persisted explicitly by the operator.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.task3c_oof import (  # noqa: E402
    OOFAlignmentBuilder,
    TASK3C_EXPERIMENT_ID,
    TASK3C_PILOT_DEFAULT,
    Task3CBatchPlan,
    Task3CBatchRunner,
    Task3CError,
    Task3CDiagnosticReporter,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen Task 3C OOF batch incrementally"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--dry-run", action="store_true", help="scan and print the 16-job plan")
    action.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the pilot and every already-present Task 3C run",
    )
    action.add_argument(
        "--run-missing",
        action="store_true",
        help="execute missing/partial non-pilot jobs incrementally",
    )
    action.add_argument(
        "--build-diagnostics",
        action="store_true",
        help="validate all jobs, align their logits, and write diagnostics",
    )
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--config-root", default="configs")
    parser.add_argument("--artifact-root", default="artifacts/oof")
    parser.add_argument("--pilot-root", default=TASK3C_PILOT_DEFAULT)
    parser.add_argument("--experiment-id", default=TASK3C_EXPERIMENT_ID)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="maximum number of missing/partial jobs for this session",
    )
    parser.add_argument(
        "--execute-full",
        action="store_true",
        help="authorize the frozen 200-epoch jobs (required by --run-missing)",
    )
    parser.add_argument(
        "--write-plan",
        action="store_true",
        help="persist batch_manifest.json during a dry run",
    )
    return parser


def _build_runner(args: argparse.Namespace) -> Task3CBatchRunner:
    plan = Task3CBatchPlan(
        data_root=args.data_root,
        config_root=args.config_root,
        artifact_root=args.artifact_root,
        pilot_root=args.pilot_root,
        experiment_id=args.experiment_id,
        device=args.device,
    )
    return Task3CBatchRunner(plan)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        runner = _build_runner(args)
        if args.dry_run:
            statuses = runner.inspect()
            runner.print_plan(statuses)
            if args.write_plan:
                print(f"batch manifest: {runner.plan.write_batch_manifest()}")
            return 0

        if args.validate_only:
            statuses = runner.inspect()
            runner.print_plan(statuses)
            return int(any(status.state != "validated_existing" for status in statuses))

        if args.run_missing:
            statuses = runner.run_missing(
                max_jobs=args.max_jobs,
                execute_full=args.execute_full,
            )
            runner.print_plan(statuses)
            return 0

        # --build-diagnostics
        if args.max_jobs is not None:
            raise Task3CError("--max-jobs is only valid with --run-missing")
        runs = runner.validated_runs()
        dataset = OOFAlignmentBuilder(runner.plan).build(runs)
        arrays_path, metadata_path = dataset.save(
            runner.store.base_dir,
            manager=runner.plan.manager,
        )
        report_path = Task3CDiagnosticReporter(runner.plan).write(
            dataset, runner.store.base_dir
        )
        print(f"aligned OOF arrays: {arrays_path}")
        print(f"aligned OOF metadata: {metadata_path}")
        print(f"diagnostic report: {report_path}")
        print(
            "primary diagnostics use router-fitting inner folds 1–3; "
            "full-development diagnostics disclose inner-fold-0 label use"
        )
        return 0
    except (Task3CError, OSError, RuntimeError, ValueError) as exc:
        print(f"Task 3C error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
