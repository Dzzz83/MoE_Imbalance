"""Deprecated command adapters for the pre-config-driven script CLIs.

The adapters retain the old flags while routing work through the same study,
bundle, and analysis services as ``python -m expert_method``.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Sequence

from expert_method.config import ConfigError, RuntimeProfile, load_runtime_profile, load_study


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_STUDY = _PROJECT_ROOT / "configs" / "studies" / "ridge_sinkhorn_3seed_v1.yaml"
_KAGGLE_PROFILE = _PROJECT_ROOT / "configs" / "profiles" / "kaggle.yaml"
_CONFIG_ROOT = _PROJECT_ROOT / "configs"


def _reuse_roots(values: Sequence[str]) -> dict[str, str]:
    from expert_method.ridge_sinkhorn.matrix import parse_reuse_roots

    return {name: str(path) for name, path in parse_reuse_roots(values).items()}


def _runtime_profile(
    *,
    data_root: str,
    artifact_root: str,
    reuse_root: Sequence[str],
    shard_index: int = 0,
    shard_count: int = 1,
    max_jobs: int | None = None,
    device: str | None = None,
) -> RuntimeProfile:
    profile = load_runtime_profile(_KAGGLE_PROFILE)
    return replace(
        profile,
        profile_id="deprecated-script-adapter",
        data_root=data_root,
        run_root=artifact_root,
        reuse_roots=_reuse_roots(reuse_root),
        shard_index=shard_index,
        shard_count=shard_count,
        max_jobs=profile.max_jobs if max_jobs is None else max_jobs,
        device=profile.device if device is None else device,
    )


def parse_matrix_reuse_root(value: str, roots: dict[str, Path]) -> None:
    """Compatibility helper shared with the former analysis script."""
    parsed = _reuse_roots((value,))
    for name, path in parsed.items():
        existing = roots.get(name)
        candidate = Path(path)
        if existing is not None and existing != candidate:
            raise argparse.ArgumentTypeError(f"reuse-root logical name is ambiguous: {name!r}")
        roots[name] = candidate


def build_matrix_parser() -> argparse.ArgumentParser:
    """Build the legacy matrix parser without owning execution behavior."""
    from expert_method.ridge_sinkhorn.matrix import STUDY_ID

    parser = argparse.ArgumentParser(
        description="Deprecated adapter; use python -m expert_method study/bundle commands"
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan", action="store_true", help="read-only job preview")
    modes.add_argument("--run-missing", action="store_true", help="run a bounded missing-job batch")
    modes.add_argument("--export-bundle", metavar="PATH")
    modes.add_argument("--import-bundle", metavar="PATH")
    modes.add_argument("--merge-bundles", nargs="+", metavar="PATH")
    parser.add_argument("--stage", choices=("inner", "outer"))
    parser.add_argument("--study-id", required=True, help=f"must be {STUDY_ID}")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--artifact-root", default="artifacts/oof")
    parser.add_argument("--reuse-root", action="append", default=[], metavar="[NAME=]PATH")
    parser.add_argument("--config-root", default="configs")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--execute-full", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser


def matrix_main(argv: Sequence[str] | None = None) -> int:
    """Translate old matrix flags into the config-driven package services."""
    parser = build_matrix_parser()
    args = parser.parse_args(argv)
    print(
        "warning: scripts/run_ridge_sinkhorn_matrix.py is deprecated; "
        "use python -m expert_method",
        file=sys.stderr,
    )
    try:
        from expert_method.ridge_sinkhorn.matrix import STUDY_ID

        if args.study_id != STUDY_ID:
            raise ConfigError(f"--study-id is frozen at {STUDY_ID!r}")
        if Path(args.config_root).expanduser().resolve() != _CONFIG_ROOT.resolve():
            raise ConfigError("expert recipes are defined by the canonical study YAML; --config-root cannot override them")
        if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
            raise ConfigError("shard selection requires 0 <= --shard-index < --shard-count")
        if args.max_jobs is not None and args.max_jobs < 1:
            raise ConfigError("--max-jobs must be positive")
        study = load_study(_STUDY)
        profile = _runtime_profile(
            data_root=args.data_root,
            artifact_root=args.artifact_root,
            reuse_root=args.reuse_root,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            max_jobs=args.max_jobs,
            device=args.device,
        )

        from expert_method import cli, workflow

        if args.import_bundle or args.merge_bundles:
            paths = [args.import_bundle] if args.import_bundle else args.merge_bundles
            result = workflow.restore_bundles(study, profile, paths=paths)
        elif args.plan:
            if args.stage is None:
                parser.error("--stage inner|outer is required for --plan")
            # Historical matrix --plan froze the study inventory. Keep that
            # behavior isolated to this deprecated adapter; the package CLI's
            # study plan remains a read-only preview.
            freeze_summary = cli._freeze_study(study, profile)
            if args.stage == "outer":
                freeze = cli._read_freeze(cli._freeze_path(profile, study))
                manager = cli._load_manager(study, profile)
                planner = cli._make_planner(
                    study,
                    profile,
                    manager,
                    read_only=False,
                    freeze_sha256=freeze["freeze_sha256"],
                )
                # Do not compute or persist the outer reuse audit until every
                # method lock and its inner-source references have validated.
                planner.validate_complete_lock_matrix(frozen=planner.frozen_manifest())
                planner.freeze(stage="outer", freeze_sha256=freeze["freeze_sha256"])
            result = cli._plan_payload(study, profile, stage=args.stage)
            result["legacy_freeze"] = freeze_summary
        elif args.run_missing:
            if args.stage is None:
                parser.error("--stage inner|outer is required for --run-missing")
            result = workflow.run_batch(
                study, profile, stage=args.stage, max_jobs=args.max_jobs,
                confirmed_full_run=args.execute_full,
            )
        elif args.export_bundle:
            if args.stage is None:
                parser.error("--stage inner|outer is required for --export-bundle")
            result = workflow.export_stage_bundle(
                study, profile, stage=args.stage, path=args.export_bundle
            )
        else:
            parser.error("choose one of --plan, --run-missing, --export-bundle, --import-bundle, or --merge-bundles")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result.get("success", True) else 1
    except (ConfigError, OSError, RuntimeError, ValueError) as exc:
        print(f"Ridge--Sinkhorn matrix error: {exc}", file=sys.stderr)
        return 2


def build_analysis_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Deprecated adapter; use python -m expert_method study lock/evaluate/report"
    )
    parser.add_argument("--stage", choices=("lock", "evaluate", "report"), required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--artifact-root", default="artifacts/oof")
    parser.add_argument("--reuse-root", action="append", default=[], metavar="[NAME=]PATH")
    return parser


def analysis_main(argv: Sequence[str] | None = None) -> int:
    """Translate the old analysis flags into the package stage service."""
    parser = build_analysis_parser()
    args = parser.parse_args(argv)
    print(
        "warning: scripts/run_ridge_sinkhorn_3seed.py is deprecated; "
        "use python -m expert_method",
        file=sys.stderr,
    )
    try:
        study = load_study(_STUDY)
        profile = _runtime_profile(
            data_root=args.data_root,
            artifact_root=args.artifact_root,
            reuse_root=args.reuse_root,
        )
        from expert_method.workflow import run_analysis_stage

        result = run_analysis_stage(study, profile, stage=args.stage)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ConfigError, OSError, RuntimeError, ValueError) as exc:
        print(f"Ridge/Sinkhorn three-seed study error: {exc}", file=sys.stderr)
        return 2
