"""Command line seams for validating, planning, freezing, and inspecting studies."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence

from expert_method.attempts import summarize_attempts
from expert_method.config import ConfigError, RuntimeProfile, StudyDefinition, load_runtime_profile, load_study


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_STUDY = _PROJECT_ROOT / "configs" / "studies" / "ridge_sinkhorn_3seed_v1.yaml"
_DEFAULT_PROFILE = _PROJECT_ROOT / "configs" / "profiles" / "local-smoke.yaml"
_FREEZE_SCHEMA = "expert_method.study_freeze.v1"
_SHARD_RULE = "unsigned_big_endian(first_8_bytes(SHA-256(job_id))) % shard_count"


def build_parser() -> argparse.ArgumentParser:
    """Build the public module CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m expert_method",
        description="Validate and manage reproducible experiment studies",
    )
    parser.add_argument("--config", default=str(_DEFAULT_STUDY), help="scientific study YAML")
    parser.add_argument("--profile", default=str(_DEFAULT_PROFILE), help="runtime profile YAML")
    groups = parser.add_subparsers(dest="group", required=True)
    study = groups.add_parser("study", help="study validation, planning, and provenance")
    study_commands = study.add_subparsers(dest="study_command", required=True)
    study_commands.add_parser("validate", help="validate study and runtime YAML without writing files")
    plan = study_commands.add_parser("plan", help="read-only artifact and reuse preview")
    plan.add_argument("--stage", choices=("inner", "outer"), default="inner")
    study_commands.add_parser("freeze", help="record committed study and reuse identities")
    study_commands.add_parser("status", help="summarize frozen study progress and next action")
    study_commands.add_parser("doctor", help="read-only dependency, Git, path, disk, and CUDA preflight")
    run = study_commands.add_parser("run", help="execute a bounded batch of missing expert jobs")
    run.add_argument("--stage", choices=("inner", "outer"), required=True)
    run.add_argument("--max-jobs", type=int, help="optional lower limit than the profile maximum")
    run.add_argument("--execute-full", action="store_true", help="confirm full 200-epoch expert training")
    session = study_commands.add_parser("session", help="restore, run a bounded batch, revalidate, and export")
    session.add_argument("--stage", choices=("inner", "outer"), required=True)
    session.add_argument("--execute-full", action="store_true", help="confirm full 200-epoch expert training")
    for command in ("lock", "evaluate", "report"):
        study_commands.add_parser(command, help=f"run the explicit {command} stage")

    bundle = groups.add_parser("bundle", help="restore and export portable experiment artifacts")
    bundle_commands = bundle.add_subparsers(dest="bundle_command", required=True)
    restore = bundle_commands.add_parser("restore", help="restore configured or explicitly named bundles")
    restore.add_argument("paths", nargs="*", help="bundle paths; defaults to profile.bundle_inputs")
    export = bundle_commands.add_parser("export", help="export validated artifacts for the configured shard")
    export.add_argument("--stage", choices=("inner", "outer"), default="inner")
    export.add_argument("--kind", choices=("matrix", "locks"), default="matrix")
    export.add_argument("--path", help="override the profile bundle output path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one public command and emit machine-readable JSON."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        study = load_study(args.config)
        profile = load_runtime_profile(args.profile)
        if args.group == "study" and args.study_command == "validate":
            _emit(_validate_payload(study, profile))
            return 0
        if args.group == "study" and args.study_command == "plan":
            _emit(_plan_payload(study, profile, stage=args.stage))
            return 0
        if args.group == "study" and args.study_command == "freeze":
            payload = _freeze_study(study, profile)
            _emit(payload)
            return 0
        if args.group == "study" and args.study_command == "status":
            _emit(_status_payload(study, profile))
            return 0
        if args.group == "study" and args.study_command == "doctor":
            payload = _doctor_payload(study, profile)
            _emit(payload)
            return 0 if payload["ok"] else 1
        if args.group == "study" and args.study_command in {"run", "session"}:
            if not args.execute_full:
                raise ConfigError(
                    f"study {args.study_command} launches the frozen 200-epoch recipe; "
                    "explicitly confirm with --execute-full"
                )
            from expert_method.workflow import run_batch, run_session

            payload = (
                run_batch(
                    study, profile, stage=args.stage, max_jobs=args.max_jobs,
                    confirmed_full_run=True,
                )
                if args.study_command == "run"
                else run_session(
                    study, profile, stage=args.stage, confirmed_full_run=True
                )
            )
            _emit(payload)
            return 0 if payload.get("success", True) else 1
        if args.group == "study" and args.study_command in {"lock", "evaluate", "report"}:
            from expert_method.workflow import run_analysis_stage

            _emit(run_analysis_stage(study, profile, stage=args.study_command))
            return 0
        if args.group == "bundle" and args.bundle_command == "restore":
            from expert_method.workflow import restore_bundles

            _emit(restore_bundles(study, profile, paths=args.paths))
            return 0
        if args.group == "bundle" and args.bundle_command == "export":
            from expert_method.workflow import export_lock_bundle, export_stage_bundle

            payload = (
                export_lock_bundle(study, profile, path=args.path)
                if args.kind == "locks"
                else export_stage_bundle(study, profile, stage=args.stage, path=args.path)
            )
            _emit(payload)
            return 0
        parser.error("unsupported command")
    except (ConfigError, ValueError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


def _validate_payload(study: StudyDefinition, profile: RuntimeProfile) -> dict[str, Any]:
    """Summarize the validated inputs without touching data or output paths."""
    return {
        "valid": True,
        "study_id": study.study_id,
        "scientific_sha256": study.scientific_sha256,
        "expert_config_sha256": _expert_config_hashes(study),
        "profile_id": profile.profile_id,
        "runtime": {
            "data_root": profile.data_root,
            "run_root": profile.run_root,
            "reuse_root_names": sorted(profile.reuse_roots),
            "device": profile.device,
            "shard": {"index": profile.shard_index, "count": profile.shard_count},
            "max_jobs": profile.max_jobs,
        },
    }


def _plan_payload(study: StudyDefinition, profile: RuntimeProfile, *, stage: str) -> dict[str, Any]:
    """Inspect artifacts and reuse roots using a planner that cannot write."""
    freeze_path = _freeze_path(profile, study)
    frozen = _read_freeze(freeze_path) if freeze_path.exists() or freeze_path.is_symlink() else None
    if frozen is not None:
        source = _source_identity(study.source_path.parents[2])
        _validate_frozen_identity(study, profile, frozen, source)
        if sorted(profile.reuse_roots) != frozen.get("runtime_root_names"):
            raise ConfigError("named reuse roots differ from the frozen study configuration")
    planner = _make_planner(
        study, profile, _load_manager(study, profile),
        freeze_sha256=frozen.get("freeze_sha256") if frozen is not None else None,
    )
    if stage == "outer":
        # Make the public preview seam enforce the outer protocol before
        # entering general planning or reuse-audit code. The planner repeats
        # this guard for non-CLI callers as defense in depth.
        planner.validate_complete_lock_matrix(frozen=planner.frozen_manifest())
    if frozen is not None and planner.frozen_manifest()["fold_manifest_sha256"] != frozen.get("fold_manifest_sha256"):
        raise ConfigError("canonical fold membership differs from the frozen study")
    result = planner.plan(
        stage=stage,
        shard_index=profile.shard_index,
        shard_count=profile.shard_count,
        max_jobs=profile.max_jobs,
        output_path=profile.run_root,
    )
    if frozen is not None and stage == "inner":
        _verify_reuse_audit(result.compatibility_table, frozen["reuse_decisions"]["inner"], stage)
    all_jobs = tuple(job for job in planner.inventory if job.stage == stage)
    identity = planner.frozen_manifest()
    summary = result.to_dict()
    summary.pop("jobs", None)
    summary["reuse_audit"] = [dict(row) for row in result.compatibility_table]
    summary["invalid_job_ids"] = [
        status.job.job_id for status in result.statuses if status.state == "invalid"
    ]
    if result.invalid_count:
        next_action = "inspect invalid artifacts; immutable runs must not be overwritten"
    elif result.missing_count:
        next_action = "restore or execute the reported missing jobs"
    else:
        next_action = "create method locks before outer work" if stage == "inner" else "study evaluate"
    return {
        **summary,
        "scientific_sha256": study.scientific_sha256,
        "source_commit": identity["source_commit"],
        "source_tree_dirty": identity["source_tree_dirty"],
        "plan_sha256": identity["plan_sha256"],
        "protocol_config_sha256": identity["protocol_config_sha256"],
        "stage_inventory_count": len(all_jobs),
        "full_inventory_count": len(planner.inventory),
        "available_output_bytes": result.available_output_bytes,
        "freeze_path": str(freeze_path),
        "frozen": frozen is not None,
        "read_only": True,
        "next_action": next_action,
    }


def _freeze_study(study: StudyDefinition, profile: RuntimeProfile) -> dict[str, Any]:
    """Write one immutable study inventory after confirming a clean Git source."""
    repository_root = study.source_path.parents[2]
    source = _source_identity(repository_root)
    if not source["commit"] or source["dirty"]:
        raise ConfigError("study freeze requires a valid HEAD and a clean committed source tree")
    plan_path = repository_root / "docs" / "PLAN.md"
    if not plan_path.is_file():
        raise ConfigError(f"study plan is missing: {plan_path}")

    planner = _make_planner(
        study, profile, _load_manager(study, profile), read_only=False
    )
    legacy_identity = planner.frozen_manifest()
    inventory = _inventory_payload(planner.inventory)
    inner_jobs = tuple(job for job in planner.inventory if job.stage == "inner")
    reuse_rows = [dict(row) for row in planner.audit_compatibility(inner_jobs)]
    expert_hashes = _expert_config_hashes(study)
    payload: dict[str, Any] = {
        "schema_version": _FREEZE_SCHEMA,
        "study_id": study.study_id,
        "source_commit": source["commit"],
        "source_tree_dirty": False,
        "plan_sha256": _sha256_file(plan_path),
        "study_config_sha256": study.scientific_sha256,
        "study_config_yaml_sha256": _sha256_file(study.source_path),
        "resolved_study_config": json.loads(study.canonical_scientific_json),
        "expert_config_sha256": expert_hashes,
        "protocol_config_sha256": legacy_identity["protocol_config_sha256"],
        "fold_manifest_sha256": legacy_identity["fold_manifest_sha256"],
        "inventory_count": len(inventory),
        "inventory": inventory,
        "job_inventory_sha256": hashlib.sha256(_canonical_json(inventory).encode("utf-8")).hexdigest(),
        "shard_rule": _SHARD_RULE,
        "reuse_decisions": {
            "inner": reuse_rows,
            "outer": "deferred_until_all_inner_locks_are_valid",
        },
        "runtime_root_names": sorted(profile.reuse_roots),
    }
    payload["freeze_sha256"] = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    freeze_path = _freeze_path(profile, study)
    _check_immutable_json(freeze_path, payload)
    planner.freeze(stage="inner", freeze_sha256=payload["freeze_sha256"])
    _write_immutable_json(freeze_path, payload)
    return {
        "frozen": True,
        "path": str(freeze_path),
        "study_id": study.study_id,
        "source_commit": source["commit"],
        "plan_sha256": payload["plan_sha256"],
        "study_config_sha256": study.scientific_sha256,
        "inventory_count": len(inventory),
        "inner_reuse_candidates_audited": len(reuse_rows),
        "outer_reuse_audit": payload["reuse_decisions"]["outer"],
        "freeze_sha256": payload["freeze_sha256"],
    }


def _status_payload(study: StudyDefinition, profile: RuntimeProfile) -> dict[str, Any]:
    """Summarize frozen progress; never inspect outer artifacts before locks."""
    freeze_path = _freeze_path(profile, study)
    attempts = summarize_attempts(profile.run_root, study.study_id)
    if not freeze_path.exists() and not freeze_path.is_symlink():
        return {
            "status": "unfrozen",
            "study_id": study.study_id,
            "planned_jobs": 300,
            "next_action": "study freeze",
            "freeze_path": str(freeze_path),
            "attempts": attempts,
            "counts": {"attempts": {key: attempts[key] for key in ("running", "failed", "succeeded")}},
        }
    freeze = _read_freeze(freeze_path)
    repository_root = study.source_path.parents[2]
    source = _source_identity(repository_root)
    _validate_frozen_identity(study, profile, freeze, source)
    if sorted(profile.reuse_roots) != freeze.get("runtime_root_names"):
        raise ConfigError("named reuse roots differ from the frozen study configuration")

    planner = _make_planner(
        study, profile, _load_manager(study, profile),
        freeze_sha256=freeze.get("freeze_sha256"),
    )
    if planner.frozen_manifest()["fold_manifest_sha256"] != freeze.get("fold_manifest_sha256"):
        raise ConfigError("canonical fold membership differs from the frozen study")
    current_inventory = _inventory_payload(planner.inventory)
    if freeze["inventory"] != current_inventory:
        raise ConfigError("canonical expert-job inventory differs from the frozen study")
    inner = planner.plan(stage="inner", shard_count=1, shard_index=0, output_path=profile.run_root)
    _verify_reuse_audit(inner.compatibility_table, freeze["reuse_decisions"]["inner"], "inner")
    inner_counts = {
        "validated": inner.validated_count,
        "reused": inner.reused_count,
        "missing": inner.missing_count,
        "invalid": inner.invalid_count,
    }
    locks_ready = False
    lock_error: str | None = None
    try:
        planner.validate_complete_lock_matrix(frozen=planner.frozen_manifest())
        locks_ready = True
    except (ValueError, OSError, RuntimeError) as exc:
        lock_error = str(exc)

    outer_counts = {"validated": 0, "reused": 0, "missing": 60, "invalid": 0}
    if locks_ready:
        outer = planner.plan(stage="outer", shard_count=1, shard_index=0, output_path=profile.run_root)
        outer_counts = {
            "validated": outer.validated_count,
            "reused": outer.reused_count,
            "missing": outer.missing_count,
            "invalid": outer.invalid_count,
        }
    if inner_counts["invalid"] or outer_counts["invalid"]:
        status = "invalid_artifacts"
        next_action = "inspect invalid artifacts; immutable runs must not be overwritten"
    elif inner_counts["missing"]:
        status = "inner_incomplete"
        next_action = "complete or restore inner expert jobs, then run study status again"
    elif not locks_ready:
        status = "inner_complete_locks_pending"
        next_action = "create and validate all 15 method locks before outer planning"
    elif outer_counts["missing"]:
        status = "outer_incomplete"
        next_action = "complete or restore outer expert jobs, then run study status again"
    else:
        status = "expert_matrix_complete"
        next_action = "study evaluate"
    return {
        "status": status,
        "study_id": study.study_id,
        "source_commit": freeze["source_commit"],
        "study_config_sha256": freeze["study_config_sha256"],
        "counts": {
            "inner": inner_counts,
            "outer": outer_counts,
            "attempts": {key: attempts[key] for key in ("running", "failed", "succeeded")},
            "locks_valid": 15 if locks_ready else 0,
            "outer_access_gated": not locks_ready,
        },
        "attempts": attempts,
        "lock_validation_error": lock_error,
        "next_action": next_action,
    }


def _doctor_payload(study: StudyDefinition, profile: RuntimeProfile) -> dict[str, Any]:
    """Inspect local prerequisites without loading data or creating paths."""
    required_modules = ("torch", "torchvision", "sklearn", "scipy", "yaml", "numpy")
    dependencies = {
        name: importlib.util.find_spec(name) is not None for name in required_modules
    }
    repository_root = study.source_path.parents[2]
    identity = _source_identity(repository_root)
    data_path = Path(profile.data_root).expanduser()
    run_path = Path(profile.run_root).expanduser()
    bundle_path = Path(profile.bundle_output_dir).expanduser()

    def path_status(path: Path, *, access: int = os.R_OK) -> dict[str, Any]:
        parent = path
        while not parent.exists() and parent != parent.parent:
            parent = parent.parent
        is_dir = path.is_dir()
        return {
            "path": str(path),
            "exists": path.exists(),
            "is_directory": is_dir,
            "accessible": bool(is_dir and os.access(path, access)) if path.exists() else None,
            "nearest_existing_directory": str(parent),
            "nearest_parent_writable": bool(parent.is_dir() and os.access(parent, os.W_OK)),
        }

    run_parent = run_path
    while not run_parent.exists() and run_parent != run_parent.parent:
        run_parent = run_parent.parent
    bundle_parent = bundle_path
    while not bundle_parent.exists() and bundle_parent != bundle_parent.parent:
        bundle_parent = bundle_parent.parent
    disk: dict[str, Any] = {}
    for name, path in (("run_root", run_parent), ("bundle_output_dir", bundle_parent)):
        try:
            usage = shutil.disk_usage(path)
            disk[name] = {"path": str(path), "total_bytes": usage.total, "free_bytes": usage.free}
        except OSError as exc:
            disk[name] = {"path": str(path), "error": str(exc)}

    cuda_available = False
    torch_version: str | None = None
    if dependencies["torch"]:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            torch_version = str(torch.__version__)
        except Exception as exc:  # noqa: BLE001 - report a failed preflight check
            dependencies["torch"] = False
            torch_version = f"unavailable: {type(exc).__name__}: {exc}"

    data_status = path_status(data_path)
    run_status = path_status(run_path, access=os.W_OK)
    bundle_status = path_status(bundle_path, access=os.W_OK)
    reuse_status = {
        name: path_status(Path(path).expanduser()) for name, path in profile.reuse_roots.items()
    }
    cuda_required = profile.device == "cuda"
    checks = {
        "dependencies": all(dependencies.values()),
        "git_head": bool(identity["commit"]),
        "git_clean": not identity["dirty"],
        "data_path": bool(data_status["exists"] and data_status["is_directory"] and data_status["accessible"]),
        "run_parent_writable": bool(run_status["nearest_parent_writable"]),
        "bundle_parent_writable": bool(bundle_status["nearest_parent_writable"]),
        "reuse_paths": all(
            row["exists"] and row["is_directory"] and row["accessible"]
            for row in reuse_status.values()
        ),
        "cuda": (not cuda_required) or cuda_available,
        "disk": all("free_bytes" in row for row in disk.values()),
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "dependencies": dependencies,
        "git": {"head": identity["commit"], "dirty": identity["dirty"], "repository_root": str(repository_root)},
        "paths": {
            "data_root": data_status,
            "run_root": run_status,
            "bundle_output_dir": bundle_status,
            "reuse_roots": reuse_status,
        },
        "disk": disk,
        "device": {"requested": profile.device, "cuda_available": cuda_available, "torch_version": torch_version},
        "test_accessed": False,
        "mutated": False,
    }


def _make_planner(
    study: StudyDefinition,
    profile: RuntimeProfile,
    manager: Any,
    *,
    read_only: bool = True,
    freeze_sha256: str | None = None,
) -> Any:
    """Construct the matrix planner from the loaded science and expert YAML."""
    from scripts.config import TrainingConfig
    from expert_method.ridge_sinkhorn.matrix import OOFMatrixPlanner

    repository_root = study.source_path.parents[2]
    training_configs = {
        expert: TrainingConfig.from_file(repository_root / relative_path)
        for expert, relative_path in study.expert_config_paths.items()
    }
    return OOFMatrixPlanner(
        manager=manager,
        study_id=study.study_id,
        artifact_root=profile.run_root,
        reuse_roots=profile.reuse_roots,
        training_configs=training_configs,
        study_config=study.to_study_config(),
        training_epochs=study.protocol["epochs"],
        freeze_sha256=freeze_sha256,
        read_only=read_only,
    )


def _load_manager(study: StudyDefinition, profile: RuntimeProfile) -> Any:
    """Load folds from the canonical training population only."""
    from data.nested_oof import NestedOOFFoldManager

    protocol = study.protocol
    return NestedOOFFoldManager.from_canonical_training_data(
        profile.data_root,
        seed=protocol["fold_generation_seed"],
        outer_folds=protocol["outer_folds"],
        inner_folds=protocol["inner_folds"],
        expert_order=protocol["expert_order"],
    )


def _expert_config_hashes(study: StudyDefinition) -> dict[str, str]:
    return {
        expert: hashlib.sha256(_canonical_json(_plain(recipe)).encode("utf-8")).hexdigest()
        for expert, recipe in study.resolved_expert_configs.items()
    }


def _freeze_path(profile: RuntimeProfile, study: StudyDefinition) -> Path:
    return Path(profile.run_root).expanduser() / study.study_id / "manifests" / "study-freeze.json"


def _source_identity(repository_root: Path) -> dict[str, Any]:
    """Return committed HEAD and whether any tracked or untracked source is dirty."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"], cwd=repository_root,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository_root,
            check=True, capture_output=True, text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": True}
    if len(commit) not in {40, 64}:
        return {"commit": None, "dirty": True}
    try:
        int(commit, 16)
    except ValueError:
        return {"commit": None, "dirty": True}
    return {"commit": commit, "dirty": bool(status.strip())}


def _read_freeze(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ConfigError(f"frozen study record is not a regular file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"cannot read frozen study record {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != _FREEZE_SCHEMA:
        raise ConfigError("unsupported or malformed study freeze record")
    required = {
        "schema_version", "study_id", "source_commit", "source_tree_dirty", "plan_sha256",
        "study_config_sha256", "study_config_yaml_sha256", "resolved_study_config",
        "expert_config_sha256", "protocol_config_sha256", "fold_manifest_sha256",
        "inventory_count", "inventory", "job_inventory_sha256", "shard_rule", "reuse_decisions",
        "runtime_root_names", "freeze_sha256",
    }
    if set(payload) != required:
        raise ConfigError("study freeze record has unknown or missing fields")
    recorded_hash = payload.get("freeze_sha256")
    unhashed = dict(payload)
    unhashed.pop("freeze_sha256", None)
    if recorded_hash != hashlib.sha256(_canonical_json(unhashed).encode("utf-8")).hexdigest():
        raise ConfigError("study freeze record content hash does not match")
    if path.read_bytes() != (_canonical_json(payload) + "\n").encode("utf-8"):
        raise ConfigError("study freeze record is not canonical JSON")
    inventory = payload.get("inventory")
    if (
        payload.get("study_id") != "ridge_sinkhorn_3seed_v1"
        or payload.get("source_tree_dirty") is not False
        or payload.get("inventory_count") != 300
        or not isinstance(inventory, list)
        or len(inventory) != 300
    ):
        raise ConfigError("study freeze record has an incomplete canonical inventory")
    job_ids = [item.get("job_id") if isinstance(item, Mapping) else None for item in inventory]
    if any(not isinstance(job_id, str) for job_id in job_ids) or job_ids != sorted(set(job_ids)):
        raise ConfigError("study freeze job IDs must be unique and canonically sorted")
    if sum(item.get("stage") == "inner" for item in inventory) != 240 or sum(
        item.get("stage") == "outer" for item in inventory
    ) != 60:
        raise ConfigError("study freeze must contain 240 inner and 60 outer jobs")
    if hashlib.sha256(_canonical_json(inventory).encode("utf-8")).hexdigest() != payload.get("job_inventory_sha256"):
        raise ConfigError("study freeze job inventory hash does not match")
    if payload.get("shard_rule") != _SHARD_RULE:
        raise ConfigError("study freeze has an unsupported shard assignment rule")
    reuse_decisions = payload.get("reuse_decisions")
    if (
        not isinstance(reuse_decisions, Mapping)
        or set(reuse_decisions) != {"inner", "outer"}
        or not isinstance(reuse_decisions["inner"], list)
        or len(reuse_decisions["inner"]) != 16
        or reuse_decisions["outer"] != "deferred_until_all_inner_locks_are_valid"
    ):
        raise ConfigError("study freeze has incomplete historical reuse decisions")
    _require_sha256(payload["source_commit"], "source commit", lengths={40, 64})
    for name in (
        "plan_sha256", "study_config_sha256", "study_config_yaml_sha256",
        "protocol_config_sha256", "fold_manifest_sha256",
    ):
        _require_sha256(payload[name], name)
    return payload


def _validate_frozen_identity(
    study: StudyDefinition,
    profile: RuntimeProfile,
    freeze: Mapping[str, Any],
    source: Mapping[str, Any],
) -> None:
    """Require the current checkout, study YAML, and protocol to match freeze."""
    if source.get("dirty") or source.get("commit") != freeze.get("source_commit"):
        raise ConfigError("current checkout is dirty or differs from the frozen study commit")
    if study.study_id != freeze.get("study_id"):
        raise ConfigError("study ID differs from the frozen study")
    plan_path = study.source_path.parents[2] / "docs" / "PLAN.md"
    if _sha256_file(plan_path) != freeze.get("plan_sha256"):
        raise ConfigError("docs/PLAN.md differs from the frozen study plan")
    if _sha256_file(study.source_path) != freeze.get("study_config_yaml_sha256"):
        raise ConfigError("study YAML bytes differ from the frozen study")
    if study.scientific_sha256 != freeze.get("study_config_sha256"):
        raise ConfigError("resolved study configuration differs from the frozen study")
    if json.loads(study.canonical_scientific_json) != freeze.get("resolved_study_config"):
        raise ConfigError("resolved study snapshot differs from the frozen study")


def _require_sha256(value: Any, name: str, *, lengths: set[int] | None = None) -> None:
    valid_lengths = lengths or {64}
    if not isinstance(value, str) or len(value) not in valid_lengths:
        raise ConfigError(f"study freeze has malformed {name}")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ConfigError(f"study freeze has non-hexadecimal {name}") from exc


def _inventory_payload(jobs: Sequence[Any]) -> list[dict[str, Any]]:
    """Serialize jobs with canonical membership and deterministic shard identity."""
    from expert_method.ridge_sinkhorn.matrix import shard_key

    return [
        {
            **asdict(job),
            "run_relative_path": job.run_relative_path,
            "shard_key": f"{shard_key(job.job_id):016x}",
            "shard_rule": "first_8_sha256_bytes_big_endian_mod_shard_count",
        }
        for job in jobs
    ]


def _verify_reuse_audit(
    actual: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]], stage: str
) -> None:
    """Reject any historical reuse result that differs from the freeze record."""
    actual_rows = [dict(row) for row in actual]
    expected_rows = [dict(row) for row in expected]
    if _canonical_json(actual_rows) != _canonical_json(expected_rows):
        raise ConfigError(f"{stage} historical reuse decisions differ from the frozen audit")


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write canonical JSON once, refusing to replace any existing bytes."""
    path = path.expanduser()
    serialized = (_canonical_json(dict(payload)) + "\n").encode("utf-8")
    if path.exists() or path.is_symlink():
        if path.is_file() and not path.is_symlink() and path.read_bytes() == serialized:
            return
        raise ConfigError(f"refusing to replace an incompatible immutable freeze: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".freeze-", delete=False) as handle:
            temporary_path = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if not path.is_file() or path.is_symlink() or path.read_bytes() != serialized:
                raise ConfigError(f"refusing to replace an incompatible immutable freeze: {path}")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _check_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Reject a competing freeze before creating legacy manifest files."""
    if not path.exists() and not path.is_symlink():
        return
    serialized = (_canonical_json(dict(payload)) + "\n").encode("utf-8")
    if path.is_file() and not path.is_symlink() and path.read_bytes() == serialized:
        return
    raise ConfigError(f"refusing to replace an incompatible immutable freeze: {path}")


def _sha256_file(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise ConfigError(f"expected regular file while hashing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ConfigError("value cannot be serialized as canonical JSON") from exc


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _emit(payload: Mapping[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False))
