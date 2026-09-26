"""Stable application services behind the public study and bundle commands."""

from __future__ import annotations

from dataclasses import replace
import glob
import hashlib
import io
import json
import fcntl
from pathlib import PurePosixPath
from pathlib import Path
import os
import shutil
import tarfile
import tempfile
from typing import Any, Mapping, Sequence

from expert_method.attempts import AttemptWorkspace, summarize_attempts
from expert_method.config import ConfigError, RuntimeProfile, StudyDefinition


class WorkflowError(ValueError):
    """Raised when a requested study stage is not safe or ready to run."""


_LOCK_BUNDLE_SCHEMA = "expert_method.study_lock_bundle.v1"


def _promote_attempt_run(
    *,
    attempt_store: Any,
    canonical_store: Any,
    context: Any,
    expected_config: Mapping[str, Any],
    attempt: AttemptWorkspace,
) -> bool:
    """Atomically move one validated attempt payload into its immutable path.

    Returns true when another session already promoted a valid canonical run.
    A per-job advisory lock serializes writers and is released by the kernel if
    a process exits unexpectedly. The lock file itself is intentionally kept:
    unlinking it could let a waiter and a new process lock different inodes.
    """
    source = attempt_store.run_dir(context).resolve()
    destination = canonical_store.run_dir(context).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    lock_path = destination.parent / f".{destination.name}.promotion.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        if destination.exists() or destination.is_symlink():
            canonical_store.validate_completed_run(
                context, expected_config=expected_config
            )
            return True
        if not source.is_dir() or source.is_symlink():
            raise WorkflowError(f"validated attempt payload is missing: {source}")
        if source.stat().st_dev != destination.parent.stat().st_dev:
            raise WorkflowError("attempt and canonical job paths must share a filesystem")
        os.rename(source, destination)
        _fsync_directory(destination.parent)
        try:
            canonical_store.validate_completed_run(
                context, expected_config=expected_config
            )
        except Exception:
            # Keep a failed promotion debuggable and retryable without leaving a
            # partial payload at the canonical location.
            if destination.exists() and not source.exists():
                os.rename(destination, source)
                _fsync_directory(destination.parent)
            raise
        return False
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _frozen_context(
    study: StudyDefinition,
    profile: RuntimeProfile,
    *,
    read_only: bool = True,
) -> tuple[dict[str, Any], Any, Any]:
    """Load and validate the immutable freeze plus native matrix manifests."""
    from expert_method import cli

    freeze_path = cli._freeze_path(profile, study)
    freeze = cli._read_freeze(freeze_path)
    source = cli._source_identity(study.source_path.parents[2])
    cli._validate_frozen_identity(study, profile, freeze, source)
    if sorted(profile.reuse_roots) != freeze.get("runtime_root_names"):
        raise ConfigError("named reuse roots differ from the frozen study configuration")
    manager = cli._load_manager(study, profile)
    planner = cli._make_planner(
        study,
        profile,
        manager,
        read_only=read_only,
        freeze_sha256=freeze["freeze_sha256"],
    )
    _validate_matrix_manifests(study, profile, freeze, planner)
    return freeze, manager, planner


def _validate_matrix_manifests(
    study: StudyDefinition,
    profile: RuntimeProfile,
    freeze: Mapping[str, Any],
    planner: Any,
    *,
    stages: Sequence[str] = ("inner",),
) -> None:
    """Require fold/job/reuse manifests to match the immutable freeze."""
    from expert_method import cli
    from expert_method.ridge_sinkhorn.matrix import (
        COMPATIBILITY_SCHEMA_VERSION,
        canonical_json_bytes,
    )

    base = planner.native_store.base_dir
    fold_path = planner.native_store.manifest_path
    if not fold_path.is_file() or fold_path.is_symlink():
        raise WorkflowError("frozen fold_manifest.json is missing; run study freeze first")
    expected_fold = planner.manager.manifest().to_json()
    if fold_path.read_text(encoding="utf-8") != expected_fold:
        raise WorkflowError("fold_manifest.json differs from canonical fold membership")
    if hashlib.sha256(expected_fold.encode("utf-8")).hexdigest() != freeze["fold_manifest_sha256"]:
        raise WorkflowError("frozen fold manifest differs from the study freeze record")

    expected_job = planner.frozen_manifest()
    if expected_job.get("study_freeze_sha256") != freeze.get("freeze_sha256"):
        raise WorkflowError("matrix planner is not bound to the current study freeze")
    job_path = base / "job_manifest.json"
    if not job_path.is_file() or job_path.is_symlink():
        raise WorkflowError("frozen job_manifest.json is missing; run study freeze first")
    try:
        recorded_job = json.loads(job_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError("frozen job_manifest.json is invalid") from exc
    if job_path.read_bytes() != canonical_json_bytes(recorded_job) or recorded_job != expected_job:
        raise WorkflowError("job_manifest.json differs from the immutable study identity")
    current_inventory = cli._inventory_payload(planner.inventory)
    if current_inventory != freeze.get("inventory") or current_inventory != expected_job.get("inventory"):
        raise WorkflowError("canonical job inventory differs from the study freeze record")

    requested = set(stages)
    if "outer" in requested:
        # This is deliberately ahead of every outer reuse/artifact audit.
        planner.validate_complete_lock_matrix(frozen=expected_job)
    for stage in sorted(requested):
        if stage not in {"inner", "outer"}:
            raise WorkflowError(f"unsupported matrix stage: {stage}")
        audit_path = base / f"reuse_compatibility_{stage}.json"
        if not audit_path.is_file() or audit_path.is_symlink():
            raise WorkflowError(f"frozen {stage} reuse compatibility audit is missing")
        try:
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WorkflowError(f"frozen {stage} reuse compatibility audit is invalid") from exc
        expected_audit = {
            "schema_version": COMPATIBILITY_SCHEMA_VERSION,
            "study_id": study.study_id,
            "study_freeze_sha256": freeze["freeze_sha256"],
            "stage": stage,
            "rows": [
                dict(row)
                for row in planner.audit_compatibility(
                    tuple(job for job in planner.inventory if job.stage == stage)
                )
            ],
        }
        if audit_path.read_bytes() != canonical_json_bytes(audit) or audit != expected_audit:
            raise WorkflowError(f"frozen {stage} reuse decisions differ from the current audit")
        if stage == "inner":
            from expert_method.cli import _verify_reuse_audit

            _verify_reuse_audit(
                expected_audit["rows"], freeze["reuse_decisions"]["inner"], stage
            )


def run_batch(
    study: StudyDefinition,
    profile: RuntimeProfile,
    *,
    stage: str,
    max_jobs: int | None = None,
    confirmed_full_run: bool = False,
) -> dict[str, Any]:
    """Run no more than one profile-bounded batch of missing expert jobs."""
    if not confirmed_full_run:
        raise WorkflowError("full expert training requires explicit confirmation")
    if stage not in {"inner", "outer"}:
        raise WorkflowError("stage must be inner or outer")
    limit = profile.max_jobs if max_jobs is None else max_jobs
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise WorkflowError("max_jobs must be a positive integer")
    if limit > profile.max_jobs:
        raise WorkflowError(f"max_jobs cannot exceed runtime profile limit {profile.max_jobs}")

    freeze, manager, read_planner = _frozen_context(study, profile, read_only=False)
    if stage == "outer":
        # Ensure locks are complete before planning can inspect any outer run.
        read_planner.validate_complete_lock_matrix(frozen=read_planner.frozen_manifest())
        _validate_matrix_manifests(study, profile, freeze, read_planner, stages=("outer",))
    plan = read_planner.plan(
        stage=stage,
        shard_index=profile.shard_index,
        shard_count=profile.shard_count,
        max_jobs=limit,
        output_path=profile.run_root,
    )
    if plan.invalid_count:
        raise WorkflowError(f"found {plan.invalid_count} invalid immutable artifacts; refusing to train")
    missing = [status.job for status in plan.statuses if status.state == "missing"][:limit]
    completed: list[dict[str, str]] = []
    failures: list[dict[str, str]] = []
    if missing:
        from expert_method.oof.pipeline import OOFArtifactStore, OOFPipeline
        from expert_method.ridge_sinkhorn.matrix import cuda_preflight, sha256_file

        try:
            cuda_preflight(profile.device)
        except Exception as exc:
            failures.append({"job_id": "<cuda-preflight>", "error": str(exc)})
        if not failures:
            for job in missing:
                config = replace(
                    read_planner.training_configs[job.expert_key],
                    data=replace(
                        read_planner.training_configs[job.expert_key].data,
                        root=profile.data_root,
                    ),
                )
                spec = job.run_spec(
                    experiment_id=study.study_id,
                    device=profile.device,
                    epochs=read_planner.training_epochs,
                )
                context = spec.resolve(manager)
                canonical_run_dir = read_planner.native_store.run_dir(context)
                attempt = AttemptWorkspace.create(
                    run_root=profile.run_root,
                    study_id=study.study_id,
                    job_id=job.job_id,
                    stage=stage,
                    expert_key=job.expert_key,
                    training_seed=job.training_seed,
                    outer_fold_id=job.outer_fold_id,
                    inner_fold_id=job.inner_fold_id,
                    device=profile.device,
                )
                phase = "resolving_config"
                attempt.update(
                    status="running",
                    phase=phase,
                    locations={"canonical_job": str(canonical_run_dir)},
                )
                try:
                    attempt_store = OOFArtifactStore(
                        root=profile.run_root,
                        base_dir=attempt.path,
                        manager=manager,
                        experiment_id=study.study_id,
                    )
                    resolved_config = config.replace(
                        seed=context.training_seed,
                        device=profile.device,
                        epochs=read_planner.training_epochs,
                        checkpoint_dir=str(canonical_run_dir / "checkpoints"),
                    ).to_dict()
                    resolved_config["resolved_device"] = config.replace(
                        seed=context.training_seed,
                        device=profile.device,
                        epochs=read_planner.training_epochs,
                    ).resolved_device
                    attempt.write_resolved_config(resolved_config)
                    attempt.update(
                        status="running",
                        phase="preparing_attempt",
                        locations={
                            "canonical_job": str(canonical_run_dir),
                            "attempt_run": str(attempt_store.run_dir(context)),
                        },
                    )
                    trainer = OOFPipeline(manager=manager, store=attempt_store)

                    def on_phase(next_phase: str) -> None:
                        nonlocal phase
                        phase = next_phase
                        attempt.update(
                            status="running",
                            phase=phase,
                            locations={"canonical_job": str(canonical_run_dir)},
                        )

                    with attempt.capture_output():
                        result = trainer.run(
                            spec,
                            config,
                            metrics_sink=attempt.metrics_sink,
                            phase_callback=on_phase,
                            provenance_run_dir=canonical_run_dir,
                        )
                        phase = "validating_attempt"
                        attempt.update(
                            status="running",
                            phase=phase,
                            locations={"canonical_job": str(canonical_run_dir)},
                        )
                        attempt_store.validate_completed_run(
                            context, expected_config=resolved_config
                        )
                        promoted_from_existing = _promote_attempt_run(
                            attempt_store=attempt_store,
                            canonical_store=read_planner.native_store,
                            context=context,
                            expected_config=resolved_config,
                            attempt=attempt,
                        )
                    canonical_result = read_planner.native_store.validate_completed_run(
                        context, expected_config=resolved_config
                    )
                    if promoted_from_existing:
                        phase = "existing_canonical_valid"
                    else:
                        phase = "promoted"
                    attempt.update(
                        status="succeeded",
                        phase=phase,
                        locations={
                            "canonical_job": str(canonical_result.run_dir),
                            "checkpoint": str(canonical_result.checkpoint_path),
                            "prediction": str(canonical_result.prediction_path),
                        },
                        finished=True,
                    )
                    completed.append({
                        "job_id": job.job_id,
                        "checkpoint_sha256": canonical_result.metadata["checkpoint"]["sha256"],
                        "prediction_sha256": sha256_file(canonical_result.prediction_path),
                        "attempt_id": attempt.attempt_id,
                        "attempt_path": str(attempt.path),
                        "canonical_path": str(canonical_result.run_dir),
                    })
                except Exception as exc:
                    failure = attempt.record_failure(phase=phase, error=exc)
                    failures.append({
                        "job_id": job.job_id,
                        "error": str(exc),
                        "attempt_id": attempt.attempt_id,
                        "attempt_path": str(attempt.path),
                        "phase": phase,
                        "exception_type": failure["exception_type"],
                    })
                    break

    # Re-read native files after training; do not infer success from trainer return values.
    verified = read_planner.plan(
        stage=stage,
        shard_index=profile.shard_index,
        shard_count=profile.shard_count,
        max_jobs=limit,
        output_path=profile.run_root,
    )
    trained_ids = {row["job_id"] for row in completed}
    missing_trained = trained_ids.intersection(
        status.job.job_id for status in verified.statuses if status.state == "missing"
    )
    if missing_trained:
        failures.append({
            "job_id": ",".join(sorted(missing_trained)),
            "error": "training returned without valid immutable artifacts",
        })
    if verified.invalid_count:
        failures.append({
            "job_id": "<post-run-validation>",
            "error": f"found {verified.invalid_count} invalid immutable artifacts",
        })
    return {
        "success": not failures,
        "study_id": study.study_id,
        "stage": stage,
        "shard": {"index": profile.shard_index, "count": profile.shard_count},
        "profile_max_jobs": profile.max_jobs,
        "batch_limit": limit,
        "completed_count": len(completed),
        "completed": completed,
        "failures": failures,
        "post_run_counts": {
            "validated": verified.validated_count,
            "reused": verified.reused_count,
            "missing": verified.missing_count,
            "invalid": verified.invalid_count,
        },
        "freeze_sha256": freeze["freeze_sha256"],
        "test_accessed": False,
    }


def restore_bundles(
    study: StudyDefinition,
    profile: RuntimeProfile,
    *,
    paths: Sequence[str] = (),
) -> dict[str, Any]:
    """Idempotently restore configured or explicitly listed cumulative bundles."""
    from expert_method import cli
    configured = tuple(paths) if paths else profile.bundle_inputs
    expanded: list[str] = []
    for value in configured:
        matches = sorted(glob.glob(value)) if any(char in value for char in "*?[") else [value]
        expanded.extend(matches)
    unique = tuple(dict.fromkeys(expanded))
    if not unique and not _freeze_path(profile, study).is_file():
        raise WorkflowError("no local freeze or configured bundles; run study freeze first")
    matrix_inner: list[str] = []
    matrix_outer: list[str] = []
    lock_paths: list[str] = []
    envelopes: dict[str, Mapping[str, Any]] = {}
    for value in unique:
        envelope = _read_bundle_envelope(value)
        envelopes[value] = envelope
        schema = envelope.get("schema_version")
        if schema == _LOCK_BUNDLE_SCHEMA:
            lock_paths.append(value)
        elif schema == "ridge_sinkhorn_bundle.v1":
            shard = envelope.get("shard")
            if not isinstance(shard, Mapping) or shard.get("stage") not in {"inner", "outer"}:
                raise WorkflowError(f"matrix bundle has malformed stage identity: {value}")
            (matrix_inner if shard["stage"] == "inner" else matrix_outer).append(value)
        else:
            raise WorkflowError(f"unsupported bundle format: {value}")

    freeze_path = _freeze_path(profile, study)
    if freeze_path.is_file() and not freeze_path.is_symlink():
        freeze = cli._read_freeze(freeze_path)
    else:
        seed_bundle = matrix_inner[0] if matrix_inner else (lock_paths[0] if lock_paths else None)
        if seed_bundle is None:
            raise WorkflowError("a fresh artifact root needs an inner or lock bundle carrying the study freeze")
        freeze = _freeze_from_bundle(seed_bundle, envelopes[seed_bundle], cli)
        source = cli._source_identity(study.source_path.parents[2])
        cli._validate_frozen_identity(study, profile, freeze, source)
        if sorted(profile.reuse_roots) != freeze.get("runtime_root_names"):
            raise ConfigError("named reuse roots differ from the frozen study configuration")

    source = cli._source_identity(study.source_path.parents[2])
    cli._validate_frozen_identity(study, profile, freeze, source)
    if sorted(profile.reuse_roots) != freeze.get("runtime_root_names"):
        raise ConfigError("named reuse roots differ from the frozen study configuration")

    from expert_method.ridge_sinkhorn.matrix import merge_bundles

    imported: tuple[str, ...] = ()
    if matrix_inner:
        imported += merge_bundles(
            matrix_inner,
            artifact_root=profile.run_root,
            reuse_roots=profile.reuse_roots,
            study_config=study.to_study_config(),
        )
    for lock_path in lock_paths:
        _restore_lock_bundle(lock_path, study, profile, freeze)
    if matrix_outer:
        # The importer checks every lock before reading any outer payload member.
        imported += merge_bundles(
            matrix_outer,
            artifact_root=profile.run_root,
            reuse_roots=profile.reuse_roots,
            study_config=study.to_study_config(),
        )
    _frozen_context(study, profile)
    if lock_paths or matrix_outer:
        _freeze, _manager, planner = _frozen_context(study, profile)
        planner.validate_complete_lock_matrix(frozen=planner.frozen_manifest())
    return {
        "restored_job_count": len(imported),
        "job_ids": list(imported),
        "freeze_sha256": freeze["freeze_sha256"],
    }


def export_lock_bundle(
    study: StudyDefinition,
    profile: RuntimeProfile,
    *,
    path: str | None = None,
) -> dict[str, Any]:
    """Export the complete lock matrix and its immutable matrix/freeze identity."""
    freeze, manager, planner = _frozen_context(study, profile)
    _validate_matrix_manifests(study, profile, freeze, planner, stages=("outer",))
    locks = planner.validate_complete_lock_matrix(frozen=planner.frozen_manifest())
    if len(locks) != 15:
        raise WorkflowError("lock bundle export requires all 15 validated method locks")
    from expert_method.ridge_sinkhorn.three_seed_study import StudyArtifactRepository
    from expert_method.ridge_sinkhorn.matrix import canonical_json_bytes, sha256_file

    repository = StudyArtifactRepository(profile.run_root, config=study.to_study_config())
    study_root = planner.native_store.base_dir
    files: dict[str, Path] = {}
    for relative in (
        "fold_manifest.json",
        "job_manifest.json",
        "reuse_compatibility_inner.json",
        "reuse_compatibility_outer.json",
        "manifests/study-freeze.json",
    ):
        candidate = study_root / relative
        if not candidate.is_file() or candidate.is_symlink():
            raise WorkflowError(f"required frozen matrix file is missing: {relative}")
        files[f"{study.study_id}/{relative}"] = candidate
    for candidate in sorted(repository.base_dir.rglob("*")):
        if not candidate.is_file() or candidate.is_symlink():
            continue
        relative_to_analysis = candidate.relative_to(repository.base_dir).as_posix()
        if relative_to_analysis not in {"study_config.json", "study_lock.json"} and not relative_to_analysis.startswith("locks/"):
            # Evaluation payloads are deliberately not part of a lock archive.
            continue
        relative = candidate.relative_to(study_root).as_posix()
        files[f"{study.study_id}/{relative}"] = candidate
    records = [
        {"path": name, "byte_size": source.stat().st_size, "sha256": sha256_file(source)}
        for name, source in sorted(files.items())
    ]
    manifest = {
        "schema_version": _LOCK_BUNDLE_SCHEMA,
        "study_id": study.study_id,
        "study_freeze_sha256": freeze["freeze_sha256"],
        "source_commit": freeze["source_commit"],
        "plan_sha256": freeze["plan_sha256"],
        "protocol_config_sha256": planner.study_config.sha256,
        "files": records,
    }
    destination = Path(path) if path is not None else (
        Path(profile.bundle_output_dir).expanduser() / f"{study.study_id}-locks.tar"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, mode="w", format=tarfile.GNU_FORMAT) as archive:
        manifest_bytes = canonical_json_bytes(manifest)
        archive.addfile(_tar_info("manifest.json", len(manifest_bytes)), io.BytesIO(manifest_bytes))
        for name, source in sorted(files.items()):
            with source.open("rb") as handle:
                archive.addfile(_tar_info(name, source.stat().st_size), handle)
    return {"path": str(destination), "lock_count": len(locks), "freeze_sha256": freeze["freeze_sha256"]}


def export_stage_bundle(
    study: StudyDefinition,
    profile: RuntimeProfile,
    *,
    stage: str,
    path: str | None = None,
) -> dict[str, Any]:
    """Export all validated native jobs in the configured shard cumulatively."""
    freeze, manager, planner = _frozen_context(study, profile, read_only=False)
    if stage == "outer":
        _validate_matrix_manifests(study, profile, freeze, planner, stages=("outer",))
    if stage not in {"inner", "outer"}:
        raise WorkflowError("stage must be inner or outer")
    destination = Path(path) if path is not None else (
        Path(profile.bundle_output_dir).expanduser()
        / f"{study.study_id}-{stage}-s{profile.shard_index}-of{profile.shard_count}.tar"
    )
    from expert_method.ridge_sinkhorn.matrix import StudyArtifactView, export_bundle

    view = StudyArtifactView(
        manager=manager,
        study_id=study.study_id,
        artifact_root=profile.run_root,
        reuse_roots=profile.reuse_roots,
        training_configs=planner.training_configs,
        study_config=study.to_study_config(),
        training_epochs=study.protocol["epochs"],
        freeze_sha256=freeze["freeze_sha256"],
    )
    output = export_bundle(
        view=view,
        destination=destination,
        stage=stage,
        shard_index=profile.shard_index,
        shard_count=profile.shard_count,
    )
    return {"path": str(output), "stage": stage, "freeze_sha256": freeze["freeze_sha256"]}


def run_session(
    study: StudyDefinition,
    profile: RuntimeProfile,
    *,
    stage: str,
    confirmed_full_run: bool = False,
) -> dict[str, Any]:
    """Restore, validate, train one bounded batch, revalidate, and export."""
    if not confirmed_full_run:
        raise WorkflowError("full expert training requires explicit confirmation")
    restored = restore_bundles(study, profile)
    run = run_batch(
        study,
        profile,
        stage=stage,
        max_jobs=profile.max_jobs,
        confirmed_full_run=True,
    )
    bundle: dict[str, Any] | None = None
    bundle_error: str | None = None
    try:
        bundle = export_stage_bundle(study, profile, stage=stage)
    except (WorkflowError, OSError, RuntimeError, ValueError) as exc:
        bundle_error = str(exc)
    return {
        "success": bool(run["success"] and bundle is not None),
        "study_id": study.study_id,
        "stage": stage,
        "restored_job_count": restored["restored_job_count"],
        "completed_count": run["completed_count"],
        "completed": run["completed"],
        "post_run_counts": run["post_run_counts"],
        "bundle_path": bundle["path"] if bundle is not None else None,
        "bundle_error": bundle_error,
        "failures": run["failures"],
        "attempts": summarize_attempts(profile.run_root, study.study_id),
        "freeze_sha256": run["freeze_sha256"],
        "test_accessed": False,
    }


def run_analysis_stage(
    study: StudyDefinition,
    profile: RuntimeProfile,
    *,
    stage: str,
) -> dict[str, Any]:
    """Run lock/evaluate/report through the stable package service boundary."""
    if stage not in {"lock", "evaluate", "report"}:
        raise WorkflowError("analysis stage must be lock, evaluate, or report")
    freeze, manager, planner = _frozen_context(study, profile, read_only=False)
    from expert_method.ridge_sinkhorn.matrix import StudyArtifactView
    from expert_method.ridge_sinkhorn.three_seed_study import StudyArtifactRepository
    from expert_method import analysis as analysis_service

    if stage == "lock":
        inner = planner.plan(stage="inner", shard_count=1, shard_index=0, output_path=profile.run_root)
        if inner.invalid_count or inner.missing_count:
            raise WorkflowError("all inner jobs must validate before study lock")
    else:
        planner.validate_complete_lock_matrix(frozen=planner.frozen_manifest())
        _validate_matrix_manifests(study, profile, freeze, planner, stages=("outer",))

    view = StudyArtifactView(
        manager=manager,
        study_id=study.study_id,
        artifact_root=profile.run_root,
        reuse_roots=profile.reuse_roots,
        training_configs=planner.training_configs,
        study_config=study.to_study_config(),
        training_epochs=study.protocol["epochs"],
        freeze_sha256=freeze["freeze_sha256"],
    )
    manifest = planner.frozen_manifest()
    repository = StudyArtifactRepository(profile.run_root, config=study.to_study_config())
    if stage == "lock":
        analysis_service.lock_study(
            view=view,
            manager=manager,
            repository=repository,
            manifest=manifest,
            manifest_sha256=hashlib.sha256(
                (Path(profile.run_root) / study.study_id / "job_manifest.json").read_bytes()
            ).hexdigest(),
        )
        planner.freeze(stage="outer", freeze_sha256=freeze["freeze_sha256"])
    elif stage == "evaluate":
        analysis_service.evaluate_study(
            view=view, manager=manager, repository=repository, manifest=manifest
        )
    else:
        analysis_service.report_study(
            view=view, manager=manager, repository=repository, manifest=manifest
        )
    return {
        "stage": stage,
        "study_id": study.study_id,
        "test_accessed": False,
        "analysis_root": str(repository.study_root),
        "freeze_sha256": freeze["freeze_sha256"],
    }


def _read_bundle_envelope(path: str | Path) -> dict[str, Any]:
    """Read only the first, metadata-only tar member for safe restore ordering."""
    from expert_method.ridge_sinkhorn.matrix import canonical_json_bytes

    try:
        with tarfile.open(path, mode="r:") as archive:
            first = archive.next()
            if first is None or first.name != "manifest.json" or not first.isfile():
                raise WorkflowError(f"bundle must start with a regular manifest.json: {path}")
            stream = archive.extractfile(first)
            if stream is None:
                raise WorkflowError(f"cannot read bundle manifest: {path}")
            encoded = stream.read()
    except (OSError, tarfile.TarError) as exc:
        raise WorkflowError(f"cannot open bundle {path}: {exc}") from exc
    try:
        manifest = json.loads(encoded)
    except json.JSONDecodeError as exc:
        raise WorkflowError(f"bundle manifest is invalid JSON: {path}") from exc
    if not isinstance(manifest, dict) or canonical_json_bytes(manifest) != encoded:
        raise WorkflowError(f"bundle manifest is not canonical JSON: {path}")
    return manifest


def _freeze_from_bundle(path: str | Path, envelope: Mapping[str, Any], cli: Any) -> dict[str, Any]:
    """Validate the freeze record embedded in a cumulative inner/lock bundle."""
    if envelope.get("schema_version") == "ridge_sinkhorn_bundle.v1":
        shard = envelope.get("shard")
        if not isinstance(shard, Mapping) or shard.get("stage") != "inner":
            raise WorkflowError("a fresh root must bootstrap from an inner or lock bundle")
        relative = f"{envelope.get('study_id')}/manifests/study-freeze.json"
        records = envelope.get("files")
        record = next((item for item in records if isinstance(item, Mapping) and item.get("path") == relative), None) if isinstance(records, list) else None
    elif envelope.get("schema_version") == _LOCK_BUNDLE_SCHEMA:
        relative = f"{envelope.get('study_id')}/manifests/study-freeze.json"
        records = envelope.get("files")
        record = next((item for item in records if isinstance(item, Mapping) and item.get("path") == relative), None) if isinstance(records, list) else None
    else:
        raise WorkflowError("bundle does not carry a supported study freeze")
    if not isinstance(record, Mapping):
        raise WorkflowError("bundle is missing the immutable study freeze record")
    try:
        with tarfile.open(path, mode="r:") as archive:
            member = archive.getmember(relative)
            if not member.isfile() or member.issym() or member.islnk() or member.size != record.get("byte_size"):
                raise WorkflowError("embedded study freeze archive member is unsafe or has the wrong size")
            stream = archive.extractfile(member)
            if stream is None:
                raise WorkflowError("cannot read embedded study freeze record")
            encoded = stream.read()
    except (KeyError, OSError, tarfile.TarError) as exc:
        raise WorkflowError("bundle freeze record is missing or unreadable") from exc
    if len(encoded) != record.get("byte_size") or hashlib.sha256(encoded).hexdigest() != record.get("sha256"):
        raise WorkflowError("embedded study freeze record hash differs from its bundle manifest")
    with tempfile.TemporaryDirectory(prefix="expert_method_freeze_") as temporary:
        candidate = Path(temporary) / "study-freeze.json"
        candidate.write_bytes(encoded)
        freeze = cli._read_freeze(candidate)
    if freeze.get("freeze_sha256") != envelope.get("study_freeze_sha256"):
        raise WorkflowError("bundle header and embedded study freeze identity disagree")
    return freeze


def _restore_lock_bundle(
    path: str | Path,
    study: StudyDefinition,
    profile: RuntimeProfile,
    freeze: Mapping[str, Any],
) -> tuple[str, ...]:
    """Verify and immutably install a separately exported complete lock matrix."""
    from expert_method.ridge_sinkhorn.matrix import canonical_json_bytes

    envelope = _read_bundle_envelope(path)
    if (
        envelope.get("schema_version") != _LOCK_BUNDLE_SCHEMA
        or envelope.get("study_id") != study.study_id
        or envelope.get("study_freeze_sha256") != freeze.get("freeze_sha256")
        or envelope.get("source_commit") != freeze.get("source_commit")
        or envelope.get("plan_sha256") != freeze.get("plan_sha256")
        or envelope.get("protocol_config_sha256") != freeze.get("protocol_config_sha256")
    ):
        raise WorkflowError("lock bundle identity differs from the local immutable study freeze")
    records = envelope.get("files")
    if not isinstance(records, list) or not records:
        raise WorkflowError("lock bundle file inventory is missing")
    expected: dict[str, Mapping[str, Any]] = {}
    for item in records:
        if not isinstance(item, Mapping):
            raise WorkflowError("lock bundle contains a malformed file record")
        name = item.get("path")
        if not isinstance(name, str):
            raise WorkflowError("lock bundle file path is malformed")
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != study.study_id:
            raise WorkflowError(f"unsafe lock bundle path: {name}")
        if name in expected:
            raise WorkflowError(f"duplicate lock bundle path: {name}")
        expected[name] = item
    required = {
        f"{study.study_id}/fold_manifest.json",
        f"{study.study_id}/job_manifest.json",
        f"{study.study_id}/reuse_compatibility_inner.json",
        f"{study.study_id}/reuse_compatibility_outer.json",
        f"{study.study_id}/manifests/study-freeze.json",
        f"{study.study_id}/study_analysis/study_config.json",
        f"{study.study_id}/study_analysis/study_lock.json",
    }
    required.update(
        f"{study.study_id}/study_analysis/locks/seed_{seed}_outer_{outer}.json"
        for seed in study.to_study_config().seeds
        for outer in study.to_study_config().outer_folds
    )
    if not required.issubset(expected):
        raise WorkflowError("lock bundle does not carry the complete 15-lock matrix and manifests")
    staged_paths: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(prefix="expert_method_locks_") as temporary:
        stage_root = Path(temporary)
        try:
            with tarfile.open(path, mode="r:") as archive:
                seen = {"manifest.json"}
                first = archive.next()
                if first is None or first.name != "manifest.json" or not first.isfile():
                    raise WorkflowError("lock bundle must start with a regular manifest")
                while True:
                    member = archive.next()
                    if member is None:
                        break
                    if member.name in seen or not member.isfile() or member.issym() or member.islnk():
                        raise WorkflowError(f"unsafe or duplicate lock bundle member: {member.name}")
                    seen.add(member.name)
                    record = expected.get(member.name)
                    if record is None or member.size != record.get("byte_size"):
                        raise WorkflowError(f"unlisted or mismatched lock bundle member: {member.name}")
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise WorkflowError(f"cannot read lock bundle member: {member.name}")
                    relative = PurePosixPath(member.name)
                    staged = stage_root.joinpath(*relative.parts)
                    staged.parent.mkdir(parents=True, exist_ok=True)
                    digest = hashlib.sha256()
                    size = 0
                    with staged.open("xb") as output:
                        while chunk := stream.read(1024 * 1024):
                            output.write(chunk)
                            digest.update(chunk)
                            size += len(chunk)
                    if size != record.get("byte_size") or digest.hexdigest() != record.get("sha256"):
                        raise WorkflowError(f"lock bundle content hash mismatch: {member.name}")
                    staged_paths[member.name] = staged
        except (OSError, tarfile.TarError) as exc:
            raise WorkflowError(f"cannot validate lock bundle: {exc}") from exc
        if seen != {"manifest.json", *expected}:
            raise WorkflowError("lock bundle archive members differ from the signed file inventory")
        staged_freeze_path = staged_paths[f"{study.study_id}/manifests/study-freeze.json"]
        from expert_method import cli

        staged_freeze = cli._read_freeze(staged_freeze_path)
        if staged_freeze.get("freeze_sha256") != freeze.get("freeze_sha256"):
            raise WorkflowError("lock bundle contains a different study freeze")
        root = Path(profile.run_root).expanduser().resolve()
        installed: list[str] = []
        for name, staged in staged_paths.items():
            relative = PurePosixPath(name)
            target = root.joinpath(*relative.parts)
            if root != target.resolve(strict=False) and root not in target.resolve(strict=False).parents:
                raise WorkflowError(f"lock bundle destination escapes artifact root: {name}")
            if target.exists():
                if target.is_symlink() or not target.is_file() or target.read_bytes() != staged.read_bytes():
                    raise WorkflowError(f"refusing to overwrite differing frozen lock artifact: {target}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with target.open("xb") as output, staged.open("rb") as source:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                    output.flush()
                    os.fsync(output.fileno())
            except FileExistsError:
                if target.is_symlink() or target.read_bytes() != staged.read_bytes():
                    raise WorkflowError(f"concurrent lock restore produced conflicting bytes: {target}")
            installed.append(name)
    freeze_after, _manager, planner = _frozen_context(study, profile)
    if freeze_after["freeze_sha256"] != freeze["freeze_sha256"]:
        raise WorkflowError("restored lock bundle changed the local study freeze")
    planner.validate_complete_lock_matrix(frozen=planner.frozen_manifest())
    return tuple(sorted(installed))


def _tar_info(name: str, size: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = 0o644
    info.type = tarfile.REGTYPE
    info.pax_headers = {}
    return info


def _freeze_path(profile: RuntimeProfile, study: StudyDefinition) -> Path:
    return Path(profile.run_root).expanduser() / study.study_id / "manifests" / "study-freeze.json"
