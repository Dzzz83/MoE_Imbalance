"""Paper-study lock, outer evaluation, and reporting services."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from data.nested_oof import NestedOOFFoldManager
from scripts.analysis import ArtifactReader
from expert_method.ridge_sinkhorn.matrix import MATRIX_SCHEMA_VERSION, canonical_json_bytes
from expert_method.ridge_sinkhorn.three_seed_study import (
    FoldStudyRunner,
    InferenceBatch,
    InnerOOFDataset,
    MarkdownReportBuilder,
    StudyArtifactRepository,
    StudyError,
    StudyEvaluator,
    StudyConfig,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_frozen_matrix_identity(
    artifact_root: str | Path,
    config: StudyConfig,
    *,
    expected_manifest: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Validate the canonical, complete expert inventory manifest."""
    from expert_method.ridge_sinkhorn.three_seed_study import STUDY_ID

    manifest_path = Path(artifact_root).expanduser().resolve() / STUDY_ID / "job_manifest.json"
    reader = ArtifactReader(error_type=StudyError)
    manifest = reader.read_json(manifest_path, name="frozen expert-job manifest")
    if manifest_path.read_bytes() != canonical_json_bytes(manifest):
        raise StudyError("expert-job manifest is not canonical JSON")
    plan_path = _PROJECT_ROOT / "docs" / "PLAN.md"
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    if manifest.get("schema_version") != MATRIX_SCHEMA_VERSION or manifest.get("study_id") != STUDY_ID:
        raise StudyError("expert-job manifest has an unsupported study identity")
    if manifest.get("plan_sha256") != plan_sha:
        raise StudyError("PLAN.md changed after the expert job inventory was frozen")
    if manifest.get("protocol_config_sha256") != config.sha256:
        raise StudyError("expert-job manifest uses a different frozen StudyConfig")
    source_commit = manifest.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) not in {40, 64}:
        raise StudyError("expert-job manifest has no exact Git source commit")
    try:
        int(source_commit, 16)
    except ValueError as exc:
        raise StudyError("expert-job manifest source commit is not hexadecimal") from exc
    inventory = manifest.get("inventory")
    if manifest.get("inventory_count") != 300 or not isinstance(inventory, list) or len(inventory) != 300:
        raise StudyError("expert-job manifest does not freeze all 300 jobs")
    if any(not isinstance(item, Mapping) or not isinstance(item.get("job_id"), str) for item in inventory):
        raise StudyError("expert-job manifest has malformed inventory rows")
    job_ids = [item["job_id"] for item in inventory]
    if len(set(job_ids)) != 300 or job_ids != sorted(job_ids):
        raise StudyError("expert-job manifest IDs must be unique and canonically sorted")
    if sum(item.get("stage") == "inner" for item in inventory) != 240 or sum(
        item.get("stage") == "outer" for item in inventory
    ) != 60:
        raise StudyError("expert-job manifest must contain exactly 240 inner and 60 outer jobs")
    if expected_manifest is not None:
        identity_fields = (
            "schema_version", "study_id", "plan_sha256", "study_config_sha256",
            "protocol_config_sha256", "source_commit", "fold_manifest_sha256",
            "inventory_count", "inventory", "shard_rule",
        )
        for field in identity_fields:
            if manifest.get(field) != expected_manifest.get(field):
                raise StudyError(f"expert-job manifest field {field} differs from the canonical inventory")
    return manifest, reader.sha256_file(manifest_path, description="expert-job manifest")


def job_id_lookup(view: Any) -> dict[tuple[str, str, int, int, int | None], str]:
    return {
        (job.stage, job.expert_key, job.training_seed, job.outer_fold_id, job.inner_fold_id): job.job_id
        for job in view.planner.inventory
    }


def source_hashes(prediction: Any) -> tuple[tuple[str, str], ...]:
    prefix = prediction.job_id
    return tuple((f"{prefix}/{name}", digest) for name, digest in (
        ("checkpoint", prediction.checkpoint_sha256),
        ("prediction", prediction.prediction_sha256),
        ("resolved_config", prediction.resolved_config_sha256),
    ))


def reference_source_hashes(reference: Any) -> tuple[tuple[str, str], ...]:
    prefix = reference.job.job_id
    return tuple((f"{prefix}/{name}", digest) for name, digest in (
        ("checkpoint", reference.checkpoint_sha256),
        ("prediction", reference.prediction_sha256),
        ("resolved_config", reference.resolved_config_sha256),
    ))


def validate_fold_lock_membership(lock: Any, manager: NestedOOFFoldManager) -> None:
    outer_training = set(manager.outer_fold(lock.outer_fold_id).expert_training_indices)
    observed_union: set[int] = set()
    for fold_record in lock.inner_fold_memberships:
        expected = set(manager.inner_fold(lock.outer_fold_id, fold_record.inner_fold_id).prediction_indices)
        recorded = set(int(value) for value in fold_record.validation_sample_ids)
        if recorded != expected:
            raise StudyError(
                f"fold lock has non-canonical inner fold {fold_record.inner_fold_id} membership "
                f"for seed={lock.training_seed}, outer={lock.outer_fold_id}"
            )
        observed_union.update(recorded)
    if observed_union != outer_training:
        raise StudyError("fold lock's inner validation union differs from canonical outer training")


def validate_complete_job_references(view: Any, locks: tuple[Any, ...]) -> dict[str, Any]:
    inventory = tuple(view.planner.inventory)
    if len(inventory) != 300 or len({job.job_id for job in inventory}) != 300:
        raise StudyError("canonical expert inventory is not the complete 300-job matrix")
    references: dict[str, Any] = {}
    for job in inventory:
        reference = view.resolve(job.job_id)
        if reference.job != job:
            raise StudyError(f"resolved expert reference changed canonical identity for {job.job_id}")
        for _name, digest in reference_source_hashes(reference):
            if not isinstance(digest, str) or len(digest) != 64:
                raise StudyError(f"resolved expert reference has a malformed content hash: {job.job_id}")
            try:
                int(digest, 16)
            except ValueError as exc:
                raise StudyError(f"resolved expert reference has a non-hexadecimal hash: {job.job_id}") from exc
        references[job.job_id] = reference
    if len(references) != 300:
        raise StudyError("complete expert reference validation found a missing or duplicate job")
    locks_by_pair = {(lock.training_seed, lock.outer_fold_id): lock for lock in locks}
    if len(locks_by_pair) != 15:
        raise StudyError("complete expert-reference validation requires all 15 fold locks")
    for (seed, outer), lock in locks_by_pair.items():
        expected_inner = tuple(sorted(
            digest
            for job in inventory
            if job.stage == "inner" and job.training_seed == seed and job.outer_fold_id == outer
            for digest in reference_source_hashes(references[job.job_id])
        ))
        if tuple(sorted(lock.source_hashes)) != expected_inner:
            raise StudyError(f"fold-lock inner-source hashes differ for seed={seed}, outer={outer}")
    return references


def expected_outer_source_hashes(references: Mapping[str, Any], seed: int, outer: int) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(
        digest
        for reference in references.values()
        if reference.job.stage == "outer"
        and reference.job.training_seed == seed
        and reference.job.outer_fold_id == outer
        for digest in reference_source_hashes(reference)
    ))


def validate_outer_stage_prerequisites(
    *, view: Any, manager: NestedOOFFoldManager, repository: StudyArtifactRepository,
    manifest: Mapping[str, Any],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    locks = repository.validate_complete_lock_matrix(
        plan_sha256=str(manifest["plan_sha256"]),
        source_commit=str(manifest["source_commit"]),
    )
    for lock in locks:
        validate_fold_lock_membership(lock, manager)
    view.planner.validate_complete_lock_matrix(frozen=manifest)
    return locks, validate_complete_job_references(view, locks)


def assemble_inner_dataset(
    view: Any,
    jobs: Mapping[tuple[str, str, int, int, int | None], str],
    manager: NestedOOFFoldManager,
    seed: int,
    outer: int,
) -> InnerOOFDataset:
    """Load and align exactly four held-out inner prediction folds."""
    from scripts.task3e_fixed import EXPERT_KEYS

    logits_chunks: list[np.ndarray] = []
    labels_chunks: list[np.ndarray] = []
    id_chunks: list[np.ndarray] = []
    fold_chunks: list[np.ndarray] = []
    hashes: list[tuple[str, str]] = []
    canonical_labels = dict(zip(manager.canonical_indices, manager.training_labels))
    for inner in range(4):
        expected = np.asarray(manager.inner_fold(outer, inner).prediction_indices, dtype=np.int64)
        expert_rows: list[np.ndarray] = []
        reference_labels: np.ndarray | None = None
        for expert_key, expert_name in zip(EXPERT_KEYS, manager.expert_order):
            prediction = view.load_logits(jobs[("inner", expert_key, seed, outer, inner)])
            if (
                prediction.stage != "inner" or prediction.expert_name != expert_name
                or prediction.training_seed != seed or prediction.outer_fold_id != outer
                or prediction.inner_fold_id != inner
            ):
                raise StudyError("validated inner prediction identity disagrees with its job ID")
            if len(prediction.sample_ids) != len(expected) or set(prediction.sample_ids.tolist()) != set(expected.tolist()):
                raise StudyError("inner prediction membership differs from canonical fold")
            by_id = {int(sample_id): row for row, sample_id in enumerate(prediction.sample_ids)}
            order = np.asarray([by_id[int(sample_id)] for sample_id in expected], dtype=np.int64)
            labels = prediction.labels[order]
            if reference_labels is None:
                reference_labels = labels
            elif not np.array_equal(reference_labels, labels):
                raise StudyError("inner experts disagree on held-out labels")
            expert_rows.append(prediction.logits[order])
            hashes.extend(source_hashes(prediction))
        if reference_labels is None or any(
            int(reference_labels[row]) != int(canonical_labels[int(sample_id)])
            for row, sample_id in enumerate(expected)
        ):
            raise StudyError("inner OOF labels disagree with canonical training labels")
        logits_chunks.append(np.stack(expert_rows, axis=1))
        labels_chunks.append(reference_labels)
        id_chunks.append(expected)
        fold_chunks.append(np.full(len(expected), inner, dtype=np.int64))
    return InnerOOFDataset(
        np.concatenate(logits_chunks), np.concatenate(labels_chunks), np.concatenate(id_chunks),
        np.concatenate(fold_chunks), seed, outer, tuple(sorted(hashes)),
    )


def assemble_outer_batch(
    view: Any,
    jobs: Mapping[tuple[str, str, int, int, int | None], str],
    manager: NestedOOFFoldManager,
    seed: int,
    outer: int,
) -> tuple[InferenceBatch, tuple[tuple[str, str], ...], str]:
    """Load outer logits without exposing labels until all predictions complete."""
    from scripts.task3e_fixed import EXPERT_KEYS

    expected = np.asarray(manager.outer_fold(outer).evaluation_indices, dtype=np.int64)
    experts: list[np.ndarray] = []
    hashes: list[tuple[str, str]] = []
    label_job_id = ""
    for expert_key, expert_name in zip(EXPERT_KEYS, manager.expert_order):
        job_id = jobs[("outer", expert_key, seed, outer, None)]
        prediction = view.load_outer_logits(job_id)
        if (
            prediction.stage != "outer" or prediction.expert_name != expert_name
            or prediction.training_seed != seed or prediction.outer_fold_id != outer
        ):
            raise StudyError("validated outer prediction identity disagrees with its job ID")
        if not np.array_equal(prediction.sample_ids, expected):
            raise StudyError("outer expert predictions are not in canonical evaluation order")
        experts.append(prediction.logits)
        hashes.extend(source_hashes(prediction))
        label_job_id = label_job_id or job_id
    return InferenceBatch(np.stack(experts, axis=1), expected), tuple(sorted(hashes)), label_job_id


def lock_study(
    *, view: Any, manager: NestedOOFFoldManager, repository: StudyArtifactRepository,
    manifest: Mapping[str, Any], manifest_sha256: str,
) -> None:
    """Fit and persist all 15 immutable method locks from complete inner OOF data."""
    jobs = job_id_lookup(view)
    repository.write_study_lock(
        plan_sha256=str(manifest["plan_sha256"]),
        source_commit=str(manifest["source_commit"]),
        job_manifest_sha256=manifest_sha256,
        job_inventory=manifest["inventory"],
    )
    runner = FoldStudyRunner(repository.config)
    class_counts = np.asarray(manager.canonical_class_counts, dtype=np.int64)
    for seed in repository.config.seeds:
        for outer in repository.config.outer_folds:
            dataset = assemble_inner_dataset(view, jobs, manager, seed, outer)
            lock = runner.lock(
                dataset, class_counts, plan_sha256=str(manifest["plan_sha256"]),
                source_commit=str(manifest["source_commit"]),
            )
            validate_fold_lock_membership(lock, manager)
            repository.write_fold_lock(lock)


def evaluate_study(
    *, view: Any, manager: NestedOOFFoldManager, repository: StudyArtifactRepository,
    manifest: Mapping[str, Any],
) -> None:
    """Predict all locked outer methods first; then score canonical train-fold labels."""
    locks, references = validate_outer_stage_prerequisites(
        view=view, manager=manager, repository=repository, manifest=manifest
    )
    jobs = job_id_lookup(view)
    evaluator = StudyEvaluator(repository.config)
    batches: dict[tuple[int, int], tuple[InferenceBatch, tuple[tuple[str, str], ...], str]] = {}
    for seed in repository.config.seeds:
        for outer in repository.config.outer_folds:
            lock = repository.read_fold_lock(seed, outer)
            batch, hashes, label_job_id = assemble_outer_batch(view, jobs, manager, seed, outer)
            if hashes != expected_outer_source_hashes(references, seed, outer):
                raise StudyError(f"outer source hashes differ for seed={seed}, outer={outer}")
            evaluator.predict_locked_methods(lock, batch)
            batches[(seed, outer)] = (batch, hashes, label_job_id)
    canonical_labels = dict(zip(manager.canonical_indices, manager.training_labels))
    counts = np.asarray(manager.canonical_class_counts, dtype=np.int64)
    for seed in repository.config.seeds:
        for outer in repository.config.outer_folds:
            lock = repository.read_fold_lock(seed, outer)
            batch, hashes, label_job_id = batches[(seed, outer)]
            ids, labels = view.load_outer_labels(label_job_id)
            if not np.array_equal(ids, batch.sample_ids):
                raise StudyError("outer labels and inference samples have different canonical order")
            if any(int(labels[row]) != int(canonical_labels[int(sample_id)]) for row, sample_id in enumerate(ids)):
                raise StudyError("outer labels disagree with canonical training labels")
            evaluation = evaluator.evaluate_fold(
                lock, batch, labels, counts, source_hashes=hashes,
                expected_outer_sample_ids=manager.outer_fold(outer).evaluation_indices,
            )
            repository.write_fold_evaluation(
                evaluation, expected_outer_sample_ids=manager.outer_fold(outer).evaluation_indices
            )


def report_study(
    *, view: Any, manager: NestedOOFFoldManager, repository: StudyArtifactRepository,
    manifest: Mapping[str, Any],
) -> None:
    """Revalidate saved nested-OOF evaluations and publish the paper report."""
    _locks, references = validate_outer_stage_prerequisites(
        view=view, manager=manager, repository=repository, manifest=manifest
    )
    expected_outer = {
        fold: np.asarray(manager.outer_fold(fold).evaluation_indices, dtype=np.int64)
        for fold in repository.config.outer_folds
    }
    evaluations = repository.validate_complete_evaluation_matrix(expected_outer_sample_ids=expected_outer)
    canonical_labels = dict(zip(manager.canonical_indices, manager.training_labels))
    for evaluation in evaluations:
        if evaluation.source_hashes != expected_outer_source_hashes(
            references, evaluation.training_seed, evaluation.outer_fold_id
        ):
            raise StudyError("saved outer source hashes differ from validated references")
        if any(
            int(evaluation.labels[row]) != canonical_labels[int(sample_id)]
            for row, sample_id in enumerate(evaluation.sample_ids)
        ):
            raise StudyError("saved outer labels disagree with canonical training labels")
    result = StudyEvaluator(repository.config).aggregate(
        evaluations, np.asarray(manager.canonical_indices, dtype=np.int64),
        np.asarray(manager.canonical_class_counts, dtype=np.int64), expected_outer,
    )
    markdown = MarkdownReportBuilder(repository.config).build(result)
    repository.write_result(result, markdown)


def read_frozen_matrix_identity(
    artifact_root: str | Path,
    config: StudyConfig,
    *,
    expected_manifest: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """Validate the immutable 300-job matrix identity for compatibility callers."""
    from expert_method.ridge_sinkhorn.matrix import MATRIX_SCHEMA_VERSION, canonical_json_bytes

    project_root = Path(__file__).resolve().parents[1]
    manifest_path = Path(artifact_root).expanduser().resolve() / "ridge_sinkhorn_3seed_v1" / "job_manifest.json"
    reader = ArtifactReader(error_type=StudyError)
    manifest = reader.read_json(manifest_path, name="frozen expert-job manifest")
    if manifest_path.read_bytes() != canonical_json_bytes(manifest):
        raise StudyError("expert-job manifest is not canonical JSON")
    plan_path = project_root / "docs" / "PLAN.md"
    plan_sha = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    if manifest.get("schema_version") != MATRIX_SCHEMA_VERSION or manifest.get("study_id") != "ridge_sinkhorn_3seed_v1":
        raise StudyError("expert-job manifest has an unsupported study identity")
    if manifest.get("plan_sha256") != plan_sha or manifest.get("protocol_config_sha256") != config.sha256:
        raise StudyError("expert-job manifest differs from the frozen plan or StudyConfig")
    source_commit = manifest.get("source_commit")
    if not isinstance(source_commit, str) or len(source_commit) not in {40, 64}:
        raise StudyError("expert-job manifest has no exact Git source commit")
    try:
        int(source_commit, 16)
    except ValueError as exc:
        raise StudyError("expert-job manifest source commit is malformed") from exc
    inventory = manifest.get("inventory")
    if manifest.get("inventory_count") != 300 or not isinstance(inventory, list) or len(inventory) != 300:
        raise StudyError("expert-job manifest does not freeze all 300 jobs")
    job_ids = [item.get("job_id") for item in inventory if isinstance(item, Mapping)]
    if len(job_ids) != 300 or len(set(job_ids)) != 300 or job_ids != sorted(job_ids):
        raise StudyError("expert-job manifest job IDs are malformed")
    if expected_manifest is not None:
        fields = (
            "schema_version", "study_id", "study_freeze_sha256", "plan_sha256",
            "study_config_sha256", "protocol_config_sha256", "source_commit",
            "fold_manifest_sha256", "inventory_count", "inventory", "shard_rule",
        )
        for field in fields:
            if manifest.get(field) != expected_manifest.get(field):
                raise StudyError(f"expert-job manifest field {field} differs from canonical identity")
    return manifest, reader.sha256_file(manifest_path, description="expert-job manifest")
