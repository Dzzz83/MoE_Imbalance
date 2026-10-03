"""Read-only validation and loading for frozen inner-study artifacts.

This adapter deliberately uses :class:`OOFMatrixPlanner` in read-only mode and
the existing lock/reuse validators. It does not construct ``StudyArtifactView``:
that class correctly expects its source commit to equal the current source
checkout, while diagnostics need to inspect a result snapshot from a different
commit without changing the original lock contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import numpy as np

from data.nested_oof import NestedOOFFoldManager
from expert_method import cli as study_cli
from expert_method.config import StudyDefinition, load_study
from expert_method.ridge_sinkhorn.matrix import (
    AlignedExpertPrediction,
    ExpertJob,
    OOFMatrixPlanner,
    StudyArtifactView,
    COMPATIBILITY_SCHEMA_VERSION,
    canonical_json_bytes,
    manager_from_fold_manifest,
    sha256_file,
)
from expert_method.ridge_sinkhorn.three_seed_study import (
    StudyArtifactRepository,
    StudyConfig,
    StudyError,
)
from scripts.analysis import ArtifactReader
from scripts.config import TrainingConfig

from .contracts import DiagnosticsError


STUDY_ID = "ridge_sinkhorn_3seed_v1"
SNAPSHOT_MARKER_NAME = ".snapshot-provenance.json"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PLAN_PATH = _PROJECT_ROOT / "docs" / "PLAN.md"


@dataclass(frozen=True)
class DiagnosticsInputs:
    """All validated references needed for inner-only reporting."""

    manager: NestedOOFFoldManager
    planner: OOFMatrixPlanner
    repository: StudyArtifactRepository
    manifest: Mapping[str, Any]
    manifest_sha256: str
    references: Mapping[str, Any]
    locks: Mapping[tuple[int, int], Any]
    snapshot_commit: str
    snapshot_marker_sha256: str | None
    config_file_sha256: str
    validation_counts: Mapping[str, int]


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _require_regular_file(path: Path, *, name: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise DiagnosticsError(f"{name} must be a regular, non-symlink file: {path}")


def _digest(value: Any, *, name: str, lengths: tuple[int, ...] = (64,)) -> str:
    if not isinstance(value, str) or len(value) not in lengths:
        raise DiagnosticsError(f"{name} must be a {', '.join(str(size) for size in lengths)}-character hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise DiagnosticsError(f"{name} is not hexadecimal") from exc
    return value.lower()


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _hash_mismatch(label: str, path: Path, expected: Any, actual: str) -> DiagnosticsError:
    """Describe a failed frozen hash comparison with both values and its path."""
    return DiagnosticsError(
        f"{label} hash mismatch at {path}: expected {expected!r}, computed {actual!r}"
    )


def _require_same_reuse_rows(
    audit_path: Path,
    frozen_rows: Any,
    current_rows: Any,
) -> None:
    """Require exact reuse decisions and identify the first differing job."""
    if frozen_rows == current_rows:
        return
    frozen = frozen_rows if isinstance(frozen_rows, list) else []
    current = current_rows if isinstance(current_rows, list) else []
    row_count = max(len(frozen), len(current))
    for index in range(row_count):
        expected = frozen[index] if index < len(frozen) else None
        actual = current[index] if index < len(current) else None
        if expected == actual:
            continue
        expected_row = expected if isinstance(expected, Mapping) else {}
        actual_row = actual if isinstance(actual, Mapping) else {}
        job_id = actual_row.get("target_job_id", expected_row.get("target_job_id", "<unknown>"))
        differing = sorted(
            key for key in set(expected_row) | set(actual_row)
            if expected_row.get(key) != actual_row.get(key)
        )
        raise DiagnosticsError(
            f"historical reuse decisions differ from the immutable audit at {audit_path}, "
            f"job {job_id!r}: expected source_experiment_id="
            f"{expected_row.get('source_experiment_id')!r}, source_root_name="
            f"{expected_row.get('source_root_name')!r}; resolved source_experiment_id="
            f"{actual_row.get('source_experiment_id')!r}, source_root_name="
            f"{actual_row.get('source_root_name')!r}; differing fields={differing!r}"
        )
    raise DiagnosticsError(
        f"historical reuse audit metadata differs from the immutable audit at {audit_path}"
    )


def _require_pretty_json(path: Path, *, name: str) -> dict[str, Any]:
    """Require the byte representation produced by the immutable study writer."""
    _require_regular_file(path, name=name)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticsError(f"cannot read {name}: {path}") from exc
    if not isinstance(payload, dict):
        raise DiagnosticsError(f"{name} must contain a JSON object: {path}")
    expected = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if path.read_bytes() != expected:
        raise DiagnosticsError(f"{name} is not in immutable-writer JSON format: {path}")
    return payload


def _study_for_config(path: Path) -> tuple[StudyDefinition, StudyConfig]:
    """Use the project's strict study loader and its resolved executable config."""
    _require_regular_file(path, name="study configuration")
    try:
        definition = load_study(path)
        config = definition.to_study_config()
    except Exception as exc:
        raise DiagnosticsError(f"cannot validate study configuration {path}: {exc}") from exc
    return definition, config


def _validate_freeze(
    freeze_path: Path,
    definition: StudyDefinition,
    config: StudyConfig,
    *,
    reuse_root_names: tuple[str, ...],
) -> dict[str, Any]:
    """Validate the immutable study freeze while allowing a different analysis commit."""
    try:
        freeze = study_cli._read_freeze(freeze_path)
    except Exception as exc:
        raise DiagnosticsError(f"cannot validate frozen study record {freeze_path}: {exc}") from exc
    if freeze.get("source_tree_dirty") is not False:
        raise DiagnosticsError(
            f"frozen training source was dirty in the immutable study record: {freeze_path}"
        )
    if freeze.get("study_id") != STUDY_ID:
        raise DiagnosticsError(f"frozen study record identifies a different study: {freeze_path}")
    if freeze.get("source_commit") is None:
        raise DiagnosticsError(f"frozen study record has no training-source commit: {freeze_path}")
    try:
        _digest(freeze.get("source_commit"), name="training-source commit", lengths=(40, 64))
    except DiagnosticsError as exc:
        raise DiagnosticsError(f"invalid training-source commit in frozen study record {freeze_path}: {exc}") from exc
    actual_plan_hash = _sha256(_PLAN_PATH)
    if freeze.get("plan_sha256") != actual_plan_hash:
        raise _hash_mismatch("frozen docs/PLAN.md", _PLAN_PATH, freeze.get("plan_sha256"), actual_plan_hash)
    actual_yaml_hash = _sha256(definition.source_path)
    if freeze.get("study_config_yaml_sha256") != actual_yaml_hash:
        raise _hash_mismatch(
            "study YAML", definition.source_path,
            freeze.get("study_config_yaml_sha256"), actual_yaml_hash,
        )
    if freeze.get("study_config_sha256") != definition.scientific_sha256:
        raise _hash_mismatch(
            "resolved study configuration", definition.source_path,
            freeze.get("study_config_sha256"), definition.scientific_sha256,
        )
    if freeze.get("resolved_study_config") != json.loads(definition.canonical_scientific_json):
        raise DiagnosticsError(
            f"resolved scientific study settings differ from the immutable freeze record: "
            f"{definition.source_path}"
        )
    try:
        expert_hashes = study_cli._expert_config_hashes(definition)
    except Exception as exc:
        raise DiagnosticsError(f"cannot hash resolved expert recipes: {exc}") from exc
    if freeze.get("expert_config_sha256") != expert_hashes:
        recipe_paths = [str(definition.source_path.parents[2] / value)
                        for value in definition.expert_config_paths.values()]
        raise DiagnosticsError(
            f"resolved expert recipe hashes differ at {recipe_paths}: "
            f"expected {freeze.get('expert_config_sha256')!r}, computed {expert_hashes!r}"
        )
    if freeze.get("protocol_config_sha256") != config.sha256:
        raise _hash_mismatch(
            "protocol configuration", definition.source_path,
            freeze.get("protocol_config_sha256"), config.sha256,
        )
    if freeze.get("runtime_root_names") != list(reuse_root_names):
        raise DiagnosticsError(
            f"named reuse roots differ from the immutable freeze at {freeze_path}: "
            f"expected {freeze.get('runtime_root_names')!r}, received {list(reuse_root_names)!r}"
        )
    return freeze


def _validate_stage_audit(planner: OOFMatrixPlanner, freeze: Mapping[str, Any]) -> None:
    """Require the canonical inner reuse audit before resolving native or reused jobs."""
    path = planner.native_store.base_dir / "reuse_compatibility_inner.json"
    _require_regular_file(path, name="frozen inner reuse audit")
    reader = ArtifactReader(error_type=DiagnosticsError)
    audit = reader.read_json(path, name="frozen inner reuse audit")
    if path.read_bytes() != canonical_json_bytes(audit):
        raise DiagnosticsError(f"frozen inner reuse audit is not canonical JSON: {path}")
    expected = {
        "schema_version": COMPATIBILITY_SCHEMA_VERSION,
        "study_id": STUDY_ID,
        "study_freeze_sha256": freeze.get("freeze_sha256"),
        "stage": "inner",
    }
    mismatches = {
        field: {"expected": value, "found": audit.get(field)}
        for field, value in expected.items()
        if audit.get(field) != value
    }
    if mismatches:
        raise DiagnosticsError(f"frozen inner reuse audit identity is invalid at {path}: {mismatches}")


def _snapshot_provenance(artifact_root: Path) -> tuple[str, str | None]:
    """Get the result-snapshot commit from an explicit marker or its Git tree."""
    marker = artifact_root / SNAPSHOT_MARKER_NAME
    if marker.exists() or marker.is_symlink():
        _require_regular_file(marker, name="snapshot provenance marker")
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DiagnosticsError(f"snapshot provenance marker is invalid: {marker}") from exc
        if not isinstance(payload, Mapping):
            raise DiagnosticsError("snapshot provenance marker must be a JSON object")
        if payload.get("schema_version") not in {"diagnostics_snapshot.v1", "artifact_snapshot.v1"}:
            raise DiagnosticsError("snapshot provenance marker has an unsupported schema")
        commit = _digest(payload.get("snapshot_commit"), name="snapshot commit", lengths=(40, 64))
        file_hashes = payload.get("file_hashes", {})
        if not isinstance(file_hashes, Mapping):
            raise DiagnosticsError("snapshot marker file_hashes must be an object")
        for relative, expected_hash in sorted(file_hashes.items()):
            if not isinstance(relative, str):
                raise DiagnosticsError("snapshot file-hash entries require relative paths")
            candidate = (artifact_root / relative).resolve()
            try:
                candidate.relative_to(artifact_root.resolve())
            except ValueError as exc:
                raise DiagnosticsError(f"snapshot file path escapes artifact root: {relative}") from exc
            _require_regular_file(candidate, name=f"snapshot file {relative}")
            expected = _digest(expected_hash, name=f"snapshot file hash {relative}")
            actual = _sha256(candidate)
            if actual != expected:
                raise _hash_mismatch(f"snapshot file {relative}", candidate, expected, actual)
        return commit, _sha256(marker)

    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=artifact_root,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=artifact_root,
            check=True, capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        raise DiagnosticsError(
            f"cannot determine result-snapshot commit; add {SNAPSHOT_MARKER_NAME} under {artifact_root}"
        )
    _digest(commit, name="result-snapshot Git commit", lengths=(40, 64))
    if not top:
        raise DiagnosticsError("result-snapshot Git root is empty")
    if Path(top).resolve() == _PROJECT_ROOT.resolve():
        raise DiagnosticsError(
            f"artifact root resolves to the analysis checkout; add {SNAPSHOT_MARKER_NAME} "
            "with the pinned result-snapshot commit"
        )
    return commit, None


def _validate_matrix_identity(
    manifest: Mapping[str, Any],
    computed: Mapping[str, Any],
    *,
    manifest_path: Path,
    freeze: Mapping[str, Any],
) -> None:
    """Compare every scientific identity field while preserving source provenance."""
    expected_fields = (
        "schema_version", "study_id", "study_freeze_sha256", "plan_sha256",
        "study_config_sha256", "protocol_config_sha256", "fold_manifest_sha256",
        "inventory_count", "inventory", "shard_rule",
    )
    for field in expected_fields:
        if manifest.get(field) != computed.get(field):
            if field.endswith("_sha256"):
                raise _hash_mismatch(
                    f"frozen matrix {field}", manifest_path,
                    manifest.get(field), str(computed.get(field)),
                )
            raise DiagnosticsError(
                f"frozen matrix {field} differs from the canonical analysis protocol at {manifest_path}"
            )
    if manifest.get("schema_version") != "ridge_sinkhorn_matrix.v1" or manifest.get("study_id") != STUDY_ID:
        raise DiagnosticsError(f"unsupported frozen matrix identity: {manifest_path}")
    try:
        _digest(manifest.get("source_commit"), name="training-source commit", lengths=(40, 64))
    except DiagnosticsError as exc:
        raise DiagnosticsError(f"invalid training-source commit in matrix manifest {manifest_path}: {exc}") from exc
    if manifest.get("source_commit") != freeze.get("source_commit"):
        raise DiagnosticsError(
            f"matrix manifest training-source commit differs from the study freeze at {manifest_path}: "
            f"expected {freeze.get('source_commit')!r}, found {manifest.get('source_commit')!r}"
        )
    if manifest.get("study_freeze_sha256") != freeze.get("freeze_sha256"):
        raise _hash_mismatch(
            "matrix study-freeze binding", manifest_path,
            freeze.get("freeze_sha256"), str(manifest.get("study_freeze_sha256")),
        )
    if manifest.get("source_tree_dirty") is not False:
        raise DiagnosticsError(f"frozen matrix training source was dirty: {manifest_path}")
    if manifest.get("inventory_count") != 300 or not isinstance(manifest.get("inventory"), list):
        raise DiagnosticsError("frozen matrix must contain the complete 300-job inventory")
    inventory = manifest["inventory"]
    if len(inventory) != 300:
        raise DiagnosticsError("frozen matrix inventory must contain exactly 300 jobs")
    ids = [item.get("job_id") for item in inventory if isinstance(item, Mapping)]
    if len(ids) != 300 or len(set(ids)) != 300 or ids != sorted(ids):
        raise DiagnosticsError("frozen matrix job IDs are malformed, duplicated, or unsorted")
    if sum(item.get("stage") == "inner" for item in inventory) != 240:
        raise DiagnosticsError("frozen matrix must contain exactly 240 inner jobs")
    if sum(item.get("stage") == "outer" for item in inventory) != 60:
        raise DiagnosticsError("frozen matrix must contain exactly 60 outer jobs")
    try:
        plan_hash = _sha256(_PLAN_PATH)
    except (OSError, ValueError) as exc:
        raise DiagnosticsError("cannot hash the current frozen docs/PLAN.md") from exc
    if manifest.get("plan_sha256") != plan_hash:
        raise _hash_mismatch(
            "frozen matrix docs/PLAN.md", _PLAN_PATH,
            manifest.get("plan_sha256"), plan_hash,
        )


def _validate_manager_protocol(
    manager: NestedOOFFoldManager,
    protocol: Mapping[str, Any],
) -> None:
    """Bind restored fold dimensions and population to the executable study.

    This guards against a self-consistent but wrong fold manifest: the manager
    validates its own serialized dimensions, while this check ties them to the
    reviewed study protocol and canonical CIFAR-100-LT population.
    """
    expected = {
        "canonical population": (len(manager.canonical_indices), protocol["training_samples"]),
        "number of classes": (manager.num_classes, protocol["num_classes"]),
        "fold generation seed": (manager.fold_generation_seed, protocol["fold_generation_seed"]),
        "outer fold count": (manager.outer_fold_count, protocol["outer_folds"]),
        "inner fold count": (manager.inner_fold_count, protocol["inner_folds"]),
        "expert order": (tuple(manager.expert_order), tuple(protocol["expert_order"])),
        "fold algorithm": (manager.manifest().fold_algorithm, protocol["fold_algorithm"]),
    }
    for label, (actual, expected_value) in expected.items():
        if actual != expected_value:
            raise DiagnosticsError(
                f"canonical fold manifest {label} {actual!r} differs from study protocol {expected_value!r}"
            )
    if protocol["dataset"] != "cifar100_lt" or protocol["imbalance_ratio"] != 100:
        raise DiagnosticsError("study protocol is not canonical CIFAR-100-LT at imbalance ratio 100")


class InnerArtifactReader:
    """Validate and load the inner-only portion of a frozen three-seed study.

    Artifact access validates checkpoint, prediction, resolved-config, job,
    fold, lock, membership, and historical reuse hashes. It never resolves an
    outer job or opens an outer prediction/label artifact.
    """

    def __init__(
        self,
        *,
        config_path: str | Path,
        artifact_root: str | Path,
        reuse_roots: Mapping[str, str | Path] | None = None,
        study_config: StudyConfig | None = None,
    ) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        self.definition, loaded_config = _study_for_config(self.config_path)
        if study_config is not None and study_config.to_dict() != loaded_config.to_dict():
            raise DiagnosticsError(
                f"supplied StudyConfig differs from the loaded scientific YAML at {self.config_path}"
            )
        self.config = loaded_config
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        self.reuse_roots = {
            str(name): Path(path).expanduser().resolve()
            for name, path in (reuse_roots or {}).items()
        }
        self._references: dict[str, Any] = {}
        self._validated: DiagnosticsInputs | None = None

    def validate(self) -> DiagnosticsInputs:
        """Validate frozen identities, 15 locks, all 240 inner jobs, and reuse references."""
        if self._validated is not None:
            return self._validated
        study_root = self.artifact_root / STUDY_ID
        manifest_path = study_root / "job_manifest.json"
        fold_manifest_path = study_root / "fold_manifest.json"
        freeze_path = study_root / "manifests" / "study-freeze.json"
        for path, name in (
            (manifest_path, "frozen matrix manifest"),
            (fold_manifest_path, "fold manifest"),
            (freeze_path, "study freeze record"),
        ):
            _require_regular_file(path, name=name)
        freeze = _validate_freeze(
            freeze_path,
            self.definition,
            self.config,
            reuse_root_names=tuple(sorted(self.reuse_roots)),
        )
        config_hash = _sha256(self.config_path)
        reader = ArtifactReader(error_type=DiagnosticsError)
        manifest = reader.read_json(manifest_path, name="frozen matrix manifest")
        if manifest_path.read_bytes() != canonical_json_bytes(manifest):
            raise DiagnosticsError(f"frozen matrix manifest is not canonical JSON: {manifest_path}")
        manifest_sha = reader.sha256_file(manifest_path, description="frozen matrix manifest")
        try:
            manager = manager_from_fold_manifest(fold_manifest_path)
        except (OSError, ValueError, RuntimeError) as exc:
            raise DiagnosticsError(f"cannot reconstruct canonical fold manager: {fold_manifest_path}: {exc}") from exc
        _validate_manager_protocol(manager, self.definition.protocol)
        canonical_fold_text = manager.manifest().to_json().encode("utf-8")
        if fold_manifest_path.read_bytes() != canonical_fold_text:
            raise DiagnosticsError(f"fold manifest is not the canonical serialized membership: {fold_manifest_path}")
        fold_file_hash = hashlib.sha256(canonical_fold_text).hexdigest()
        if fold_file_hash != freeze.get("fold_manifest_sha256"):
            raise _hash_mismatch(
                "canonical fold manifest", fold_manifest_path,
                freeze.get("fold_manifest_sha256"), fold_file_hash,
            )

        repository_root = self.definition.source_path.parents[2]
        try:
            training_configs = {
                expert: TrainingConfig.from_file(repository_root / relative_path)
                for expert, relative_path in self.definition.expert_config_paths.items()
            }
        except Exception as exc:
            raise DiagnosticsError(f"cannot load frozen expert recipe files: {exc}") from exc

        planner = OOFMatrixPlanner(
            manager=manager,
            study_id=self.definition.study_id,
            artifact_root=self.artifact_root,
            reuse_roots=self.reuse_roots,
            training_configs=training_configs,
            study_config=self.config,
            training_epochs=int(self.definition.protocol["epochs"]),
            freeze_sha256=str(freeze["freeze_sha256"]),
            read_only=True,
        )
        try:
            computed_manifest = planner.frozen_manifest()
        except Exception as exc:
            raise DiagnosticsError(f"cannot derive current scientific matrix identity: {exc}") from exc
        _validate_matrix_identity(
            manifest, computed_manifest, manifest_path=manifest_path, freeze=freeze,
        )
        try:
            current_inventory = study_cli._inventory_payload(planner.inventory)
        except Exception as exc:
            raise DiagnosticsError(f"cannot build canonical frozen inventory: {exc}") from exc
        if current_inventory != freeze.get("inventory") or current_inventory != manifest.get("inventory"):
            raise DiagnosticsError(
                f"matrix inventory differs from the frozen inventory at {manifest_path} and {freeze_path}"
            )
        current_inventory_hash = hashlib.sha256(
            study_cli._canonical_json(current_inventory).encode("utf-8")
        ).hexdigest()
        if current_inventory_hash != freeze.get("job_inventory_sha256"):
            raise _hash_mismatch(
                "canonical job inventory", manifest_path,
                freeze.get("job_inventory_sha256"), current_inventory_hash,
            )
        _validate_stage_audit(planner, freeze)
        try:
            actual_reuse_rows = [
                dict(row) for row in planner.audit_compatibility(
                    tuple(job for job in planner.inventory if job.stage == "inner")
                )
            ]
            expected_audit = {
                "schema_version": COMPATIBILITY_SCHEMA_VERSION,
                "study_id": STUDY_ID,
                "study_freeze_sha256": freeze["freeze_sha256"],
                "stage": "inner",
                "rows": actual_reuse_rows,
            }
            audit_path = planner.native_store.base_dir / "reuse_compatibility_inner.json"
            recorded_audit = reader.read_json(audit_path, name="frozen inner reuse audit")
            if recorded_audit != expected_audit:
                _require_same_reuse_rows(
                    audit_path, recorded_audit.get("rows"), actual_reuse_rows,
                )
                raise DiagnosticsError(
                    f"historical reuse audit metadata differs from immutable identity at {audit_path}"
                )
            study_cli._verify_reuse_audit(
                actual_reuse_rows, freeze["reuse_decisions"]["inner"], "inner"
            )
        except DiagnosticsError:
            raise
        except Exception as exc:
            raise DiagnosticsError(f"cannot validate frozen historical reuse rows: {exc}") from exc

        inventory_by_id = {job.job_id: job for job in planner.inventory}
        if len(inventory_by_id) != 300:
            raise DiagnosticsError("read-only planner did not derive all 300 canonical jobs")

        # Resolve each inner artifact exactly once here, retain the validated
        # reference, and compare its hashes against all persisted locks below.
        self._references = {}
        hashes_by_pair: dict[tuple[int, int], list[tuple[str, str]]] = {
            (seed, outer): []
            for seed in self.config.seeds for outer in self.config.outer_folds
        }
        for job in planner.inventory:
            if job.stage != "inner":
                continue
            try:
                reference, state, detail = planner.resolve_reference(job)
            except Exception as exc:
                raise DiagnosticsError(f"invalid inner artifact {job.job_id}: {exc}") from exc
            if reference is None or state not in {"validated", "reused"}:
                raise DiagnosticsError(f"inner artifact {job.job_id} is {state}: {detail}")
            if reference.job != job:
                raise DiagnosticsError(f"inner artifact identity differs from inventory: {job.job_id}")
            self._references[job.job_id] = reference
            key = (job.training_seed, job.outer_fold_id)
            hashes_by_pair[key].extend((
                (f"{job.job_id}/checkpoint", reference.checkpoint_sha256),
                (f"{job.job_id}/prediction", reference.prediction_sha256),
                (f"{job.job_id}/resolved_config", reference.resolved_config_sha256),
            ))
        if len(self._references) != 240:
            raise DiagnosticsError(f"validated {len(self._references)} inner jobs; expected 240")

        repository = StudyArtifactRepository(self.artifact_root, self.config)
        _require_pretty_json(repository.config_path, name="saved study configuration")
        _require_pretty_json(repository.study_lock_path, name="immutable study lock")
        for seed in self.config.seeds:
            for outer in self.config.outer_folds:
                _require_pretty_json(repository.lock_path(seed, outer), name="fold lock")
        try:
            locks = repository.validate_complete_lock_matrix(
                plan_sha256=str(manifest["plan_sha256"]),
                source_commit=str(manifest["source_commit"]),
            )
            planner._validate_lock_memberships(locks)
        except Exception as exc:
            raise DiagnosticsError(f"the complete 15-lock matrix did not validate: {exc}") from exc
        locks_by_pair = {(lock.training_seed, lock.outer_fold_id): lock for lock in locks}
        if len(locks_by_pair) != 15 or set(locks_by_pair) != set(hashes_by_pair):
            raise DiagnosticsError("validated lock matrix does not cover all 15 seed/fold pairs")
        for pair, lock in locks_by_pair.items():
            expected_hashes = tuple(sorted(hashes_by_pair[pair]))
            if tuple(sorted(lock.source_hashes)) != expected_hashes:
                raise DiagnosticsError(f"lock source hashes disagree with inner inputs for seed/fold {pair}")

        snapshot_commit, snapshot_marker_hash = _snapshot_provenance(self.artifact_root)
        native_count = sum(not reference.is_historical_reuse for reference in self._references.values())
        reused_count = len(self._references) - native_count
        counts = {
            "locks": len(locks_by_pair),
            "inner_jobs": len(self._references),
            "native_inner_jobs": native_count,
            "historical_inner_jobs": reused_count,
        }
        self._validated = DiagnosticsInputs(
            manager=manager,
            planner=planner,
            repository=repository,
            manifest=manifest,
            manifest_sha256=manifest_sha,
            references=dict(self._references),
            locks=locks_by_pair,
            snapshot_commit=snapshot_commit,
            snapshot_marker_sha256=snapshot_marker_hash,
            config_file_sha256=config_hash,
            validation_counts=counts,
        )
        return self._validated

    @property
    def job_lookup(self) -> dict[tuple[str, str, int, int, int | None], str]:
        """Map canonical inner job attributes to job IDs for the existing assembler."""
        inputs = self.validate()
        return {
            (job.stage, job.expert_key, job.training_seed, job.outer_fold_id, job.inner_fold_id): job.job_id
            for job in inputs.planner.inventory
        }

    def load_logits(self, job_id: str) -> AlignedExpertPrediction:
        """Read one previously validated inner artifact into aligned arrays."""
        self.validate()
        try:
            reference = self._references[job_id]
        except KeyError as exc:
            raise DiagnosticsError(f"unknown or non-inner artifact requested: {job_id}") from exc
        if reference.job.stage != "inner":
            raise DiagnosticsError(f"outer artifact access is forbidden in inner diagnostics: {job_id}")
        try:
            current_prediction_hash = sha256_file(reference.prediction_path)
        except Exception as exc:
            raise DiagnosticsError(f"cannot recheck validated inner prediction {job_id}: {exc}") from exc
        if current_prediction_hash != reference.prediction_sha256:
            raise DiagnosticsError(f"inner prediction changed after validation: {job_id}")
        # Reuse the existing serialized prediction reader after the planner has
        # validated the file, recipe, checkpoint, membership, and source hashes.
        try:
            sample_ids, labels, logits = StudyArtifactView._arrays(reference)
        except Exception as exc:
            raise DiagnosticsError(f"cannot read validated inner prediction {job_id}: {exc}") from exc
        job = reference.job
        if not np.isfinite(logits).all():
            raise DiagnosticsError(f"inner logits contain non-finite values: {job_id}")
        return AlignedExpertPrediction(
            job_id=job.job_id,
            stage=job.stage,
            expert_key=job.expert_key,
            expert_name=job.expert_name,
            training_seed=job.training_seed,
            outer_fold_id=job.outer_fold_id,
            inner_fold_id=job.inner_fold_id,
            sample_ids=sample_ids,
            labels=labels,
            logits=logits,
            source_experiment_id=reference.source_experiment_id,
            source_run_relative_path=reference.source_run_relative_path,
            training_membership_sha256=job.training_membership_sha256,
            checkpoint_sha256=reference.checkpoint_sha256,
            prediction_sha256=reference.prediction_sha256,
            resolved_config_sha256=reference.resolved_config_sha256,
        )
