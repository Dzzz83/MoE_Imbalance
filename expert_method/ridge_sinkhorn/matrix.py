"""Planning, validation, and portable I/O for the Ridge--Sinkhorn expert matrix.

The module deliberately delegates expert training and fold construction to the
existing :mod:`expert_method.oof.pipeline` and :mod:`data.nested_oof` contracts. Its
job is to inventory the 300 immutable expert runs, validate references to
existing runs, and move native run artifacts between sessions without
rewriting them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from data.nested_oof import FoldManifest, NestedOOFFoldManager, OOFProtocolError
from scripts.config import ConfigError, TrainingConfig
from expert_method.oof.pipeline import (
    OOFArtifactError,
    OOFArtifactMissingError,
    OOFArtifactStore,
    OOFCompletedRun,
    OOFRunContext,
    OOFRunSpec,
    OOF_RUN_SCHEMA_VERSION,
    _canonical_json,
    _sha256_file,
    _sha256_text,
)
from expert_method.ridge_sinkhorn.three_seed_study import StudyConfig


STUDY_ID = "ridge_sinkhorn_3seed_v1"
MATRIX_SCHEMA_VERSION = "ridge_sinkhorn_matrix.v1"
BUNDLE_SCHEMA_VERSION = "ridge_sinkhorn_bundle.v1"
COMPATIBILITY_SCHEMA_VERSION = "ridge_sinkhorn_reuse_compatibility.v1"
DEFAULT_SEEDS = (78, 88, 1034)
DEFAULT_EXPERT_KEYS = ("ce", "logit_adjusted", "balanced_softmax", "mixup")
EXPERT_NAMES = {
    "ce": "CE",
    "logit_adjusted": "LAL",
    "balanced_softmax": "BalancedSoftmax",
    "mixup": "Mixup",
}
HISTORICAL_INNER_EXPERIMENT = "task3c_oof"
HISTORICAL_PILOT_EXPERIMENT = "task3b_pilot_ce_s78_o0_i0"
HISTORICAL_OUTER_EXPERIMENT = "ridge_sinkhorn_outer_s78_o0"
HISTORICAL_EPOCHS = 200
_PLAN_PATH = Path(__file__).resolve().parents[2] / "docs" / "PLAN.md"


class MatrixError(OOFArtifactError):
    """Raised when matrix planning, reuse, or bundle integrity fails."""


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a JSON value in the byte-stable format used by bundles."""
    try:
        return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MatrixError(f"value is not JSON serializable: {value!r}") from exc


def sha256_file(path: str | Path) -> str:
    """Hash a regular file incrementally."""
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise MatrixError(f"expected a regular file while hashing: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise MatrixError(f"cannot hash artifact file: {path}") from exc
    return digest.hexdigest()


def membership_sha256(indices: Iterable[int]) -> str:
    """Return the canonical int64 membership hash used by the fold manager."""
    values = np.asarray(sorted(int(value) for value in indices), dtype="<i8")
    return hashlib.sha256(values.tobytes()).hexdigest()


def shard_key(job_id: str) -> int:
    """Return the unsigned big-endian integer from the first eight SHA bytes."""
    return int.from_bytes(hashlib.sha256(job_id.encode("utf-8")).digest()[:8], "big")


def shard_for_job(job_id: str, shard_count: int) -> int:
    """Return the deterministic shard membership for ``job_id``."""
    if isinstance(shard_count, bool) or not isinstance(shard_count, int) or shard_count < 1:
        raise MatrixError("shard_count must be a positive integer")
    return shard_key(job_id) % shard_count


def _assert_safe_relative_path(value: str, *, field: str = "path") -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise MatrixError(f"{field} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise MatrixError(f"unsafe {field}: {value!r}")
    if path.as_posix() != value:
        raise MatrixError(f"non-canonical {field}: {value!r}")
    return path


def _context_prediction_membership_hash(context: OOFRunContext) -> str:
    return membership_sha256(context.prediction_indices)


@dataclass(frozen=True)
class ExpertJob:
    """One stable expert/seed/fold job in the frozen inventory."""

    job_id: str
    stage: str
    expert_key: str
    expert_name: str
    training_seed: int
    outer_fold_id: int
    inner_fold_id: int | None
    training_membership_sha256: str
    prediction_membership_sha256: str
    training_sample_count: int
    prediction_sample_count: int

    @property
    def role(self) -> str:
        return "outer_evaluation" if self.stage == "outer" else "inner_oof"

    @property
    def run_relative_path(self) -> str:
        inner = "outer_eval" if self.inner_fold_id is None else f"inner_{self.inner_fold_id}"
        return f"expert_{self.expert_name}/seed_{self.training_seed}/outer_{self.outer_fold_id}/{inner}"

    def run_spec(
        self,
        *,
        experiment_id: str = STUDY_ID,
        device: str | None = None,
        epochs: int = HISTORICAL_EPOCHS,
    ) -> OOFRunSpec:
        """Build the existing OOF run spec for this job."""
        return OOFRunSpec(
            experiment_id=experiment_id,
            expert=self.expert_key,
            training_seed=self.training_seed,
            outer_fold_id=self.outer_fold_id,
            inner_fold_id=self.inner_fold_id,
            device=device,
            epochs=epochs,
        )


@dataclass(frozen=True)
class ValidatedJobReference:
    """A validated native or read-only historical run reference."""

    job: ExpertJob
    source_experiment_id: str
    source_run_relative_path: str
    source_root_name: str | None
    run_dir: Path
    checkpoint_path: Path
    prediction_path: Path
    resolved_config_path: Path
    metadata_path: Path
    resolved_config: Mapping[str, Any]
    metadata: Mapping[str, Any]
    context: OOFRunContext
    checkpoint_sha256: str
    prediction_sha256: str
    resolved_config_sha256: str
    manifest_sha256: str

    @property
    def is_historical_reuse(self) -> bool:
        return self.source_experiment_id != STUDY_ID


@dataclass(frozen=True)
class AlignedExpertPrediction:
    """Aligned inner-fold data and provenance for one expert job."""

    job_id: str
    stage: str
    expert_key: str
    expert_name: str
    training_seed: int
    outer_fold_id: int
    inner_fold_id: int | None
    sample_ids: np.ndarray
    labels: np.ndarray
    logits: np.ndarray
    source_experiment_id: str
    source_run_relative_path: str
    training_membership_sha256: str
    checkpoint_sha256: str
    prediction_sha256: str
    resolved_config_sha256: str


@dataclass(frozen=True)
class InferenceExpertPrediction:
    """Outer-fold inference inputs, deliberately without labels."""

    job_id: str
    stage: str
    expert_key: str
    expert_name: str
    training_seed: int
    outer_fold_id: int
    sample_ids: np.ndarray
    logits: np.ndarray
    source_experiment_id: str
    source_run_relative_path: str
    training_membership_sha256: str
    checkpoint_sha256: str
    prediction_sha256: str
    resolved_config_sha256: str


@dataclass(frozen=True)
class MatrixJobStatus:
    """Status of one matrix row as seen by a plan operation."""

    job: ExpertJob
    state: str
    detail: str = ""
    reference: ValidatedJobReference | None = None


@dataclass(frozen=True)
class MatrixPlan:
    """Summary and stable selection returned by a matrix planning pass."""

    study_id: str
    stage: str
    shard_index: int
    shard_count: int
    inventory_count: int
    selected_count: int
    validated_count: int
    reused_count: int
    missing_count: int
    invalid_count: int
    estimated_gpu_hours: float
    estimated_artifact_bytes: int
    available_output_bytes: int
    statuses: tuple[MatrixJobStatus, ...]
    compatibility_table: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a compact JSON-serializable plan summary."""
        return {
            "study_id": self.study_id,
            "stage": self.stage,
            "shard": {"index": self.shard_index, "count": self.shard_count},
            "inventory_count": self.inventory_count,
            "selected_count": self.selected_count,
            "counts": {
                "validated": self.validated_count,
                "reused": self.reused_count,
                "missing": self.missing_count,
                "invalid": self.invalid_count,
            },
            "estimated_gpu_hours": self.estimated_gpu_hours,
            "estimated_artifact_bytes": self.estimated_artifact_bytes,
            "available_output_bytes": self.available_output_bytes,
            "jobs": [
                {
                    "job_id": status.job.job_id,
                    "state": status.state,
                    "detail": status.detail,
                }
                for status in self.statuses
            ],
            "compatibility_table": [dict(row) for row in self.compatibility_table],
        }


def build_job_inventory(
    manager: NestedOOFFoldManager,
    *,
    study_id: str = STUDY_ID,
    seeds: Sequence[int] | None = None,
    expert_keys: Sequence[str] | None = None,
    study_config: StudyConfig | None = None,
    training_epochs: int = HISTORICAL_EPOCHS,
) -> tuple[ExpertJob, ...]:
    """Build and validate the stable 240-inner/60-outer job inventory."""
    study_config = study_config or StudyConfig()
    configured_expert_keys = tuple(
        key for key in DEFAULT_EXPERT_KEYS
        if EXPERT_NAMES[key] in study_config.expert_order
    )
    seeds = tuple(study_config.seeds if seeds is None else seeds)
    expert_keys = tuple(configured_expert_keys if expert_keys is None else expert_keys)
    if study_id != STUDY_ID:
        raise MatrixError(f"study_id is frozen at {STUDY_ID!r}")
    if seeds != tuple(study_config.seeds):
        raise MatrixError(f"training seeds disagree with the resolved StudyConfig: {study_config.seeds}")
    if expert_keys != configured_expert_keys or len(configured_expert_keys) != len(study_config.expert_order):
        raise MatrixError("expert keys/order are frozen for this study")
    if (
        manager.fold_generation_seed != 42
        or tuple(range(manager.outer_fold_count)) != tuple(study_config.outer_folds)
        or tuple(range(manager.inner_fold_count)) != tuple(study_config.inner_folds)
    ):
        raise MatrixError("fold manager does not match the resolved StudyConfig fold inventory")
    if manager.expert_order != tuple(EXPERT_NAMES[key] for key in expert_keys):
        raise MatrixError("fold manager expert order disagrees with the frozen study order")

    jobs: list[ExpertJob] = []
    for seed in seeds:
        for outer_id in range(manager.outer_fold_count):
            outer = manager.outer_fold(outer_id)
            for expert_key in expert_keys:
                expert_name = EXPERT_NAMES[expert_key]
                for inner_id in range(manager.inner_fold_count):
                    context = OOFRunSpec(
                        experiment_id=study_id,
                        expert=expert_key,
                        training_seed=seed,
                        outer_fold_id=outer_id,
                        inner_fold_id=inner_id,
                        epochs=training_epochs,
                    ).resolve(manager)
                    job_id = (
                        f"{study_id}/inner/{expert_key}/seed_{seed}/outer_{outer_id}/inner_{inner_id}"
                    )
                    jobs.append(
                        ExpertJob(
                            job_id=job_id,
                            stage="inner",
                            expert_key=expert_key,
                            expert_name=expert_name,
                            training_seed=int(seed),
                            outer_fold_id=outer_id,
                            inner_fold_id=inner_id,
                            training_membership_sha256=context.training_membership_hash,
                            prediction_membership_sha256=_context_prediction_membership_hash(context),
                            training_sample_count=len(context.training_indices),
                            prediction_sample_count=len(context.prediction_indices),
                        )
                    )
                outer_context = OOFRunSpec(
                    experiment_id=study_id,
                    expert=expert_key,
                    training_seed=seed,
                    outer_fold_id=outer_id,
                    inner_fold_id=None,
                    epochs=training_epochs,
                ).resolve(manager)
                job_id = f"{study_id}/outer/{expert_key}/seed_{seed}/outer_{outer_id}/outer_eval"
                jobs.append(
                    ExpertJob(
                        job_id=job_id,
                        stage="outer",
                        expert_key=expert_key,
                        expert_name=expert_name,
                        training_seed=int(seed),
                        outer_fold_id=outer_id,
                        inner_fold_id=None,
                        training_membership_sha256=outer_context.training_membership_hash,
                        prediction_membership_sha256=_context_prediction_membership_hash(outer_context),
                        training_sample_count=len(outer_context.training_indices),
                        prediction_sample_count=len(outer_context.prediction_indices),
                    )
                )
    inventory = tuple(sorted(jobs, key=lambda job: job.job_id))
    if len(inventory) != 300 or len({job.job_id for job in inventory}) != 300:
        raise MatrixError("frozen inventory must contain exactly 300 unique jobs")
    if sum(job.stage == "inner" for job in inventory) != 240:
        raise MatrixError("frozen inventory must contain exactly 240 inner jobs")
    if sum(job.stage == "outer" for job in inventory) != 60:
        raise MatrixError("frozen inventory must contain exactly 60 outer jobs")
    return inventory


def parse_reuse_roots(values: Sequence[str]) -> dict[str, Path]:
    """Parse repeatable ``NAME=PATH`` roots or derive stable names from basenames."""
    roots: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if separator:
            if not name or not raw_path:
                raise MatrixError("--reuse-root NAME=PATH requires both a name and path")
            root_name, path = name, Path(raw_path)
        else:
            path = Path(value)
            root_name = path.name or "reuse_root"
        if not root_name or not root_name.replace("-", "").replace("_", "").replace(".", "").isalnum():
            raise MatrixError(f"reuse-root name must be path-safe: {root_name!r}")
        if root_name in roots and roots[root_name].resolve() != path.resolve():
            raise MatrixError(f"reuse-root logical name is ambiguous: {root_name!r}")
        roots[root_name] = path.expanduser().resolve()
    return roots


def _readonly_store(
    *, root: Path, manager: NestedOOFFoldManager, experiment_id: str
) -> OOFArtifactStore:
    """Create a validated OOF store that cannot create or mutate artifacts."""
    return OOFArtifactStore(
        root=root,
        manager=manager,
        experiment_id=experiment_id,
        read_only=True,
    )


def _find_experiment_base(root: Path, experiment_id: str) -> Path | None:
    """Resolve the historical experiment beneath a logical mounted root."""
    candidates = (
        root if root.name == experiment_id else None,
        root / experiment_id,
        root / "artifacts" / "oof" / experiment_id,
        root / "oof" / experiment_id,
    )
    for candidate in candidates:
        if candidate is not None and candidate.is_dir():
            return candidate.resolve()
    return None


def _normalized_training_recipe(config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove machine-specific placement and seed from an expert recipe."""
    normalized = json.loads(_canonical_json(dict(config)))
    normalized.pop("device", None)
    normalized.pop("resolved_device", None)
    normalized.pop("seed", None)
    data = normalized.get("data")
    if isinstance(data, dict):
        data["root"] = "<data-root>"
    checkpoint = normalized.get("checkpoint")
    if isinstance(checkpoint, dict):
        checkpoint["dir"] = "<run-checkpoint-dir>"
    return normalized


def _source_experiment_for(job: ExpertJob) -> str | None:
    if job.training_seed != 78 or job.outer_fold_id != 0:
        return None
    if job.stage == "outer":
        return HISTORICAL_OUTER_EXPERIMENT
    if job.expert_key == "ce" and job.inner_fold_id == 0:
        # Task 3C's manifest records this fold as read-only reuse from the pilot.
        return HISTORICAL_PILOT_EXPERIMENT
    return HISTORICAL_INNER_EXPERIMENT


def _source_candidate_roots(
    roots: Mapping[str, Path], experiment_id: str
) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for name, root in sorted(roots.items()):
        base = _find_experiment_base(root, experiment_id)
        if base is not None:
            found.append((name, base))
    return found


def _source_prediction_path(context: OOFRunContext, store: OOFArtifactStore) -> Path:
    return store.prediction_path(context)


class OOFMatrixPlanner:
    """Plan the expert matrix and validate native and historical OOF runs."""

    def __init__(
        self,
        *,
        manager: NestedOOFFoldManager,
        study_id: str = STUDY_ID,
        artifact_root: str | Path = "artifacts/oof",
        reuse_roots: Mapping[str, str | Path] | None = None,
        training_configs: Mapping[str, TrainingConfig] | None = None,
        study_config: StudyConfig | None = None,
        training_epochs: int | None = None,
        freeze_sha256: str | None = None,
        config_root: str | Path = Path(__file__).resolve().parents[2] / "configs" / "experts",
        read_only: bool = False,
    ) -> None:
        self.manager = manager
        self.study_id = study_id
        self.study_config = study_config or StudyConfig()
        self.freeze_sha256 = freeze_sha256
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        self.reuse_roots = {
            name: Path(path).expanduser().resolve()
            for name, path in (reuse_roots or {}).items()
        }
        if training_configs is None:
            training_configs = self._load_configs(config_root)
        self.training_configs = dict(training_configs)
        if set(self.training_configs) != set(DEFAULT_EXPERT_KEYS):
            raise MatrixError("training_configs must contain the four frozen expert configs")
        configured_epochs = {config.schedule.epochs for config in self.training_configs.values()}
        if len(configured_epochs) != 1:
            raise MatrixError("all expert configs must use the same frozen epoch count")
        resolved_epochs = next(iter(configured_epochs)) if training_epochs is None else training_epochs
        if isinstance(resolved_epochs, bool) or not isinstance(resolved_epochs, int) or resolved_epochs < 1:
            raise MatrixError("training_epochs must be a positive integer")
        self.training_epochs = resolved_epochs
        for expert_key, config in self.training_configs.items():
            if config.expert != expert_key:
                raise MatrixError(f"training config for {expert_key} declares expert {config.expert!r}")
            if config.schedule.epochs != self.training_epochs:
                raise MatrixError(f"{expert_key} training config must use {self.training_epochs} epochs")
        self.inventory = build_job_inventory(
            manager, study_id=study_id, study_config=self.study_config,
            training_epochs=self.training_epochs,
        )
        self.read_only = bool(read_only)
        if self.read_only:
            self.native_store = _readonly_store(
                root=self.artifact_root,
                manager=manager,
                experiment_id=study_id,
            )
        else:
            self.native_store = OOFArtifactStore(
                root=self.artifact_root,
                manager=manager,
                experiment_id=study_id,
            )
        self._compatibility_cache: dict[str, tuple[ValidatedJobReference | None, dict[str, Any]]] = {}
        self._locks_validated = False

    @staticmethod
    def _load_configs(config_root: str | Path) -> dict[str, TrainingConfig]:
        config_root = Path(config_root)
        names = {
            "ce": "ce.yaml",
            "logit_adjusted": "lal.yaml",
            "balanced_softmax": "balanced_softmax.yaml",
            "mixup": "mixup.yaml",
        }
        output: dict[str, TrainingConfig] = {}
        for key, filename in names.items():
            try:
                output[key] = TrainingConfig.from_file(config_root / filename)
            except (ConfigError, OSError) as exc:
                raise MatrixError(f"cannot load frozen {key} config {config_root / filename}: {exc}") from exc
        return output

    def jobs(
        self,
        *,
        stage: str | None = None,
        shard_index: int = 0,
        shard_count: int = 1,
    ) -> tuple[ExpertJob, ...]:
        """Return sorted jobs selected by stage and deterministic shard."""
        if stage not in (None, "inner", "outer"):
            raise MatrixError("stage must be 'inner', 'outer', or None")
        if isinstance(shard_index, bool) or not isinstance(shard_index, int):
            raise MatrixError("shard_index must be an integer")
        if shard_count < 1 or shard_index < 0 or shard_index >= shard_count:
            raise MatrixError("shard requires 0 <= shard_index < shard_count")
        return tuple(
            job
            for job in self.inventory
            if (stage is None or job.stage == stage)
            and shard_for_job(job.job_id, shard_count) == shard_index
        )

    def resolved_config(
        self,
        job: ExpertJob,
        *,
        device: str | None = None,
        data_root: str | Path | None = None,
    ) -> dict[str, Any]:
        """Return the exact config payload ``OOFPipeline`` will persist."""
        context = job.run_spec(
            experiment_id=self.study_id, device=device, epochs=self.training_epochs
        ).resolve(self.manager)
        base = self.training_configs[job.expert_key]
        if data_root is not None:
            import dataclasses

            base = dataclasses.replace(
                base,
                data=dataclasses.replace(base.data, root=str(data_root)),
            )
        config = base.replace(
            seed=job.training_seed,
            device=device,
            epochs=self.training_epochs,
            checkpoint_dir=str(self.native_store.run_dir(context) / "checkpoints"),
        )
        payload = config.to_dict()
        payload["resolved_device"] = config.resolved_device
        return payload

    def _validate_native(self, job: ExpertJob) -> ValidatedJobReference:
        context = job.run_spec(
            experiment_id=self.study_id, epochs=self.training_epochs
        ).resolve(self.manager)
        completed = self.native_store.validate_completed_run(context)
        self._validate_job_identity(job, completed, expected_experiment=self.study_id)
        self._validate_recipe(completed.resolved_config, job)
        return self._reference(job, self.native_store, completed, source_root_name=None)

    def _validate_recipe(self, config: Mapping[str, Any], job: ExpertJob) -> None:
        expected = self.training_configs[job.expert_key].to_dict()
        expected["seed"] = job.training_seed
        expected["schedule"] = dict(expected["schedule"])
        expected["schedule"]["epochs"] = self.training_epochs
        if _normalized_training_recipe(config) != _normalized_training_recipe(expected):
            raise MatrixError("resolved expert recipe differs from the frozen training config")
        schedule = config.get("schedule")
        if not isinstance(schedule, Mapping) or schedule.get("epochs") != self.training_epochs:
            raise MatrixError("completed expert run does not use the frozen final epoch")
        if config.get("seed") != job.training_seed:
            raise MatrixError("resolved config seed differs from the job identity")

    def _validate_job_identity(
        self,
        job: ExpertJob,
        completed: OOFCompletedRun,
        *,
        expected_experiment: str,
    ) -> None:
        context = completed.context
        if context.spec.experiment_id != expected_experiment:
            raise MatrixError("run context has an unexpected experiment identity")
        if (
            context.expert_key != job.expert_key
            or context.expert_name != job.expert_name
            or context.training_seed != job.training_seed
            or context.outer_fold_id != job.outer_fold_id
            or context.inner_fold_id != job.inner_fold_id
            or context.role != job.role
        ):
            raise MatrixError("OOF run identity does not match the frozen job")
        if context.training_membership_hash != job.training_membership_sha256:
            raise MatrixError("OOF run training membership hash does not match inventory")
        if membership_sha256(context.prediction_indices) != job.prediction_membership_sha256:
            raise MatrixError("OOF run prediction membership hash does not match inventory")

    def _reference(
        self,
        job: ExpertJob,
        store: OOFArtifactStore,
        completed: OOFCompletedRun,
        *,
        source_root_name: str | None,
    ) -> ValidatedJobReference:
        checkpoint_sha = sha256_file(completed.checkpoint_path)
        prediction_sha = sha256_file(completed.prediction_path)
        config_path = store.config_path(completed.context)
        metadata_path = store.metadata_path(completed.context)
        return ValidatedJobReference(
            job=job,
            source_experiment_id=store.experiment_id,
            source_run_relative_path=job.run_relative_path,
            source_root_name=source_root_name,
            run_dir=completed.run_dir,
            checkpoint_path=completed.checkpoint_path,
            prediction_path=completed.prediction_path,
            resolved_config_path=config_path,
            metadata_path=metadata_path,
            resolved_config=completed.resolved_config,
            metadata=completed.metadata,
            context=completed.context,
            checkpoint_sha256=checkpoint_sha,
            prediction_sha256=prediction_sha,
            resolved_config_sha256=str(completed.metadata["resolved_config_sha256"]),
            manifest_sha256=str(completed.metadata["manifest_sha256"]),
        )

    def _validate_historical(
        self, job: ExpertJob, source_experiment_id: str
    ) -> tuple[ValidatedJobReference | None, dict[str, Any]]:
        if job.stage == "outer" and not self._locks_validated:
            self.validate_complete_lock_matrix()
        table: dict[str, Any] = {
            "target_job_id": job.job_id,
            "status": "missing",
            "source_experiment_id": source_experiment_id,
            "source_root_name": None,
            "source_run_relative_path": job.run_relative_path,
            "expert": job.expert_key,
            "seed": job.training_seed,
            "fold_role": job.role,
            "outer_fold_id": job.outer_fold_id,
            "inner_fold_id": job.inner_fold_id,
            "training_membership_sha256": None,
            "prediction_membership_sha256": None,
            "resolved_config_sha256": None,
            "final_epoch": None,
            "checkpoint_sha256": None,
            "prediction_sha256": None,
            "reason": "historical run was not found under the supplied reuse roots",
        }
        candidates = _source_candidate_roots(self.reuse_roots, source_experiment_id)
        if not candidates:
            return None, table
        if len(candidates) > 1:
            # Multiple mounts are acceptable only when they resolve to identical bytes.
            # Validate all of them below and reject ambiguity if hashes diverge.
            validated: list[tuple[str, ValidatedJobReference]] = []
        else:
            validated = []

        spec = job.run_spec(experiment_id=source_experiment_id, epochs=self.training_epochs)
        context = spec.resolve(self.manager)
        last_reason = "historical run did not validate"
        for root_name, source_base in candidates:
            source_root = source_base.parent
            store = _readonly_store(
                root=source_root,
                manager=self.manager,
                experiment_id=source_experiment_id,
            )
            try:
                completed = store.validate_completed_run(context)
                self._validate_job_identity(job, completed, expected_experiment=source_experiment_id)
                self._validate_recipe(completed.resolved_config, job)
                ref = self._reference(job, store, completed, source_root_name=root_name)
                validated.append((root_name, ref))
            except (OOFArtifactError, OOFProtocolError, OSError, RuntimeError, ValueError) as exc:
                # Keep mounted machine paths out of frozen compatibility
                # records and portable bundles.
                last_reason = f"source artifact failed validation ({type(exc).__name__})"
        if not validated:
            table["reason"] = last_reason
            return None, table
        identities = {
            (ref.checkpoint_sha256, ref.prediction_sha256, ref.resolved_config_sha256)
            for _, ref in validated
        }
        if len(identities) != 1:
            table["reason"] = "supplied reuse roots contain conflicting source bytes"
            return None, table
        root_name, reference = validated[0]
        record = reference.metadata.get("checkpoint", {})
        table.update(
            {
                "status": "compatible",
                "source_root_name": root_name,
                "training_membership_sha256": job.training_membership_sha256,
                "prediction_membership_sha256": job.prediction_membership_sha256,
                "resolved_config_sha256": reference.resolved_config_sha256,
                "final_epoch": record.get("epoch") if isinstance(record, Mapping) else None,
                "checkpoint_sha256": reference.checkpoint_sha256,
                "prediction_sha256": reference.prediction_sha256,
                "manifest_sha256": reference.manifest_sha256,
                "reason": "exact fold membership, recipe, final checkpoint, and predictions validated",
            }
        )
        return reference, table

    def resolve_reference(
        self, job: ExpertJob, *, include_historical: bool = True
    ) -> tuple[ValidatedJobReference | None, str, str]:
        """Resolve a job to a validated native or explicitly eligible historical run."""
        if job.stage == "outer" and not self._locks_validated:
            self.validate_complete_lock_matrix()
        run_dir = self.native_store.run_dir(
            job.run_spec(experiment_id=self.study_id, epochs=self.training_epochs).resolve(self.manager)
        )
        if run_dir.exists() or run_dir.is_symlink():
            return self._validate_native(job), "validated", "native artifact validated"
        source_experiment = _source_experiment_for(job) if include_historical else None
        if source_experiment is None:
            return None, "missing", "no native artifact or eligible historical source"
        cache_key = job.job_id
        if cache_key not in self._compatibility_cache:
            self._compatibility_cache[cache_key] = self._validate_historical(job, source_experiment)
        reference, row = self._compatibility_cache[cache_key]
        frozen_path = self.native_store.base_dir / f"reuse_compatibility_{job.stage}.json"
        if frozen_path.is_file():
            try:
                frozen_audit = json.loads(frozen_path.read_text())
                frozen_rows = frozen_audit.get("rows", [])
                frozen_row = next(
                    item for item in frozen_rows
                    if isinstance(item, Mapping) and item.get("target_job_id") == job.job_id
                )
            except (OSError, json.JSONDecodeError, StopIteration, AttributeError) as exc:
                raise MatrixError(
                    f"frozen historical compatibility reference is missing or invalid for {job.job_id}"
                ) from exc
            if frozen_row.get("status") != "compatible":
                # A source that failed the frozen pre-run audit remains a new
                # job even if a similarly named directory appears later.
                return None, "missing", str(frozen_row.get("reason", "historical source was not frozen as compatible"))
            if reference is None:
                raise MatrixError(
                    f"frozen historical reuse reference cannot be resolved for {job.job_id}"
                )
            expected_identity = (
                frozen_row.get("source_experiment_id"),
                frozen_row.get("source_root_name"),
                frozen_row.get("source_run_relative_path"),
                frozen_row.get("training_membership_sha256"),
                frozen_row.get("prediction_membership_sha256"),
                frozen_row.get("resolved_config_sha256"),
                frozen_row.get("checkpoint_sha256"),
                frozen_row.get("prediction_sha256"),
            )
            actual_identity = (
                reference.source_experiment_id,
                reference.source_root_name,
                reference.source_run_relative_path,
                job.training_membership_sha256,
                job.prediction_membership_sha256,
                reference.resolved_config_sha256,
                reference.checkpoint_sha256,
                reference.prediction_sha256,
            )
            if expected_identity != actual_identity:
                raise MatrixError(
                    f"historical reuse reference no longer matches its frozen hashes for {job.job_id}"
                )
        if reference is not None:
            return reference, "reused", "validated historical reference"
        return None, "missing", str(row.get("reason", "historical run is not compatible"))

    def audit_compatibility(self, jobs: Sequence[ExpertJob] | None = None) -> tuple[Mapping[str, Any], ...]:
        """Build the required row-by-row compatibility table for historical jobs."""
        selected = self.inventory if jobs is None else tuple(jobs)
        if any(job.stage == "outer" for job in selected) and not self._locks_validated:
            self.validate_complete_lock_matrix()
        output: list[Mapping[str, Any]] = []
        for job in selected:
            source_experiment = _source_experiment_for(job)
            if source_experiment is None:
                continue
            if job.job_id not in self._compatibility_cache:
                self._compatibility_cache[job.job_id] = self._validate_historical(job, source_experiment)
            _reference, row = self._compatibility_cache[job.job_id]
            output.append(row)
        return tuple(output)

    def plan(
        self,
        *,
        stage: str,
        shard_index: int = 0,
        shard_count: int = 1,
        output_path: str | Path | None = None,
        max_jobs: int | None = None,
    ) -> MatrixPlan:
        """Validate selected jobs and estimate historical runtime/storage needs."""
        if stage not in ("inner", "outer"):
            raise MatrixError("stage must be 'inner' or 'outer'")
        selected = self.jobs(stage=stage, shard_index=shard_index, shard_count=shard_count)
        frozen = self.frozen_manifest()
        if stage == "outer":
            # Lock completeness is established before any outer prediction
            # artifact can be opened or referenced.
            self.validate_complete_lock_matrix(frozen=frozen)
        if max_jobs is not None and max_jobs < 1:
            raise MatrixError("max_jobs must be positive")
        statuses: list[MatrixJobStatus] = []
        for job in selected:
            try:
                reference, state, detail = self.resolve_reference(job)
            except (OOFArtifactError, OOFProtocolError, OSError, RuntimeError, ValueError) as exc:
                state, detail, reference = "invalid", str(exc), None
            statuses.append(MatrixJobStatus(job=job, state=state, detail=detail, reference=reference))
        stage_jobs = tuple(job for job in self.inventory if job.stage == stage)
        table = self.audit_compatibility(stage_jobs)
        invalid_count = sum(status.state == "invalid" for status in statuses)
        validated_count = sum(status.state == "validated" for status in statuses)
        reused_count = sum(status.state == "reused" for status in statuses)
        missing_count = sum(status.state == "missing" for status in statuses)
        runnable_missing = [status for status in statuses if status.state == "missing"]
        if max_jobs is not None:
            selected_count = min(max_jobs, len(runnable_missing))
        else:
            selected_count = len(runnable_missing)

        runtime_hours, per_run_bytes = self._historical_estimates(stage)
        estimated_hours = runtime_hours * selected_count
        estimated_bytes = per_run_bytes * selected_count
        target_path = Path(output_path or self.artifact_root)
        try:
            available_bytes = shutil.disk_usage(target_path if target_path.exists() else target_path.parent).free
        except OSError:
            available_bytes = 0
        return MatrixPlan(
            study_id=self.study_id,
            stage=stage,
            shard_index=shard_index,
            shard_count=shard_count,
            inventory_count=len(selected),
            selected_count=selected_count,
            validated_count=validated_count,
            reused_count=reused_count,
            missing_count=missing_count,
            invalid_count=invalid_count,
            estimated_gpu_hours=estimated_hours,
            estimated_artifact_bytes=estimated_bytes,
            available_output_bytes=int(available_bytes),
            statuses=tuple(statuses),
            compatibility_table=table,
        )

    def _historical_estimates(self, stage: str) -> tuple[float, int]:
        """Use historical training-history timing and run sizes where available."""
        if stage == "outer" and not self._locks_validated:
            self.validate_complete_lock_matrix()
        runtime_samples: list[float] = []
        byte_samples: list[int] = []
        candidate_jobs = [
            job for job in self.inventory
            if job.stage == stage and _source_experiment_for(job) is not None
        ]
        for job in candidate_jobs:
            source_experiment = _source_experiment_for(job)
            assert source_experiment is not None
            for _root_name, base in _source_candidate_roots(self.reuse_roots, source_experiment):
                run_dir = base / job.run_relative_path
                history_path = run_dir / "training_history.json"
                checkpoint_path = run_dir / "checkpoints" / f"{job.expert_name}_seed{job.training_seed}_final.pt"
                prediction_path = run_dir / "predictions.json"
                try:
                    history = json.loads(history_path.read_text())
                    elapsed = sum(float(row["time_s"]) for row in history if isinstance(row, Mapping) and "time_s" in row)
                    if len(history) == self.training_epochs and np.isfinite(elapsed) and elapsed > 0:
                        runtime_samples.append(elapsed / 3600.0)
                    if checkpoint_path.is_file() and prediction_path.is_file():
                        byte_samples.append(
                            sum(
                                path.stat().st_size
                                for path in run_dir.rglob("*")
                                if path.is_file() and not path.is_symlink()
                            )
                        )
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    continue
                if runtime_samples and byte_samples:
                    break
        # The plan records historical project-rate estimates (~20 min/job) if
        # no representative histories are mounted in this planning session.
        runtime_hours = float(np.median(runtime_samples)) if runtime_samples else (1.0 / 3.0)
        if byte_samples:
            per_run_bytes = int(np.median(byte_samples))
        else:
            per_run_bytes = 0
        return runtime_hours, per_run_bytes

    def compatibility_payload(self, *, stage: str = "inner") -> dict[str, Any]:
        """Return the machine-readable historical reuse audit for one stage."""
        if stage not in {"inner", "outer"}:
            raise MatrixError("compatibility audit stage must be 'inner' or 'outer'")
        return {
            "schema_version": COMPATIBILITY_SCHEMA_VERSION,
            "study_id": self.study_id,
            "study_freeze_sha256": self.freeze_sha256,
            "stage": stage,
            "rows": [
                dict(row)
                for row in self.audit_compatibility(
                    tuple(job for job in self.inventory if job.stage == stage)
                )
            ],
        }

    def frozen_manifest(self) -> dict[str, Any]:
        """Build the immutable study lock and full job/shard inventory."""
        try:
            plan_sha = sha256_file(_PLAN_PATH)
        except MatrixError as exc:
            raise MatrixError(f"cannot hash frozen plan {_PLAN_PATH}: {exc}") from exc
        try:
            source_commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=_PLAN_PATH.parents[1],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            source_commit = "unavailable"

        recipe_payload = {
            "study_config": self.study_config.to_dict(),
            "study_config_sha256": self.study_config.sha256,
            "fold_manifest_sha256": _sha256_text(self.manager.manifest().to_json()),
            "training_recipes": {
                key: _normalized_training_recipe(config.to_dict())
                for key, config in sorted(self.training_configs.items())
            },
        }
        study_config_sha = _sha256_text(_canonical_json(recipe_payload))
        try:
            plan_text = _PLAN_PATH.read_bytes()
        except OSError:
            plan_text = b""
        inventory = [
            {
                **asdict(job),
                "run_relative_path": job.run_relative_path,
                "shard_key": f"{shard_key(job.job_id):016x}",
                "shard_rule": "first_8_sha256_bytes_big_endian_mod_shard_count",
            }
            for job in self.inventory
        ]
        return {
            "schema_version": MATRIX_SCHEMA_VERSION,
            "study_id": self.study_id,
            "study_freeze_sha256": self.freeze_sha256,
            "plan_sha256": hashlib.sha256(plan_text).hexdigest(),
            "study_config_sha256": study_config_sha,
            "protocol_config_sha256": self.study_config.sha256,
            "source_commit": source_commit,
            "source_tree_dirty": self._source_tree_dirty(),
            "fold_manifest_sha256": _sha256_text(self.manager.manifest().to_json()),
            "inventory_count": len(inventory),
            "inventory": inventory,
            "shard_rule": "unsigned_big_endian(first_8_bytes(SHA-256(job_id))) % shard_count",
        }

    @staticmethod
    def _source_tree_dirty() -> bool:
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=_PLAN_PATH.parents[1],
                check=True,
                capture_output=True,
                text=True,
            )
            return bool(result.stdout.strip())
        except (OSError, subprocess.CalledProcessError):
            return True

    def validate_complete_lock_matrix(
        self, *, frozen: Mapping[str, Any] | None = None
    ) -> tuple[Any, ...]:
        """Validate locks against canonical folds and all inner source artifacts."""
        if frozen is None:
            frozen = self.frozen_manifest()
        try:
            from expert_method.ridge_sinkhorn.three_seed_study import StudyArtifactRepository

            repository = StudyArtifactRepository(
                artifact_root=self.artifact_root,
                config=self.study_config,
            )
            locks = repository.validate_complete_lock_matrix(
                plan_sha256=str(frozen["plan_sha256"]),
                source_commit=str(frozen["source_commit"]),
            )
        except (ImportError, AttributeError, TypeError) as exc:
            raise MatrixError(f"study lock repository is unavailable: {exc}") from exc
        self._validate_lock_memberships(locks)
        expected_source_hashes = self._validated_inner_source_hashes()
        for lock in locks:
            pair = (int(lock.training_seed), int(lock.outer_fold_id))
            actual = tuple(sorted((str(name), str(digest)) for name, digest in lock.source_hashes))
            if actual != expected_source_hashes[pair]:
                raise MatrixError(
                    "fold-lock inner source hashes differ from validated references "
                    f"for seed={pair[0]}, outer={pair[1]}"
                )
        self._locks_validated = True
        return locks

    def _validate_lock_memberships(self, locks: Sequence[Any]) -> None:
        """Require each lock's four validation sets to equal the canonical folds."""
        expected_pairs = {
            (seed, outer_fold)
            for seed in self.study_config.seeds
            for outer_fold in self.study_config.outer_folds
        }
        locks_by_pair: dict[tuple[int, int], Any] = {}
        for lock in locks:
            pair = (int(lock.training_seed), int(lock.outer_fold_id))
            if pair not in expected_pairs or pair in locks_by_pair:
                raise MatrixError(f"study lock matrix has an unknown or duplicate fold: {pair}")
            locks_by_pair[pair] = lock
        if set(locks_by_pair) != expected_pairs:
            raise MatrixError("study lock matrix does not cover all canonical seed/outer pairs")

        for (_seed, outer_fold), lock in locks_by_pair.items():
            memberships = tuple(lock.inner_fold_memberships)
            if tuple(item.inner_fold_id for item in memberships) != self.study_config.inner_folds:
                raise MatrixError(
                    f"fold lock has incomplete inner memberships for outer fold {outer_fold}"
                )
            union: list[int] = []
            for inner_fold, membership in enumerate(memberships):
                canonical = tuple(
                    int(sample_id)
                    for sample_id in self.manager.inner_fold(outer_fold, inner_fold).prediction_indices
                )
                recorded = tuple(int(sample_id) for sample_id in membership.validation_sample_ids)
                if recorded != canonical:
                    raise MatrixError(
                        "fold-lock validation membership differs from the canonical inner fold "
                        f"for outer={outer_fold}, inner={inner_fold}"
                    )
                union.extend(recorded)
            canonical_outer_training = tuple(
                int(sample_id)
                for sample_id in self.manager.outer_fold(outer_fold).expert_training_indices
            )
            if tuple(sorted(union)) != canonical_outer_training:
                raise MatrixError(
                    f"fold-lock inner membership union differs from canonical outer training for outer={outer_fold}"
                )

    def _validated_inner_source_hashes(
        self,
    ) -> dict[tuple[int, int], tuple[tuple[str, str], ...]]:
        """Resolve only inner jobs and collect their immutable source hashes."""
        jobs_by_pair: dict[tuple[int, int], list[ExpertJob]] = {
            (seed, outer_fold): []
            for seed in self.study_config.seeds
            for outer_fold in self.study_config.outer_folds
        }
        for job in self.inventory:
            if job.stage == "inner":
                jobs_by_pair[(job.training_seed, job.outer_fold_id)].append(job)

        expected: dict[tuple[int, int], tuple[tuple[str, str], ...]] = {}
        for pair, jobs in jobs_by_pair.items():
            if len(jobs) != 16:
                raise MatrixError(f"canonical inner job inventory is incomplete for seed/outer={pair}")
            entries: list[tuple[str, str]] = []
            for job in jobs:
                try:
                    reference, state, detail = self.resolve_reference(job)
                except (OOFArtifactError, OOFProtocolError, OSError, RuntimeError, ValueError) as exc:
                    raise MatrixError(f"cannot validate inner source for {job.job_id}: {exc}") from exc
                if reference is None or state not in {"validated", "reused"}:
                    raise MatrixError(f"inner source is {state} for {job.job_id}: {detail}")
                for name, value in (
                    ("checkpoint", reference.checkpoint_sha256),
                    ("prediction", reference.prediction_sha256),
                    ("resolved_config", reference.resolved_config_sha256),
                ):
                    entries.append(
                        (f"{job.job_id}/{name}", _valid_digest(value, name=f"{name} hash for {job.job_id}"))
                    )
            if len(entries) != 48:
                raise MatrixError(f"inner source hash inventory is incomplete for seed/outer={pair}")
            expected[pair] = tuple(sorted(entries))
        return expected

    def freeze(self, *, stage: str = "inner", freeze_sha256: str | None = None) -> Path:
        """Freeze inventory and the requested stage's historical reuse audit."""
        if self.read_only:
            raise MatrixError("a read-only matrix planner cannot freeze artifacts")
        if stage not in {"inner", "outer"}:
            raise MatrixError("freeze stage must be 'inner' or 'outer'")
        if freeze_sha256 is not None:
            self.freeze_sha256 = _valid_digest(freeze_sha256, name="study freeze SHA-256")
        self.native_store.write_manifest()
        base = self.native_store.base_dir
        frozen = self.frozen_manifest()
        _write_json_immutable(base / "job_manifest.json", frozen)
        if stage == "outer":
            self.validate_complete_lock_matrix(frozen=frozen)
        stage_jobs = tuple(job for job in self.inventory if job.stage == stage)
        payload = {
            "schema_version": COMPATIBILITY_SCHEMA_VERSION,
            "study_id": self.study_id,
            "study_freeze_sha256": self.freeze_sha256,
            "stage": stage,
            "rows": [dict(row) for row in self.audit_compatibility(stage_jobs)],
        }
        _write_json_immutable(base / f"reuse_compatibility_{stage}.json", payload)
        return base / "job_manifest.json"


def _write_json_immutable(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write canonical JSON once, refusing any attempt to change an existing file."""
    serialized = canonical_json_bytes(dict(payload))
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != serialized:
            raise MatrixError(f"refusing to overwrite incompatible immutable artifact: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.read_bytes() != serialized:
            raise MatrixError(f"refusing to overwrite incompatible immutable artifact: {path}")
    return path


class StudyArtifactView:
    """Resolve native runs and named historical references without copying."""

    def __init__(
        self,
        *,
        manager: NestedOOFFoldManager,
        study_id: str = STUDY_ID,
        artifact_root: str | Path = "artifacts/oof",
        reuse_roots: Mapping[str, str | Path] | None = None,
        training_configs: Mapping[str, TrainingConfig] | None = None,
        study_config: StudyConfig | None = None,
        training_epochs: int | None = None,
        freeze_sha256: str | None = None,
        config_root: str | Path = Path(__file__).resolve().parents[2] / "configs" / "experts",
    ) -> None:
        self.planner = OOFMatrixPlanner(
            manager=manager,
            study_id=study_id,
            artifact_root=artifact_root,
            reuse_roots=reuse_roots,
            training_configs=training_configs,
            study_config=study_config,
            training_epochs=training_epochs,
            freeze_sha256=freeze_sha256,
            config_root=config_root,
        )
        self._by_id = {job.job_id: job for job in self.planner.inventory}
        self._frozen_inventory_checked = False

    def _validate_frozen_inventory(self) -> None:
        """Require this view to match the immutable matrix lock on disk."""
        if self._frozen_inventory_checked:
            return
        base = self.planner.native_store.base_dir
        path = base / "job_manifest.json"
        if not path.is_file():
            raise MatrixError("study job manifest is missing; freeze the matrix before analysis")
        try:
            recorded = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise MatrixError("study job manifest is invalid") from exc
        if canonical_json_bytes(recorded) != path.read_bytes():
            raise MatrixError("study job manifest is not canonical JSON")
        if recorded != self.planner.frozen_manifest():
            raise MatrixError("study job manifest does not match the current frozen plan/config/source")
        self._frozen_inventory_checked = True

    def _validate_stage_audit(self, stage: str) -> None:
        """Require the immutable named-source audit for the requested stage."""
        path = self.planner.native_store.base_dir / f"reuse_compatibility_{stage}.json"
        if not path.is_file():
            raise MatrixError(f"frozen {stage} reuse compatibility table is missing")
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise MatrixError(f"frozen {stage} reuse compatibility table is invalid") from exc
        if canonical_json_bytes(payload) != path.read_bytes():
            raise MatrixError(f"frozen {stage} reuse compatibility table is not canonical JSON")
        if (
            payload.get("schema_version") != COMPATIBILITY_SCHEMA_VERSION
            or payload.get("study_id") != self.planner.study_id
            or payload.get("study_freeze_sha256") != self.planner.freeze_sha256
            or payload.get("stage") != stage
        ):
            raise MatrixError(f"frozen {stage} reuse compatibility identity is invalid")

    def resolve(self, job_id: str) -> ValidatedJobReference:
        """Resolve one job ID to validated immutable run contents."""
        self._validate_frozen_inventory()
        try:
            job = self._by_id[job_id]
        except KeyError as exc:
            raise MatrixError(f"unknown frozen expert job ID: {job_id!r}") from exc
        self._validate_stage_audit(job.stage)
        reference, state, detail = self.planner.resolve_reference(job)
        if reference is None:
            raise MatrixError(f"job {job_id} is {state}: {detail}")
        return reference

    @staticmethod
    def _arrays(reference: ValidatedJobReference) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # The already validated object was discarded by OOFArtifactStore's
        # public read-only result; deserialize the same canonical artifact file.
        from data.nested_oof import OOFPredictionArtifact

        try:
            parsed = OOFPredictionArtifact.from_json(reference.prediction_path.read_text())
        except (OSError, ValueError) as exc:
            raise MatrixError(f"cannot reload validated predictions: {exc}") from exc
        sample_ids = np.asarray([record.sample_index for record in parsed.records], dtype=np.int64)
        labels = np.asarray([record.training_label for record in parsed.records], dtype=np.int64)
        logits = np.asarray([record.logits for record in parsed.records], dtype=np.float64)
        if logits.ndim != 2 or logits.shape[0] != len(sample_ids) or not np.isfinite(logits).all():
            raise MatrixError("validated OOF logits have invalid shape or non-finite values")
        if labels.shape != sample_ids.shape:
            raise MatrixError("validated OOF labels are not aligned to sample IDs")
        return sample_ids, labels, logits

    def load_logits(self, job_id: str) -> AlignedExpertPrediction:
        """Load validated aligned sample IDs, labels, and logits for an inner job."""
        reference = self.resolve(job_id)
        if reference.job.stage != "inner":
            raise MatrixError("load_logits is for inner OOF jobs; use outer loader methods")
        sample_ids, labels, logits = self._arrays(reference)
        job = reference.job
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

    def load_outer_logits(self, job_id: str) -> InferenceExpertPrediction:
        """Load outer inference inputs without exposing their labels."""
        reference = self.resolve(job_id)
        if reference.job.stage != "outer":
            raise MatrixError("load_outer_logits requires an outer job")
        sample_ids, _labels, logits = self._arrays(reference)
        job = reference.job
        return InferenceExpertPrediction(
            job_id=job.job_id,
            stage=job.stage,
            expert_key=job.expert_key,
            expert_name=job.expert_name,
            training_seed=job.training_seed,
            outer_fold_id=job.outer_fold_id,
            sample_ids=sample_ids,
            logits=logits,
            source_experiment_id=reference.source_experiment_id,
            source_run_relative_path=reference.source_run_relative_path,
            training_membership_sha256=job.training_membership_sha256,
            checkpoint_sha256=reference.checkpoint_sha256,
            prediction_sha256=reference.prediction_sha256,
            resolved_config_sha256=reference.resolved_config_sha256,
        )

    def load_outer_labels(self, job_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Load labels for an evaluator after it has completed router locking."""
        reference = self.resolve(job_id)
        if reference.job.stage != "outer":
            raise MatrixError("load_outer_labels requires an outer job")
        sample_ids, labels, _logits = self._arrays(reference)
        return sample_ids, labels


def manager_from_fold_manifest(path: str | Path) -> NestedOOFFoldManager:
    """Reconstruct a fold manager from the immutable OOF manifest."""
    try:
        manifest = FoldManifest.from_json(Path(path).read_text())
        manifest.validate()
        config = manifest.experiment_config
        manager = NestedOOFFoldManager(
            manifest.canonical_training_indices,
            manifest.canonical_training_labels,
            seed=manifest.fold_generation_seed,
            outer_folds=int(config["outer_folds"]),
            inner_folds=int(config["inner_folds"]),
            num_classes=int(config["num_classes"]),
            expert_order=tuple(config["expert_order"]),
            canonical_training_index_sha256=manifest.canonical_training_index_sha256,
        )
    except (OSError, OOFProtocolError, KeyError, TypeError, ValueError) as exc:
        raise MatrixError(f"cannot rebuild fold manager from manifest {path}: {exc}") from exc
    if manager.manifest().to_dict() != manifest.to_dict():
        raise MatrixError("reconstructed manager does not match immutable fold manifest")
    return manager


def cuda_preflight(device: str | None) -> None:
    """Fail clearly before training if explicit CUDA cannot allocate a probe."""
    if device != "cuda":
        return
    try:
        import torch

        if not torch.cuda.is_available():
            raise MatrixError("--device cuda was requested but CUDA is unavailable")
        probe = torch.empty((16, 16), dtype=torch.float32, device="cuda")
        probe.add_(1)
        torch.cuda.synchronize()
        del probe
    except MatrixError:
        raise
    except Exception as exc:  # CUDA driver and allocator failures vary by runtime.
        raise MatrixError(f"--device cuda preflight failed: {exc}") from exc


def _run_file_entries(reference: ValidatedJobReference, study_base: Path) -> list[dict[str, Any]]:
    required = (
        reference.checkpoint_path,
        reference.prediction_path,
        reference.resolved_config_path,
        reference.metadata_path,
    )
    entries: list[dict[str, Any]] = []
    for path in required:
        if path.is_symlink() or not path.is_file():
            raise MatrixError(f"bundle payload must be a regular file: {path}")
        relative_run_path = path.relative_to(reference.run_dir).as_posix()
        _assert_safe_relative_path(relative_run_path, field="run-relative artifact path")
        entries.append(
            {
                "path": f"{STUDY_ID}/{reference.job.run_relative_path}/{relative_run_path}",
                "run_relative_path": relative_run_path,
                "schema": (
                    "nested_oof_prediction.v1" if path == reference.prediction_path
                    else OOF_RUN_SCHEMA_VERSION if path == reference.metadata_path
                    else "resolved_training_config.v1" if path == reference.resolved_config_path
                    else "torch_checkpoint.final.v1"
                ),
                "byte_size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return sorted(entries, key=lambda item: item["path"])


def _tar_info(name: str, size: int, *, mode: int = 0o644) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = mode
    info.type = tarfile.REGTYPE
    info.pax_headers = {}
    return info


def _validate_outer_bundle_lock_gate(
    manifest: Mapping[str, Any],
    *,
    artifact_root: str | Path | None,
    reuse_roots: Mapping[str, str | Path] | None = None,
    study_config: StudyConfig | None = None,
) -> None:
    """Run the full planner gate before reading outer bundle payloads."""
    shard = manifest.get("shard")
    if not isinstance(shard, Mapping) or shard.get("stage") != "outer":
        return
    if artifact_root is None:
        raise MatrixError("outer bundle import requires artifact-root with all 15 validated locks")
    study_config = study_config or StudyConfig()
    if manifest.get("protocol_config_sha256") != study_config.sha256:
        raise MatrixError("outer bundle uses a different frozen StudyConfig")
    root = Path(artifact_root).expanduser().resolve()
    lock_manifest_path = root / STUDY_ID / "job_manifest.json"
    try:
        lock_manifest = json.loads(lock_manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise MatrixError("outer bundle import requires the frozen job_manifest.json") from exc
    if (
        lock_manifest.get("plan_sha256") != manifest.get("plan_sha256")
        or lock_manifest.get("source_commit") != manifest.get("source_commit")
        or lock_manifest.get("study_config_sha256") != manifest.get("study_config_sha256")
        or lock_manifest.get("protocol_config_sha256") != manifest.get("protocol_config_sha256")
        or lock_manifest.get("study_freeze_sha256") != manifest.get("study_freeze_sha256")
    ):
        raise MatrixError("outer bundle identity differs from the local frozen study manifest")
    try:
        fold_manifest_path = root / STUDY_ID / "fold_manifest.json"
        manager = manager_from_fold_manifest(fold_manifest_path)
        planner = OOFMatrixPlanner(
            manager=manager,
            study_id=STUDY_ID,
            artifact_root=root,
            reuse_roots=reuse_roots,
            study_config=study_config,
            config_root=_PLAN_PATH.parents[1] / "configs" / "experts",
        )
        planner.validate_complete_lock_matrix(frozen=lock_manifest)
    except MatrixError:
        raise
    except Exception as exc:  # repository StudyError conveys precise lock failure details.
        raise MatrixError(f"outer bundle lock preflight failed: {exc}") from exc


def export_bundle(
    *,
    view: StudyArtifactView,
    destination: str | Path,
    stage: str,
    shard_index: int = 0,
    shard_count: int = 1,
    compatibility_table: Sequence[Mapping[str, Any]] = (),
) -> Path:
    """Export validated native jobs from one shard as a deterministic tar."""
    planner = view.planner
    if stage == "outer":
        planner.validate_complete_lock_matrix()
    view._validate_frozen_inventory()
    view._validate_stage_audit(stage)
    if not compatibility_table:
        compatibility_table = planner.audit_compatibility(
            tuple(job for job in planner.inventory if job.stage == stage)
        )
    jobs = planner.jobs(stage=stage, shard_index=shard_index, shard_count=shard_count)
    refs: list[ValidatedJobReference] = []
    for job in jobs:
        reference, state, detail = planner.resolve_reference(job)
        if reference is None:
            continue
        if reference.is_historical_reuse:
            continue
        if state != "validated":
            raise MatrixError(f"cannot export invalid job {job.job_id}: {detail}")
        refs.append(reference)
    if not refs:
        raise MatrixError("selected shard has no validated native jobs to export")

    study_base = planner.native_store.base_dir
    fold_manifest = planner.native_store.manifest_path
    if not fold_manifest.is_file():
        raise MatrixError("native OOF fold manifest is missing; run planning before export")
    file_records: list[dict[str, Any]] = []
    payload_sources: dict[str, Path] = {}
    manifest_relative = f"{STUDY_ID}/fold_manifest.json"
    payload_sources[manifest_relative] = fold_manifest
    file_records.append(
        {
            "path": manifest_relative,
            "run_relative_path": "fold_manifest.json",
            "schema": "nested_oof_manifest.v1",
            "byte_size": fold_manifest.stat().st_size,
            "sha256": sha256_file(fold_manifest),
        }
    )
    if planner.freeze_sha256 is not None:
        freeze_path = study_base / "manifests" / "study-freeze.json"
        if not freeze_path.is_file() or freeze_path.is_symlink():
            raise MatrixError("immutable study freeze record is missing; freeze before exporting bundles")
        freeze_relative = f"{STUDY_ID}/manifests/study-freeze.json"
        payload_sources[freeze_relative] = freeze_path
        file_records.append(
            {
                "path": freeze_relative,
                "run_relative_path": "manifests/study-freeze.json",
                "schema": "expert_method.study_freeze.v1",
                "byte_size": freeze_path.stat().st_size,
                "sha256": sha256_file(freeze_path),
            }
        )
    jobs_payload = []
    for reference in sorted(refs, key=lambda ref: ref.job.job_id):
        files = _run_file_entries(reference, study_base)
        file_records.extend(files)
        for item in files:
            local_rel = item["path"][len(f"{STUDY_ID}/"):]
            payload_sources[item["path"]] = study_base / local_rel
        jobs_payload.append(
            {
                "job_id": reference.job.job_id,
                "stage": reference.job.stage,
                "expert": reference.job.expert_key,
                "training_seed": reference.job.training_seed,
                "outer_fold_id": reference.job.outer_fold_id,
                "inner_fold_id": reference.job.inner_fold_id,
                "run_relative_path": reference.job.run_relative_path,
                "training_membership_sha256": reference.job.training_membership_sha256,
                "prediction_membership_sha256": reference.job.prediction_membership_sha256,
                "shard_key": f"{shard_key(reference.job.job_id):016x}",
                "shard_index": shard_index,
                "shard_count": shard_count,
                "checkpoint_sha256": reference.checkpoint_sha256,
                "prediction_sha256": reference.prediction_sha256,
                "resolved_config_sha256": reference.resolved_config_sha256,
                "run_identity_sha256": _sha256_text(
                    _canonical_json(
                        {
                            "job_id": reference.job.job_id,
                            "run_relative_path": reference.job.run_relative_path,
                            "files": files,
                        }
                    )
                ),
                "files": files,
            }
        )
    root_lock = study_base / "job_manifest.json"
    compatibility_path = study_base / f"reuse_compatibility_{stage}.json"
    for path in (root_lock, compatibility_path):
        if path.is_file():
            rel = f"{STUDY_ID}/{path.name}"
            payload_sources[rel] = path
            file_records.append(
                {
                    "path": rel,
                    "run_relative_path": path.name,
                    "schema": "study_lock.v1" if path == root_lock else COMPATIBILITY_SCHEMA_VERSION,
                    "byte_size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    frozen = planner.frozen_manifest()
    payload_manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "study_id": planner.study_id,
        "study_freeze_sha256": frozen["study_freeze_sha256"],
        "study_config_sha256": frozen["study_config_sha256"],
        "protocol_config_sha256": frozen["protocol_config_sha256"],
        "plan_sha256": frozen["plan_sha256"],
        "source_commit": frozen["source_commit"],
        "shard": {
            "stage": stage,
            "index": shard_index,
            "count": shard_count,
            "selected_job_ids": [ref.job.job_id for ref in refs],
        },
        "jobs": jobs_payload,
        "historical_references": [
            dict(row) for row in compatibility_table
            if row.get("status") == "compatible"
            and row.get("target_job_id") in {job.job_id for job in jobs}
            and row.get("target_job_id") not in {ref.job.job_id for ref in refs}
        ],
        "files": sorted(file_records, key=lambda item: item["path"]),
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    manifest_bytes = canonical_json_bytes(payload_manifest)
    try:
        with tarfile.open(destination, mode="w", format=tarfile.GNU_FORMAT) as archive:
            archive.addfile(_tar_info("manifest.json", len(manifest_bytes)), io.BytesIO(manifest_bytes))
            for relative_path in sorted(payload_sources):
                _assert_safe_relative_path(relative_path, field="archive path")
                path = payload_sources[relative_path]
                if path.is_symlink() or not path.is_file():
                    raise MatrixError(f"bundle source is not a regular file: {path}")
                with path.open("rb") as handle:
                    archive.addfile(_tar_info(relative_path, path.stat().st_size), handle)
    except (OSError, tarfile.TarError) as exc:
        raise MatrixError(f"cannot create deterministic bundle {destination}: {exc}") from exc
    return destination


def _read_bundle(
    path: str | Path,
    *,
    artifact_root: str | Path | None = None,
    reuse_roots: Mapping[str, str | Path] | None = None,
    study_config: StudyConfig | None = None,
    payload_directory: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, bytes | Path]]:
    """Validate archive structure, canonical manifest, all paths, sizes and hashes."""
    path = Path(path)
    try:
        archive = tarfile.open(path, mode="r:")
    except (OSError, tarfile.TarError) as exc:
        raise MatrixError(f"cannot read bundle {path}: {exc}") from exc
    with archive:
        first = archive.next()
        if first is None or first.name != "manifest.json":
            raise MatrixError("bundle first archive member must be manifest.json")
        _assert_safe_relative_path(first.name, field="archive member")
        if not first.isfile() or first.issym() or first.islnk():
            raise MatrixError("bundle manifest member must be a regular file")
        manifest_stream = archive.extractfile(first)
        if manifest_stream is None:
            raise MatrixError("cannot read bundle manifest member")
        manifest_bytes = manifest_stream.read()
        if len(manifest_bytes) != first.size:
            raise MatrixError("bundle manifest byte count mismatch")
        try:
            manifest = json.loads(manifest_bytes)
        except json.JSONDecodeError as exc:
            raise MatrixError("bundle manifest is invalid JSON") from exc
        if not isinstance(manifest, Mapping) or manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
            raise MatrixError("unsupported bundle manifest schema")
        if canonical_json_bytes(manifest) != manifest_bytes:
            raise MatrixError("bundle manifest is not canonical JSON")
        # This check deliberately precedes extracting any payload member. Outer
        # predictions and their embedded labels stay unread until all locks pass.
        _validate_outer_bundle_lock_gate(
            manifest, artifact_root=artifact_root, reuse_roots=reuse_roots,
            study_config=study_config,
        )
        seen: set[str] = set()
        contents: dict[str, bytes | Path] = {}
        actual_hashes: dict[str, tuple[int, str]] = {}
        seen.add(first.name)
        while True:
            member = archive.next()
            if member is None:
                break
            _assert_safe_relative_path(member.name, field="archive member")
            if member.name in seen:
                raise MatrixError(f"duplicate archive member: {member.name}")
            seen.add(member.name)
            if not member.isfile() or member.issym() or member.islnk():
                raise MatrixError(f"bundle member is not a regular file: {member.name}")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise MatrixError(f"cannot read archive member: {member.name}")
            if payload_directory is None:
                data = extracted.read()
                if len(data) != member.size:
                    raise MatrixError(f"archive member byte count mismatch: {member.name}")
                contents[member.name] = data
                actual_hashes[member.name] = (len(data), hashlib.sha256(data).hexdigest())
            else:
                staging_root = Path(payload_directory).resolve()
                relative = _assert_safe_relative_path(member.name, field="staged payload path")
                staged_path = staging_root.joinpath(*relative.parts)
                staged_path.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                written = 0
                try:
                    with staged_path.open("xb") as output:
                        while True:
                            chunk = extracted.read(1024 * 1024)
                            if not chunk:
                                break
                            output.write(chunk)
                            digest.update(chunk)
                            written += len(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                except OSError as exc:
                    raise MatrixError(f"cannot stage bundle payload {member.name}: {exc}") from exc
                if written != member.size:
                    raise MatrixError(f"archive member byte count mismatch: {member.name}")
                contents[member.name] = staged_path
                actual_hashes[member.name] = (written, digest.hexdigest())
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise MatrixError("bundle manifest files must be a list")
    expected_names: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise MatrixError("bundle manifest file entry must be an object")
        name = entry.get("path")
        _assert_safe_relative_path(name, field="manifest payload path")
        if name in expected_names:
            raise MatrixError(f"duplicate manifest path: {name}")
        expected_names.add(name)
        staged = contents.get(name)
        if staged is None:
            raise MatrixError(f"bundle payload is missing: {name}")
        actual_size, actual_hash = actual_hashes[name]
        if entry.get("byte_size") != actual_size:
            raise MatrixError(f"bundle payload size mismatch: {name}")
        if entry.get("sha256") != actual_hash:
            raise MatrixError(f"bundle payload hash mismatch: {name}")
    if set(contents) != expected_names:
        unlisted = sorted(set(contents) - expected_names)
        raise MatrixError(f"bundle contains unlisted payload members: {unlisted}")
    _validate_bundle_job_identities(manifest)
    _validate_bundle_study_metadata(manifest, contents)
    return dict(manifest), contents


def _validate_bundle_job_identities(manifest: Mapping[str, Any]) -> None:
    if manifest.get("study_id") != STUDY_ID:
        raise MatrixError("bundle study ID does not match the frozen study")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        raise MatrixError("bundle jobs must be a list")
    shard = manifest.get("shard")
    if not isinstance(shard, Mapping):
        raise MatrixError("bundle shard identity is missing")
    stage = shard.get("stage")
    shard_index = shard.get("index")
    shard_count = shard.get("count")
    if stage not in {"inner", "outer"}:
        raise MatrixError("bundle stage is invalid")
    if (
        isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or isinstance(shard_count, bool)
        or not isinstance(shard_count, int)
        or shard_count < 1
        or shard_index < 0
        or shard_index >= shard_count
    ):
        raise MatrixError("bundle shard selection is malformed")
    seen_ids: set[str] = set()
    files_by_path = {
        entry.get("path"): entry
        for entry in manifest.get("files", [])
        if isinstance(entry, Mapping)
    }
    owned_payloads = {f"{STUDY_ID}/fold_manifest.json", f"{STUDY_ID}/job_manifest.json"}
    if manifest.get("study_freeze_sha256") is not None:
        owned_payloads.add(f"{STUDY_ID}/manifests/study-freeze.json")
    owned_payloads.add(f"{STUDY_ID}/reuse_compatibility_{stage}.json")
    for job in jobs:
        if not isinstance(job, Mapping):
            raise MatrixError("bundle job entry must be an object")
        job_id = job.get("job_id")
        if not isinstance(job_id, str) or not job_id.startswith(f"{STUDY_ID}/") or job_id in seen_ids:
            raise MatrixError(f"invalid or duplicate bundle job ID: {job_id!r}")
        seen_ids.add(job_id)
        expert = job.get("expert")
        seed = job.get("training_seed")
        outer = job.get("outer_fold_id")
        inner = job.get("inner_fold_id")
        if expert not in DEFAULT_EXPERT_KEYS:
            raise MatrixError(f"bundle job has an unknown expert: {expert!r}")
        if seed not in DEFAULT_SEEDS or isinstance(seed, bool):
            raise MatrixError(f"bundle job has an unknown seed: {seed!r}")
        if isinstance(outer, bool) or not isinstance(outer, int) or outer not in range(5):
            raise MatrixError(f"bundle job has an invalid outer fold: {outer!r}")
        if job.get("stage") != stage:
            raise MatrixError("bundle job stage disagrees with bundle shard stage")
        if stage == "inner":
            if isinstance(inner, bool) or not isinstance(inner, int) or inner not in range(4):
                raise MatrixError(f"inner bundle job has an invalid fold ID: {inner!r}")
            fold_component = f"inner_{inner}"
            job_id_tail = fold_component
        else:
            if inner is not None:
                raise MatrixError("outer bundle job must use a null inner-fold marker")
            fold_component = "outer_eval"
            job_id_tail = fold_component
        expected_job_id = (
            f"{STUDY_ID}/{stage}/{expert}/seed_{seed}/outer_{outer}/{job_id_tail}"
        )
        if job_id != expected_job_id:
            raise MatrixError(f"bundle job ID fields do not match its stable identity: {job_id}")
        if shard_for_job(job_id, shard_count) != shard_index:
            raise MatrixError(f"bundle job is outside the declared shard: {job_id}")
        if job.get("shard_index") != shard_index or job.get("shard_count") != shard_count:
            raise MatrixError("bundle job shard fields disagree with manifest")
        if job.get("shard_key") != f"{shard_key(job_id):016x}":
            raise MatrixError("bundle job shard key does not match SHA-256(job_id)")
        run_relative = _assert_safe_relative_path(
            job.get("run_relative_path"), field="job run path"
        ).as_posix()
        expert_name = EXPERT_NAMES[expert]
        expected_run_path = f"expert_{expert_name}/seed_{seed}/outer_{outer}/{fold_component}"
        if run_relative != expected_run_path:
            raise MatrixError(f"bundle run path does not match job identity: {run_relative}")
        files = job.get("files")
        expected_run_files = {
            f"checkpoints/{expert_name}_seed{seed}_final.pt",
            "predictions.json",
            "resolved_config.json",
            "run_metadata.json",
        }
        if not isinstance(files, list) or len(files) != len(expected_run_files):
            raise MatrixError(f"bundle job has an incomplete artifact file set: {job_id}")
        actual_run_files: set[str] = set()
        for entry in files:
            if not isinstance(entry, Mapping):
                raise MatrixError(f"bundle file list is malformed for {job_id}")
            relative = _assert_safe_relative_path(entry.get("run_relative_path"), field="run-relative file path")
            if relative.as_posix() in actual_run_files:
                raise MatrixError(f"bundle job contains duplicate run-relative paths: {job_id}")
            actual_run_files.add(relative.as_posix())
            archive_path = f"{STUDY_ID}/{run_relative}/{relative.as_posix()}"
            if entry.get("path") != archive_path or archive_path not in files_by_path:
                raise MatrixError(f"bundle job file identity does not align: {archive_path}")
            owned_payloads.add(archive_path)
        if actual_run_files != expected_run_files:
            raise MatrixError(f"bundle job contains unowned or missing run files: {job_id}")
        identity = _sha256_text(
            _canonical_json(
                {
                    "job_id": job_id,
                    "run_relative_path": run_relative,
                    "files": files,
                }
            )
        )
        if job.get("run_identity_sha256") != identity:
            raise MatrixError(f"bundle run identity hash does not match: {job_id}")
        if job.get("training_membership_sha256") != _valid_digest(
            job.get("training_membership_sha256"), name="training membership hash"
        ):
            raise MatrixError(f"bundle training membership hash is malformed: {job_id}")
        if job.get("prediction_membership_sha256") != _valid_digest(
            job.get("prediction_membership_sha256"), name="prediction membership hash"
        ):
            raise MatrixError(f"bundle prediction membership hash is malformed: {job_id}")
        for name in ("checkpoint_sha256", "prediction_sha256", "resolved_config_sha256"):
            _valid_digest(job.get(name), name=name)
        expected_selected = shard.get("selected_job_ids")
        if not isinstance(expected_selected, list) or job_id not in expected_selected:
            raise MatrixError(f"bundle job is missing from selected_job_ids: {job_id}")

    selected_ids = shard.get("selected_job_ids")
    if not isinstance(selected_ids, list) or len(set(selected_ids)) != len(selected_ids):
        raise MatrixError("bundle selected_job_ids is malformed or duplicated")
    if set(selected_ids) != seen_ids:
        raise MatrixError("bundle selected_job_ids does not equal its native job records")
    actual_payloads = set(files_by_path)
    if actual_payloads != owned_payloads:
        extras = sorted(actual_payloads - owned_payloads)
        missing = sorted(owned_payloads - actual_payloads)
        raise MatrixError(f"bundle payload ownership mismatch (extra={extras}, missing={missing})")


def _valid_digest(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise MatrixError(f"{name} must be a 64-character SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise MatrixError(f"{name} is not hexadecimal") from exc
    return value.lower()


def _validate_bundle_study_metadata(
    manifest: Mapping[str, Any], payloads: Mapping[str, bytes | Path]
) -> None:
    """Tie the bundle's selected jobs to its frozen inventory and stage audit."""
    study_prefix = f"{STUDY_ID}/"
    try:
        lock_bytes = _payload_bytes(payloads[f"{study_prefix}job_manifest.json"])
        lock = json.loads(lock_bytes)
        compatibility_bytes = _payload_bytes(
            payloads[f"{study_prefix}reuse_compatibility_{manifest['shard']['stage']}.json"]
        )
        compatibility = json.loads(compatibility_bytes)
    except (KeyError, json.JSONDecodeError) as exc:
        raise MatrixError("bundle is missing a valid frozen study lock or stage reuse audit") from exc
    if canonical_json_bytes(lock) != lock_bytes or canonical_json_bytes(compatibility) != compatibility_bytes:
        raise MatrixError("bundle study metadata is not canonical JSON")
    if (
        lock.get("schema_version") != MATRIX_SCHEMA_VERSION
        or lock.get("study_id") != STUDY_ID
        or lock.get("study_freeze_sha256") != manifest.get("study_freeze_sha256")
        or lock.get("study_config_sha256") != manifest.get("study_config_sha256")
        or lock.get("protocol_config_sha256") != manifest.get("protocol_config_sha256")
        or lock.get("plan_sha256") != manifest.get("plan_sha256")
        or lock.get("source_commit") != manifest.get("source_commit")
    ):
        raise MatrixError("bundle metadata disagrees with its frozen study/config/source identity")
    if (
        compatibility.get("schema_version") != COMPATIBILITY_SCHEMA_VERSION
        or compatibility.get("study_id") != STUDY_ID
        or compatibility.get("study_freeze_sha256") != manifest.get("study_freeze_sha256")
        or compatibility.get("stage") != manifest["shard"]["stage"]
    ):
        raise MatrixError("bundle compatibility audit has an incompatible identity")
    if manifest.get("study_freeze_sha256") is not None:
        freeze_rel = f"{study_prefix}manifests/study-freeze.json"
        try:
            freeze_bytes = _payload_bytes(payloads[freeze_rel])
            freeze = json.loads(freeze_bytes)
        except (KeyError, json.JSONDecodeError) as exc:
            raise MatrixError("bundle is missing its immutable study freeze record") from exc
        freeze_hash = freeze.get("freeze_sha256") if isinstance(freeze, Mapping) else None
        unhashed = dict(freeze) if isinstance(freeze, Mapping) else {}
        unhashed.pop("freeze_sha256", None)
        if (
            canonical_json_bytes(freeze) != freeze_bytes
            or freeze_hash != manifest.get("study_freeze_sha256")
            or freeze_hash != lock.get("study_freeze_sha256")
            or freeze_hash != _sha256_text(_canonical_json(unhashed))
            or freeze.get("source_commit") != manifest.get("source_commit")
            or freeze.get("plan_sha256") != manifest.get("plan_sha256")
            or freeze.get("fold_manifest_sha256") != lock.get("fold_manifest_sha256")
        ):
            raise MatrixError("bundle freeze record disagrees with its job/config/source identity")
    inventory = lock.get("inventory")
    if not isinstance(inventory, list) or len(inventory) != 300 or lock.get("inventory_count") != 300:
        raise MatrixError("bundle study lock has an incomplete job inventory")
    inventory_by_id: dict[str, Mapping[str, Any]] = {}
    for item in inventory:
        if not isinstance(item, Mapping) or not isinstance(item.get("job_id"), str):
            raise MatrixError("bundle frozen inventory contains a malformed job")
        job_id = item["job_id"]
        if job_id in inventory_by_id:
            raise MatrixError(f"bundle frozen inventory duplicates job ID {job_id}")
        inventory_by_id[job_id] = item
    if sum(item.get("stage") == "inner" for item in inventory) != 240 or sum(
        item.get("stage") == "outer" for item in inventory
    ) != 60:
        raise MatrixError("bundle frozen inventory does not contain 240 inner and 60 outer jobs")
    for job in manifest["jobs"]:
        frozen = inventory_by_id.get(job["job_id"])
        if frozen is None:
            raise MatrixError(f"bundle job is absent from frozen inventory: {job['job_id']}")
        for field in (
            "stage", "expert_key", "expert_name", "training_seed", "outer_fold_id",
            "inner_fold_id", "training_membership_sha256", "prediction_membership_sha256",
            "run_relative_path",
        ):
            expected = job.get("expert") if field == "expert_key" else (
                EXPERT_NAMES.get(job.get("expert")) if field == "expert_name" else
                job.get("run_relative_path") if field == "run_relative_path" else job.get(field)
            )
            if frozen.get(field) != expected:
                raise MatrixError(f"bundle job {field} disagrees with frozen inventory")
    compatibility_rows = compatibility.get("rows")
    if not isinstance(compatibility_rows, list):
        raise MatrixError("bundle compatibility rows are malformed")
    compatible_ids = {
        row.get("target_job_id")
        for row in compatibility_rows
        if isinstance(row, Mapping) and row.get("status") == "compatible"
    }
    references = manifest.get("historical_references")
    if not isinstance(references, list):
        raise MatrixError("bundle historical reference list is malformed")
    selected_ids = {job["job_id"] for job in manifest["jobs"]}
    for row in references:
        if not isinstance(row, Mapping):
            raise MatrixError("bundle historical reference row is malformed")
        job_id = row.get("target_job_id")
        if job_id not in compatible_ids or job_id in selected_ids:
            raise MatrixError("bundle historical reference is absent, unvalidated, or duplicated as native")
        if row.get("status") != "compatible" or row.get("source_root_name") is None:
            raise MatrixError("bundle historical reference lacks a named compatible reuse root")
        _assert_safe_relative_path(row.get("source_run_relative_path"), field="historical run path")


def _payload_bytes(payload: bytes | Path) -> bytes:
    """Read a small study metadata file from in-memory or staged bundle content."""
    if isinstance(payload, bytes):
        return payload
    try:
        return payload.read_bytes()
    except OSError as exc:
        raise MatrixError(f"cannot read staged study metadata {payload}") from exc


def _bundle_jobs_compatible(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Check whether bundle manifests describe the same frozen study.

    Stage and shard selection describe how a bundle was exported. They do not
    change the portable identity of a job, which is its stable job ID and the
    immutable run-relative files and hashes validated in each manifest.
    """
    return (
        left.get("study_id") == right.get("study_id")
        and left.get("study_freeze_sha256") == right.get("study_freeze_sha256")
        and left.get("study_config_sha256") == right.get("study_config_sha256")
        and left.get("protocol_config_sha256") == right.get("protocol_config_sha256")
        and left.get("plan_sha256") == right.get("plan_sha256")
        and left.get("source_commit") == right.get("source_commit")
    )


def import_bundles(
    paths: Sequence[str | Path],
    *,
    artifact_root: str | Path,
    reuse_roots: Mapping[str, str | Path] | None = None,
    study_config: StudyConfig | None = None,
) -> tuple[str, ...]:
    """Validate one or more bundles, merge idempotently, then import native bytes."""
    if not paths:
        raise MatrixError("at least one bundle path is required")
    root = Path(artifact_root).expanduser().resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ridge_sinkhorn_bundle_", dir=root.parent) as temporary:
        staging_root = Path(temporary)
        parsed = [
            _read_bundle(
                path,
                artifact_root=root,
                reuse_roots=reuse_roots,
                study_config=study_config,
                payload_directory=staging_root / f"bundle_{index:04d}",
            )
            for index, path in enumerate(paths)
        ]
        first_manifest = parsed[0][0]
        merged_jobs: dict[str, Mapping[str, Any]] = {}
        merged_payloads: dict[str, Path] = {}
        merged_file_records: dict[str, Mapping[str, Any]] = {}
        historical: dict[str, Mapping[str, Any]] = {}
        for manifest, payloads in parsed:
            if not _bundle_jobs_compatible(first_manifest, manifest):
                raise MatrixError("bundles disagree on study/config/plan/source identity")
            for job in manifest["jobs"]:
                job_id = job["job_id"]
                previous = merged_jobs.get(job_id)
                if previous is not None:
                    # Shard assignment is export provenance and can differ
                    # across otherwise identical portable bundles.
                    previous_identity = {
                        key: value for key, value in previous.items()
                        if key not in {"shard_index", "shard_count"}
                    }
                    current_identity = {
                        key: value for key, value in job.items()
                        if key not in {"shard_index", "shard_count"}
                    }
                    if _canonical_json(previous_identity) != _canonical_json(current_identity):
                        raise MatrixError(f"conflicting bundle identities for job {job_id}")
                merged_jobs[job_id] = job
            for entry in manifest["files"]:
                name = entry["path"]
                existing_record = merged_file_records.get(name)
                if existing_record is not None and _canonical_json(dict(existing_record)) != _canonical_json(dict(entry)):
                    raise MatrixError(f"bundles contain conflicting file identity for {name}")
                merged_file_records[name] = entry
                payload_path = payloads[name]
                if not isinstance(payload_path, Path):
                    raise MatrixError("streamed bundle staging did not produce a file path")
                prior_path = merged_payloads.get(name)
                if prior_path is not None and not _files_equal(prior_path, payload_path):
                    raise MatrixError(f"bundles contain differing payload bytes for {name}")
                merged_payloads[name] = payload_path
            for row in manifest.get("historical_references", []):
                if not isinstance(row, Mapping):
                    raise MatrixError("bundle historical reference is malformed")
                job_id = row.get("target_job_id")
                prior_row = historical.get(job_id)
                if prior_row is not None and _canonical_json(dict(prior_row)) != _canonical_json(dict(row)):
                    raise MatrixError(f"conflicting historical references for job {job_id}")
                historical[job_id] = row

        target_records: dict[Path, Mapping[str, Any]] = {}
        for relative, entry in merged_file_records.items():
            rel = _assert_safe_relative_path(relative, field="import destination path")
            target = root.joinpath(*rel.parts)
            resolved = target.resolve(strict=False)
            if root != resolved and root not in resolved.parents:
                raise MatrixError(f"bundle destination escapes artifact root: {relative}")
            cursor = target.parent
            while cursor != root and root in cursor.parents:
                if cursor.exists() and cursor.is_symlink():
                    raise MatrixError(f"bundle destination traverses a symlink: {cursor}")
                cursor = cursor.parent
            if target.exists():
                if target.is_symlink() or not target.is_file():
                    raise MatrixError(f"existing import target is not a regular file: {target}")
                if (
                    target.stat().st_size != entry["byte_size"]
                    or sha256_file(target) != entry["sha256"]
                ):
                    raise MatrixError(f"refusing to overwrite different existing bytes: {target}")
            target_records[target] = entry

        # All bundles, payload hashes, identity conflicts, and destination
        # conflicts have passed before the first final artifact is installed.
        for target, entry in target_records.items():
            if target.exists():
                continue
            staged_source = merged_payloads[entry["path"]]
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary_target: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb", prefix=f".{target.name}.import-", dir=target.parent, delete=False
                ) as handle:
                    temporary_target = Path(handle.name)
                    with staged_source.open("rb") as source:
                        shutil.copyfileobj(source, handle, length=1024 * 1024)
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    os.link(temporary_target, target)
                except FileExistsError:
                    if target.is_symlink() or target.stat().st_size != entry["byte_size"] or sha256_file(target) != entry["sha256"]:
                        raise MatrixError(f"concurrent import produced conflicting bytes: {target}")
            except OSError as exc:
                raise MatrixError(f"cannot install imported artifact {target}: {exc}") from exc
            finally:
                if temporary_target is not None and temporary_target.exists():
                    temporary_target.unlink()
        return tuple(sorted(merged_jobs))


def _files_equal(left: Path, right: Path) -> bool:
    """Compare staged payload files without loading them into memory."""
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_handle, right.open("rb") as right_handle:
        while True:
            left_chunk = left_handle.read(1024 * 1024)
            right_chunk = right_handle.read(1024 * 1024)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def import_bundle(
    path: str | Path,
    *,
    artifact_root: str | Path,
    reuse_roots: Mapping[str, str | Path] | None = None,
    study_config: StudyConfig | None = None,
) -> tuple[str, ...]:
    """Import one validated tar bundle."""
    return import_bundles(
        [path], artifact_root=artifact_root, reuse_roots=reuse_roots,
        study_config=study_config,
    )


def merge_bundles(
    paths: Sequence[str | Path],
    *,
    artifact_root: str | Path,
    reuse_roots: Mapping[str, str | Path] | None = None,
    study_config: StudyConfig | None = None,
) -> tuple[str, ...]:
    """Merge validated bundle inventories idempotently into the native root."""
    return import_bundles(
        paths, artifact_root=artifact_root, reuse_roots=reuse_roots,
        study_config=study_config,
    )
