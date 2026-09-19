"""Task 3C batch orchestration, alignment, and descriptive diagnostics.

This module is deliberately a thin layer over the existing single-run
``OOFPipeline``. It freezes the requested 16-job matrix, validates the CE
pilot read-only, skips completed jobs, and assembles the four experts only
after every required inner OOF artifact has passed the provenance checks.

No router, sample-weight optimizer, Ridge model, Sinkhorn solver, or test-set
path is exposed here. ``ExpertDiagnostics`` is used only for descriptive and
explicitly label-dependent diagnostic reports.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from data.nested_oof import NestedOOFFoldManager, OOFProtocolError
from scripts.config import ConfigError, TrainingConfig
from scripts.expert_diagnostics import ExpertDiagnostics
from scripts.oof_pipeline import (
    OOFArtifactError,
    OOFArtifactStore,
    OOFCompletedRun,
    OOFPipeline,
    OOFRunSpec,
)


TASK3C_SCHEMA_VERSION = "task3c_oof_batch.v1"
ALIGNED_OOF_SCHEMA_VERSION = "task3c_aligned_oof.v1"
DIAGNOSTIC_REPORT_SCHEMA_VERSION = "task3c_diagnostics.v1"

TASK3C_EXPERIMENT_ID = "task3c_oof"
TASK3C_TRAINING_SEED = 78
TASK3C_FOLD_SEED = 42
TASK3C_OUTER_FOLD = 0
TASK3C_INNER_FOLDS = (0, 1, 2, 3)
TASK3C_EPOCHS = 200
TASK3C_EXPERT_KEYS = (
    "ce",
    "logit_adjusted",
    "balanced_softmax",
    "mixup",
)
TASK3C_EXPERT_NAMES = ("CE", "LAL", "BalancedSoftmax", "Mixup")
TASK3C_CONFIG_FILES = {
    "ce": "ce.yaml",
    "logit_adjusted": "lal.yaml",
    "balanced_softmax": "balanced_softmax.yaml",
    "mixup": "mixup.yaml",
}
TASK3C_PILOT_DEFAULT = "artifacts/oof/task3b_pilot_ce_s78_o0_i0"


class Task3CError(OOFArtifactError):
    """Raised when Task 3C planning, validation, or assembly is unsafe."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise Task3CError(f"value is not JSON serializable: {value!r}") from exc


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise Task3CError(f"cannot hash file: {path}") from exc
    return digest.hexdigest()


def _write_json_once(path: Path, payload: Mapping[str, Any]) -> Path:
    serialized = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    if path.exists():
        try:
            existing = path.read_text()
        except OSError as exc:
            raise Task3CError(f"cannot read existing artifact: {path}") from exc
        if existing != serialized:
            raise Task3CError(f"refusing to overwrite an incompatible artifact: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized)
    return path


@dataclass(frozen=True)
class Task3CJob:
    """One frozen expert/inner-fold job in the Task 3C matrix."""

    expert_key: str
    expert_name: str
    inner_fold_id: int
    config_path: Path
    spec: OOFRunSpec

    @property
    def job_id(self) -> str:
        return f"{self.expert_key}/inner_{self.inner_fold_id}"

    @property
    def is_pilot(self) -> bool:
        return self.expert_key == "ce" and self.inner_fold_id == 0


@dataclass(frozen=True)
class Task3CJobStatus:
    """Read-only status returned by a batch scan."""

    job_id: str
    state: str
    artifact_dir: Path
    detail: str = ""


@dataclass(frozen=True)
class Task3CProtocol:
    """Frozen experimental values; hardware and paths are not protocol values."""

    training_seed: int = TASK3C_TRAINING_SEED
    fold_seed: int = TASK3C_FOLD_SEED
    outer_fold_id: int = TASK3C_OUTER_FOLD
    inner_fold_ids: tuple[int, ...] = TASK3C_INNER_FOLDS
    epochs: int = TASK3C_EPOCHS
    expert_keys: tuple[str, ...] = TASK3C_EXPERT_KEYS
    expert_names: tuple[str, ...] = TASK3C_EXPERT_NAMES

    def __post_init__(self) -> None:
        if self.training_seed != TASK3C_TRAINING_SEED:
            raise Task3CError("Task 3C training seed is frozen at 78")
        if self.fold_seed != TASK3C_FOLD_SEED:
            raise Task3CError("Task 3C fold-generation seed is frozen at 42")
        if self.outer_fold_id != TASK3C_OUTER_FOLD:
            raise Task3CError("Task 3C outer fold is frozen at 0")
        if tuple(self.inner_fold_ids) != TASK3C_INNER_FOLDS:
            raise Task3CError("Task 3C inner folds are frozen at 0, 1, 2, and 3")
        if self.epochs != TASK3C_EPOCHS:
            raise Task3CError("Task 3C training duration is frozen at 200 epochs")
        if tuple(self.expert_keys) != TASK3C_EXPERT_KEYS:
            raise Task3CError("Task 3C expert ordering/configuration is frozen")
        if tuple(self.expert_names) != TASK3C_EXPERT_NAMES:
            raise Task3CError("Task 3C expert names/order are frozen")


class Task3CBatchPlan:
    """Build and validate the exact 16-job Task 3C matrix."""

    def __init__(
        self,
        *,
        data_root: str | Path = "./data",
        config_root: str | Path = "configs",
        artifact_root: str | Path = "artifacts/oof",
        pilot_root: str | Path = TASK3C_PILOT_DEFAULT,
        experiment_id: str = TASK3C_EXPERIMENT_ID,
        device: str | None = None,
        manager: NestedOOFFoldManager | None = None,
        protocol: Task3CProtocol | None = None,
        enforce_canonical_population: bool = True,
    ) -> None:
        self.protocol = protocol or Task3CProtocol()
        if experiment_id != TASK3C_EXPERIMENT_ID:
            raise Task3CError(
                f"Task 3C experiment ID is frozen at {TASK3C_EXPERIMENT_ID!r}"
            )
        if device not in (None, "auto", "cpu", "cuda"):
            raise Task3CError(f"unsupported device override: {device!r}")

        self.data_root = Path(data_root)
        self.config_root = Path(config_root)
        self.artifact_root = Path(artifact_root)
        self.pilot_root = Path(pilot_root)
        self.experiment_id = experiment_id
        self.device = device
        self.manager = manager or NestedOOFFoldManager.from_canonical_training_data(
            self.data_root,
            seed=self.protocol.fold_seed,
            outer_folds=5,
            inner_folds=4,
            expert_order=self.protocol.expert_names,
        )
        if self.manager.fold_generation_seed != self.protocol.fold_seed:
            raise Task3CError("fold manager seed does not match frozen Task 3C seed")
        if self.manager.expert_order != self.protocol.expert_names:
            raise Task3CError("fold manager expert ordering does not match Task 3C")
        if self.manager.outer_fold_count != 5 or self.manager.inner_fold_count != 4:
            raise Task3CError("Task 3C requires five outer and four inner folds")
        if enforce_canonical_population and (
            len(self.manager.canonical_indices) != 10847
            or self.manager.num_classes != 100
        ):
            raise Task3CError(
                "Task 3C requires the canonical 10,847-sample CIFAR-100-LT population"
            )
        if enforce_canonical_population:
            outer = self.manager.outer_fold(self.protocol.outer_fold_id)
            if len(outer.evaluation_indices) != 2170:
                raise Task3CError(
                    "Task 3C outer fold 0 must reserve exactly 2,170 evaluation images"
                )

        self.jobs = self._build_jobs()
        if len(self.jobs) != 16:
            raise Task3CError(f"Task 3C matrix must contain 16 jobs, got {len(self.jobs)}")

    def _build_jobs(self) -> tuple[Task3CJob, ...]:
        jobs: list[Task3CJob] = []
        for expert_key, expert_name in zip(
            self.protocol.expert_keys, self.protocol.expert_names
        ):
            config_path = (self.config_root / TASK3C_CONFIG_FILES[expert_key]).resolve()
            self._load_and_validate_config(expert_key, config_path)
            for inner_fold_id in self.protocol.inner_fold_ids:
                jobs.append(
                    Task3CJob(
                        expert_key=expert_key,
                        expert_name=expert_name,
                        inner_fold_id=inner_fold_id,
                        config_path=config_path,
                        spec=OOFRunSpec(
                            experiment_id=self.experiment_id,
                            expert=expert_key,
                            training_seed=self.protocol.training_seed,
                            outer_fold_id=self.protocol.outer_fold_id,
                            inner_fold_id=inner_fold_id,
                            device=self.device,
                            epochs=self.protocol.epochs,
                        ),
                    )
                )
        return tuple(jobs)

    @staticmethod
    def _load_and_validate_config(expert_key: str, path: Path) -> TrainingConfig:
        try:
            config = TrainingConfig.from_file(path)
        except (ConfigError, OSError) as exc:
            raise Task3CError(f"cannot load Task 3C config {path}: {exc}") from exc
        if config.expert != expert_key:
            raise Task3CError(
                f"config {path} declares expert {config.expert!r}, expected {expert_key!r}"
            )
        if config.model.arch != "resnet32" or config.model.num_classes != 100:
            raise Task3CError(f"config {path} does not use the frozen ResNet-32/100 model")
        if config.optimiser.name != "sgd":
            raise Task3CError(f"config {path} does not use the frozen SGD recipe")
        if (
            config.optimiser.lr != 0.1
            or config.optimiser.momentum != 0.9
            or config.optimiser.weight_decay != 2.0e-4
            or config.optimiser.nesterov
        ):
            raise Task3CError(f"config {path} changes the frozen optimizer recipe")
        if (
            config.schedule.epochs != TASK3C_EPOCHS
            or config.schedule.warmup_epochs != 5
            or config.schedule.decay_epochs != (160, 180)
            or config.schedule.decay_factors != (0.01, 0.0001)
        ):
            raise Task3CError(f"config {path} changes the frozen 200-epoch schedule")
        if config.data.imbalance_ratio != 100.0 or config.data.batch_size != 128:
            raise Task3CError(f"config {path} changes the frozen CIFAR-LT data recipe")
        return config

    def config_for(self, job: Task3CJob) -> TrainingConfig:
        """Load a job config and point only its data root at the requested source."""
        config = self._load_and_validate_config(job.expert_key, job.config_path)
        return dataclasses.replace(
            config,
            data=dataclasses.replace(config.data, root=str(self.data_root)),
        )

    def context_for(self, job: Task3CJob):
        return job.spec.resolve(self.manager)

    def batch_manifest(self) -> dict[str, Any]:
        manifest_json = self.manager.manifest().to_json()
        return {
            "schema_version": TASK3C_SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "dataset": "CIFAR-100-LT",
            "imbalance_ratio": 100.0,
            "canonical_population_size": len(self.manager.canonical_indices),
            "training_seed": self.protocol.training_seed,
            "fold_generation_seed": self.protocol.fold_seed,
            "outer_fold_id": self.protocol.outer_fold_id,
            "outer_evaluation_size": len(
                self.manager.outer_fold(self.protocol.outer_fold_id).evaluation_indices
            ),
            "inner_fold_ids": list(self.protocol.inner_fold_ids),
            "epochs": self.protocol.epochs,
            "expert_order": list(self.protocol.expert_names),
            "fold_manifest_sha256": _sha256_text(manifest_json),
            "pilot_path": str(self.pilot_root),
            "jobs": [
                {
                    "job_id": job.job_id,
                    "expert_key": job.expert_key,
                    "expert_name": job.expert_name,
                    "inner_fold_id": job.inner_fold_id,
                    "config_path": str(job.config_path),
                    "pilot_reused_read_only": job.is_pilot,
                }
                for job in self.jobs
            ],
            "router_development": {
                "fit_inner_fold_ids": [1, 2, 3],
                "selection_inner_fold_ids": [0],
                "outer_evaluation_excluded": True,
                "router_fitting_implemented": False,
            },
        }

    def write_batch_manifest(self) -> Path:
        path = self.artifact_root / self.experiment_id / "batch_manifest.json"
        return _write_json_once(path, self.batch_manifest())


class Task3CBatchRunner:
    """Incrementally execute and validate the frozen Task 3C jobs."""

    def __init__(
        self,
        plan: Task3CBatchPlan,
        *,
        trainer_builder: Callable[..., Any] | None = None,
    ) -> None:
        self.plan = plan
        self.store = OOFArtifactStore(
            root=plan.artifact_root,
            manager=plan.manager,
            experiment_id=plan.experiment_id,
        )
        self.pipeline = OOFPipeline(
            manager=plan.manager,
            store=self.store,
            trainer_builder=trainer_builder or self._default_trainer_builder(),
        )

    @staticmethod
    def _default_trainer_builder():
        from scripts.trainers import build_trainer

        return build_trainer

    def artifact_dir(self, job: Task3CJob) -> Path:
        if job.is_pilot:
            return self.plan.pilot_root
        return self.store.run_dir(self.plan.context_for(job))

    def _pilot_store_and_context(self) -> tuple[OOFArtifactStore, Any] | None:
        pilot_root = self.plan.pilot_root
        if not pilot_root.exists():
            return None
        try:
            pilot_store = OOFArtifactStore(
                root=pilot_root.parent,
                manager=self.plan.manager,
                experiment_id=pilot_root.name,
            )
            pilot_spec = OOFRunSpec(
                experiment_id=pilot_root.name,
                expert="ce",
                training_seed=self.plan.protocol.training_seed,
                outer_fold_id=self.plan.protocol.outer_fold_id,
                inner_fold_id=0,
            )
            return pilot_store, pilot_spec.resolve(self.plan.manager)
        except (OOFProtocolError, OOFArtifactError) as exc:
            raise Task3CError(f"cannot resolve the existing CE pilot: {exc}") from exc

    def _validate_frozen_resolved_config(
        self, job: Task3CJob, resolved_config: Mapping[str, Any]
    ) -> None:
        """Check semantic frozen settings without requiring a machine path."""
        if resolved_config.get("expert") != job.expert_key:
            raise Task3CError(f"{job.job_id}: resolved config has the wrong expert")
        if resolved_config.get("seed") != self.plan.protocol.training_seed:
            raise Task3CError(f"{job.job_id}: resolved config has the wrong training seed")
        model = resolved_config.get("model", {})
        if model.get("arch") != "resnet32" or model.get("num_classes") != 100:
            raise Task3CError(f"{job.job_id}: resolved config has the wrong model")
        data = resolved_config.get("data", {})
        if data.get("imbalance_ratio") != 100.0 or data.get("batch_size") != 128:
            raise Task3CError(f"{job.job_id}: resolved config changes the data recipe")
        schedule = resolved_config.get("schedule", {})
        if (
            schedule.get("epochs") != TASK3C_EPOCHS
            or schedule.get("warmup_epochs") != 5
            or tuple(schedule.get("decay_epochs", ())) != (160, 180)
        ):
            raise Task3CError(f"{job.job_id}: resolved config changes the schedule")
        optimiser = resolved_config.get("optimiser", {})
        if (
            optimiser.get("name") != "sgd"
            or optimiser.get("lr") != 0.1
            or optimiser.get("momentum") != 0.9
            or optimiser.get("weight_decay") != 2.0e-4
            or optimiser.get("nesterov")
        ):
            raise Task3CError(f"{job.job_id}: resolved config changes the optimizer")

    def inspect(self) -> tuple[Task3CJobStatus, ...]:
        statuses: list[Task3CJobStatus] = []
        pilot_resolution = self._pilot_store_and_context()
        for job in self.plan.jobs:
            artifact_dir = self.artifact_dir(job)
            if job.is_pilot:
                if pilot_resolution is None:
                    statuses.append(
                        Task3CJobStatus(
                            job.job_id,
                            "missing",
                            artifact_dir,
                            "required CE pilot directory is absent",
                        )
                    )
                    continue
                pilot_store, context = pilot_resolution
                try:
                    completed = pilot_store.validate_completed_run(context)
                    self._validate_frozen_resolved_config(
                        job, completed.resolved_config
                    )
                except (OOFArtifactError, Task3CError, OSError, RuntimeError) as exc:
                    statuses.append(
                        Task3CJobStatus(job.job_id, "invalid", artifact_dir, str(exc))
                    )
                else:
                    statuses.append(
                        Task3CJobStatus(
                            job.job_id,
                            "validated_existing",
                            artifact_dir,
                            "validated read-only; original pilot files were not rewritten",
                        )
                    )
                continue

            if not artifact_dir.exists():
                statuses.append(Task3CJobStatus(job.job_id, "missing", artifact_dir))
                continue
            context = self.plan.context_for(job)
            try:
                completed = self.store.validate_completed_run(context)
                self._validate_frozen_resolved_config(job, completed.resolved_config)
            except (OOFArtifactError, Task3CError, OSError, RuntimeError) as exc:
                if not self.store.prediction_path(context).exists():
                    statuses.append(
                        Task3CJobStatus(
                            job.job_id,
                            "partial",
                            artifact_dir,
                            "run is present but has no validated complete prediction: "
                            + str(exc),
                        )
                    )
                else:
                    # A session may have written the complete prediction file
                    # and stopped before the final metadata update.  Accept
                    # this only when the existing prediction, checkpoint
                    # linkage, and frozen config are independently readable;
                    # the next OOFPipeline call will finalize it idempotently.
                    try:
                        self.store.load_predictions(context)
                        resolved_config = json.loads(
                            self.store.config_path(context).read_text()
                        )
                        self._validate_frozen_resolved_config(job, resolved_config)
                    except (OOFArtifactError, Task3CError, OSError, RuntimeError, json.JSONDecodeError) as recovery_exc:
                        statuses.append(
                            Task3CJobStatus(
                                job.job_id,
                                "invalid",
                                artifact_dir,
                                str(recovery_exc),
                            )
                        )
                    else:
                        statuses.append(
                            Task3CJobStatus(
                                job.job_id,
                                "partial",
                                artifact_dir,
                                "prediction is valid; completion metadata will be recovered",
                            )
                        )
            else:
                statuses.append(
                    Task3CJobStatus(
                        job.job_id,
                        "validated_existing",
                        artifact_dir,
                        "validated existing Task 3C artifact",
                    )
                )
        return tuple(statuses)

    @staticmethod
    def summarize(statuses: Sequence[Task3CJobStatus]) -> dict[str, int]:
        summary = {
            "required": len(statuses),
            "validated_existing": 0,
            "missing": 0,
            "partial": 0,
            "invalid": 0,
        }
        for status in statuses:
            if status.state not in summary:
                raise Task3CError(f"unknown Task 3C job state: {status.state}")
            if status.state != "required":
                summary[status.state] += 1
        return summary

    def print_plan(self, statuses: Sequence[Task3CJobStatus]) -> None:
        summary = self.summarize(statuses)
        print(f"Task 3C required jobs: {summary['required']}")
        print(f"  validated existing: {summary['validated_existing']}")
        print(f"  missing: {summary['missing']}")
        print(f"  partial: {summary['partial']}")
        print(f"  invalid: {summary['invalid']}")
        for status in statuses:
            suffix = f" — {status.detail}" if status.detail else ""
            print(f"  [{status.state}] {status.job_id}: {status.artifact_dir}{suffix}")

    def run_missing(
        self,
        *,
        max_jobs: int | None = None,
        execute_full: bool = False,
    ) -> tuple[Task3CJobStatus, ...]:
        if max_jobs is not None and max_jobs < 1:
            raise Task3CError("max_jobs must be positive when supplied")
        statuses = self.inspect()
        self.print_plan(statuses)
        by_id = {job.job_id: job for job in self.plan.jobs}
        pilot = next(status for status in statuses if status.job_id == "ce/inner_0")
        if pilot.state != "validated_existing":
            raise Task3CError(
                "refusing Task 3C execution until the existing CE pilot is validated; "
                f"current state is {pilot.state} ({pilot.detail})"
            )
        invalid = [status for status in statuses if status.state == "invalid"]
        if invalid:
            raise Task3CError(
                "refusing to overwrite invalid existing artifacts: "
                + ", ".join(status.job_id for status in invalid)
            )
        pending = [
            status
            for status in statuses
            if status.job_id != "ce/inner_0"
            and status.state in {"missing", "partial"}
        ]
        if max_jobs is not None:
            pending = pending[:max_jobs]
        if pending and not execute_full:
            raise Task3CError(
                "full 200-epoch Task 3C execution requires --execute-full; "
                "dry-run and validation do not launch training"
            )

        self.plan.write_batch_manifest()
        for status in pending:
            job = by_id[status.job_id]
            config = self.plan.config_for(job)
            # OOFPipeline applies the same seed/device/epoch/checkpoint
            # overrides below. A restarted process therefore uses the same
            # single-run recovery path as a direct invocation.
            self.pipeline.run(job.spec, config)
            completed = self.store.validate_completed_run(
                self.plan.context_for(job)
            )
            self._validate_frozen_resolved_config(job, completed.resolved_config)
            print(f"validated completed job: {job.job_id}")
        return self.inspect()

    def validated_runs(self) -> dict[str, OOFCompletedRun]:
        """Return all 16 validated runs, or fail before alignment/reporting."""
        statuses = self.inspect()
        failures = [status for status in statuses if status.state != "validated_existing"]
        if failures:
            details = "; ".join(
                f"{status.job_id}={status.state}: {status.detail}" for status in failures
            )
            raise Task3CError("cannot assemble Task 3C OOF data: " + details)

        runs: dict[str, OOFCompletedRun] = {}
        pilot_resolution = self._pilot_store_and_context()
        if pilot_resolution is None:
            raise Task3CError("CE pilot disappeared before alignment")
        pilot_store, pilot_context = pilot_resolution
        pilot_job = next(job for job in self.plan.jobs if job.is_pilot)
        runs[pilot_job.job_id] = pilot_store.validate_completed_run(pilot_context)
        for job in self.plan.jobs:
            if job.is_pilot:
                continue
            runs[job.job_id] = self.store.validate_completed_run(
                self.plan.context_for(job)
            )
        self.plan.write_batch_manifest()
        return runs


@dataclass
class AlignedOOFDataset:
    """Aligned inner-OOF logits and labels for the fixed four-expert pool."""

    sample_indices: np.ndarray
    outer_fold_ids: np.ndarray
    inner_fold_ids: np.ndarray
    labels: np.ndarray
    logits: np.ndarray
    expert_names: tuple[str, ...]
    metadata: dict[str, Any]

    def validate(self, manager: NestedOOFFoldManager) -> None:
        indices = np.asarray(self.sample_indices)
        outer_ids = np.asarray(self.outer_fold_ids)
        inner_ids = np.asarray(self.inner_fold_ids)
        labels = np.asarray(self.labels)
        logits = np.asarray(self.logits)
        if (
            indices.ndim != 1
            or outer_ids.shape != indices.shape
            or inner_ids.shape != indices.shape
        ):
            raise Task3CError("aligned OOF index arrays have inconsistent shapes")
        if labels.shape != indices.shape:
            raise Task3CError("aligned OOF labels are not aligned with sample IDs")
        if logits.ndim != 3 or logits.shape[:2] != (len(indices), len(self.expert_names)):
            raise Task3CError("aligned OOF logits do not match the expert axis")
        if logits.shape[2] != manager.num_classes:
            raise Task3CError("aligned OOF logits have the wrong class dimension")
        if not np.isfinite(logits).all():
            raise Task3CError("aligned OOF logits contain non-finite values")
        if len(np.unique(indices)) != len(indices):
            raise Task3CError("aligned OOF sample IDs contain duplicates")
        if tuple(self.expert_names) != manager.expert_order:
            raise Task3CError("aligned OOF expert order disagrees with the fold manifest")
        expected = set(manager.outer_fold(TASK3C_OUTER_FOLD).expert_training_indices)
        if set(indices.tolist()) != expected:
            raise Task3CError(
                "aligned OOF population must be exactly outer-fold-0 development; "
                "outer evaluation images must remain excluded"
            )
        labels_by_index = dict(zip(manager.canonical_indices, manager.training_labels))
        if any(
            int(label) != labels_by_index[int(index)]
            for index, label in zip(indices, labels)
        ):
            raise Task3CError("aligned OOF labels disagree with canonical training labels")
        if np.any(outer_ids != TASK3C_OUTER_FOLD):
            raise Task3CError("aligned OOF contains a nonzero outer fold")
        for inner_id in TASK3C_INNER_FOLDS:
            expected_inner = set(manager.inner_fold(0, inner_id).prediction_indices)
            actual_inner = set(indices[inner_ids == inner_id].tolist())
            if actual_inner != expected_inner:
                raise Task3CError(
                    f"aligned OOF inner fold {inner_id} is missing, duplicated, or misassigned"
                )

    @property
    def num_samples(self) -> int:
        return int(len(self.sample_indices))

    def select_inner_folds(self, inner_fold_ids: Sequence[int]) -> "AlignedOOFDataset":
        requested = tuple(int(value) for value in inner_fold_ids)
        if not requested or any(value not in TASK3C_INNER_FOLDS for value in requested):
            raise Task3CError(f"invalid inner-fold selection: {requested}")
        mask = np.isin(self.inner_fold_ids, np.asarray(requested, dtype=np.int64))
        return AlignedOOFDataset(
            sample_indices=self.sample_indices[mask].copy(),
            outer_fold_ids=self.outer_fold_ids[mask].copy(),
            inner_fold_ids=self.inner_fold_ids[mask].copy(),
            labels=self.labels[mask].copy(),
            logits=self.logits[mask].copy(),
            expert_names=self.expert_names,
            metadata={
                **self.metadata,
                "selected_inner_fold_ids": list(requested),
            },
        )

    def save(
        self,
        directory: str | Path,
        *,
        manager: NestedOOFFoldManager,
    ) -> tuple[Path, Path]:
        self.validate(manager)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        arrays_path = directory / "aligned_oof.npz"
        metadata_path = directory / "aligned_oof_metadata.json"
        if arrays_path.exists():
            try:
                with np.load(arrays_path, allow_pickle=False) as existing:
                    for key in (
                        "sample_indices",
                        "outer_fold_ids",
                        "inner_fold_ids",
                        "labels",
                        "logits",
                    ):
                        if not np.array_equal(existing[key], getattr(self, key)):
                            raise Task3CError(
                                f"existing aligned OOF array differs for {key}: {arrays_path}"
                            )
            except (OSError, KeyError, ValueError) as exc:
                raise Task3CError(
                    f"cannot validate existing aligned OOF arrays: {arrays_path}"
                ) from exc
        else:
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".npz", dir=directory, delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
            try:
                np.savez_compressed(
                    temporary_path,
                    sample_indices=self.sample_indices,
                    outer_fold_ids=self.outer_fold_ids,
                    inner_fold_ids=self.inner_fold_ids,
                    labels=self.labels,
                    logits=self.logits,
                )
                os.replace(temporary_path, arrays_path)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()

        metadata = {
            **self.metadata,
            "schema_version": ALIGNED_OOF_SCHEMA_VERSION,
            "expert_names": list(self.expert_names),
            "num_samples": self.num_samples,
            "num_classes": int(self.logits.shape[2]),
            "arrays_file": arrays_path.name,
            "arrays_sha256": _sha256_file(arrays_path),
            "outer_evaluation_excluded": True,
        }
        _write_json_once(metadata_path, metadata)
        return arrays_path, metadata_path

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        manager: NestedOOFFoldManager | None = None,
    ) -> "AlignedOOFDataset":
        directory = Path(directory)
        metadata_path = directory / "aligned_oof_metadata.json"
        arrays_path = directory / "aligned_oof.npz"
        try:
            metadata = json.loads(metadata_path.read_text())
            with np.load(arrays_path, allow_pickle=False) as arrays:
                dataset = cls(
                    sample_indices=np.asarray(arrays["sample_indices"]).copy(),
                    outer_fold_ids=np.asarray(arrays["outer_fold_ids"]).copy(),
                    inner_fold_ids=np.asarray(arrays["inner_fold_ids"]).copy(),
                    labels=np.asarray(arrays["labels"]).copy(),
                    logits=np.asarray(arrays["logits"]).copy(),
                    expert_names=tuple(metadata["expert_names"]),
                    metadata=dict(metadata),
                )
        except (OSError, KeyError, json.JSONDecodeError, ValueError) as exc:
            raise Task3CError(f"cannot load aligned OOF dataset from {directory}") from exc
        if metadata.get("schema_version") != ALIGNED_OOF_SCHEMA_VERSION:
            raise Task3CError("unsupported aligned OOF dataset schema")
        if metadata.get("arrays_sha256") != _sha256_file(arrays_path):
            raise Task3CError("aligned OOF array hash does not match metadata")
        if manager is not None:
            dataset.validate(manager)
        return dataset


class OOFAlignmentBuilder:
    """Join four independently stored single-expert artifacts by sample ID."""

    def __init__(self, plan: Task3CBatchPlan) -> None:
        self.plan = plan

    def build(self, runs: Mapping[str, OOFCompletedRun]) -> AlignedOOFDataset:
        expected_job_ids = {job.job_id for job in self.plan.jobs}
        if set(runs) != expected_job_ids:
            missing = sorted(expected_job_ids - set(runs))
            extra = sorted(set(runs) - expected_job_ids)
            raise Task3CError(
                "OOF run set is not the frozen 16-job matrix: "
                f"missing={missing}, extra={extra}"
            )

        for job in self.plan.jobs:
            run = runs[job.job_id]
            context = run.context
            if (
                context.expert_key != job.expert_key
                or context.expert_name != job.expert_name
                or context.outer_fold_id != TASK3C_OUTER_FOLD
                or context.inner_fold_id != job.inner_fold_id
                or context.training_seed != self.plan.protocol.training_seed
            ):
                raise Task3CError(
                    f"{job.job_id}: validated run context does not match the frozen job"
                )
            if any(
                record.expert_id != job.expert_name
                for record in run.artifact.records
            ):
                raise Task3CError(
                    f"{job.job_id}: prediction artifact contains a different expert"
                )

        rows: list[tuple[int, int, int, list[int], list[np.ndarray]]] = []
        labels_by_index = dict(
            zip(self.plan.manager.canonical_indices, self.plan.manager.training_labels)
        )
        for inner_id in TASK3C_INNER_FOLDS:
            inner = self.plan.manager.inner_fold(TASK3C_OUTER_FOLD, inner_id)
            expected_indices = tuple(inner.prediction_indices)
            records_by_expert: dict[str, dict[int, Any]] = {}
            for expert_name in self.plan.protocol.expert_names:
                job_id = next(
                    job.job_id
                    for job in self.plan.jobs
                    if job.expert_name == expert_name
                    and job.inner_fold_id == inner_id
                )
                artifact = runs[job_id].artifact
                artifact.validate(self.plan.manager, require_complete=False)
                records = {record.sample_index: record for record in artifact.records}
                if tuple(records) != expected_indices:
                    if set(records) != set(expected_indices):
                        raise Task3CError(
                            f"{job_id}: prediction IDs do not match inner fold {inner_id}"
                        )
                    raise Task3CError(
                        f"{job_id}: prediction IDs are not in manifest order"
                    )
                records_by_expert[expert_name] = records

            for sample_index in expected_indices:
                records = [
                    records_by_expert[name][sample_index]
                    for name in self.plan.protocol.expert_names
                ]
                labels = {int(record.training_label) for record in records}
                if labels != {labels_by_index[int(sample_index)]}:
                    raise Task3CError(f"sample {sample_index}: expert labels are misaligned")
                rows.append(
                    (
                        int(sample_index),
                        TASK3C_OUTER_FOLD,
                        inner_id,
                        [int(record.training_label) for record in records],
                        [
                            np.asarray(record.logits, dtype=np.float32)
                            for record in records
                        ],
                    )
                )

        rows.sort(key=lambda row: row[0])
        dataset = AlignedOOFDataset(
            sample_indices=np.asarray([row[0] for row in rows], dtype=np.int64),
            outer_fold_ids=np.asarray([row[1] for row in rows], dtype=np.int64),
            inner_fold_ids=np.asarray([row[2] for row in rows], dtype=np.int64),
            labels=np.asarray([row[3][0] for row in rows], dtype=np.int64),
            logits=np.asarray(
                [[logit for logit in row[4]] for row in rows], dtype=np.float32
            ),
            expert_names=tuple(self.plan.protocol.expert_names),
            metadata={
                "schema_version": ALIGNED_OOF_SCHEMA_VERSION,
                "outer_fold_id": TASK3C_OUTER_FOLD,
                "outer_evaluation_size": len(
                    self.plan.manager.outer_fold(TASK3C_OUTER_FOLD).evaluation_indices
                ),
                "inner_fold_ids": list(TASK3C_INNER_FOLDS),
                "router_fit_inner_fold_ids": [1, 2, 3],
                "router_selection_inner_fold_ids": [0],
                "canonical_training_index_sha256": self.plan.manager.canonical_training_index_sha256,
                "fold_generation_seed": self.plan.manager.fold_generation_seed,
                "source_job_ids": sorted(runs),
                "source_run_directories": {
                    job_id: str(run.run_dir) for job_id, run in sorted(runs.items())
                },
                "source_checkpoints": {
                    job_id: {
                        "path": str(run.checkpoint_path),
                        "sha256": run.metadata["checkpoint"]["sha256"],
                    }
                    for job_id, run in sorted(runs.items())
                },
                "primary_diagnostic_partition": "router_fit_inner_folds_1_2_3",
                "outer_evaluation_excluded": True,
            },
        )
        dataset.validate(self.plan.manager)
        return dataset


def _diagnostic_payload(
    diagnostics: ExpertDiagnostics,
    dataset: AlignedOOFDataset,
) -> dict[str, Any]:
    """Create the requested descriptive report without serializing arrays."""
    uniform_weights = np.full(
        (diagnostics.n_samples, diagnostics.num_experts),
        1.0 / diagnostics.num_experts,
        dtype=np.float64,
    )
    logit_uniform = diagnostics.evaluate_soft_mixture(uniform_weights, "logits")
    probability_uniform = diagnostics.evaluate_soft_mixture(
        uniform_weights, "probabilities"
    )
    return {
        "sample_count": diagnostics.n_samples,
        "inner_fold_ids": sorted(
            int(value) for value in np.unique(dataset.inner_fold_ids)
        ),
        "label_dependent": True,
        "descriptive_only": True,
        "no_router_fitted": True,
        "no_label_dependent_weights_optimized": True,
        "individual_experts_and_complementarity": diagnostics.complementarity(),
        "correctness_and_confidence_diagnostics": diagnostics.correctness_diagnostics(),
        "uniform_logit_ensemble": {
            "metrics": logit_uniform["metrics"],
            "combination": "uniform logit average",
        },
        "uniform_probability_ensemble": {
            "metrics": probability_uniform["metrics"],
            "combination": "uniform probability average",
        },
        "leave_one_out_ensemble_contribution": diagnostics.ensemble_contribution(),
        "hard_routing_oracle_headroom": diagnostics.hard_routing_headroom(),
    }


class Task3CDiagnosticReporter:
    """Generate primary fit-partition and explicitly caveated full-dev reports."""

    def __init__(self, plan: Task3CBatchPlan) -> None:
        self.plan = plan

    def build(self, dataset: AlignedOOFDataset) -> dict[str, Any]:
        dataset.validate(self.plan.manager)
        class_counts = np.asarray(
            self.plan.manager.canonical_class_counts, dtype=np.int64
        )
        fit_dataset = dataset.select_inner_folds((1, 2, 3))
        full_dataset = dataset.select_inner_folds((0, 1, 2, 3))

        fit_diagnostics = ExpertDiagnostics(
            logits=fit_dataset.logits,
            labels=fit_dataset.labels,
            class_counts=class_counts,
            expert_names=fit_dataset.expert_names,
        )
        full_diagnostics = ExpertDiagnostics(
            logits=full_dataset.logits,
            labels=full_dataset.labels,
            class_counts=class_counts,
            expert_names=full_dataset.expert_names,
        )
        return {
            "schema_version": DIAGNOSTIC_REPORT_SCHEMA_VERSION,
            "experiment_id": self.plan.experiment_id,
            "dataset": "CIFAR-100-LT",
            "imbalance_ratio": 100.0,
            "training_seed": self.plan.protocol.training_seed,
            "fold_generation_seed": self.plan.protocol.fold_seed,
            "outer_fold_id": self.plan.protocol.outer_fold_id,
            "outer_evaluation_size": len(
                self.plan.manager.outer_fold(self.plan.protocol.outer_fold_id).evaluation_indices
            ),
            "expert_order": list(self.plan.protocol.expert_names),
            "class_counts_source": "NestedOOFFoldManager canonical training labels",
            "head_medium_tail_source": "scripts.base_trainer.compute_class_groups",
            "research_boundary": {
                "router_fit_inner_fold_ids": [1, 2, 3],
                "router_selection_inner_fold_ids": [0],
                "outer_evaluation_excluded": True,
                "router_fitting_implemented": False,
                "inference_time_oracle_used": False,
            },
            "primary_router_fit_partition": _diagnostic_payload(
                fit_diagnostics, fit_dataset
            ),
            "full_development_descriptive_partition": {
                "disclosure": (
                    "This descriptive report includes inner-fold-0 labels. It is not an "
                    "untouched partition for research decisions informed by these observations."
                ),
                **_diagnostic_payload(full_diagnostics, full_dataset),
            },
        }

    def write(self, dataset: AlignedOOFDataset, directory: str | Path) -> Path:
        path = Path(directory) / "diagnostics.json"
        return _write_json_once(path, self.build(dataset))


__all__ = [
    "ALIGNED_OOF_SCHEMA_VERSION",
    "DIAGNOSTIC_REPORT_SCHEMA_VERSION",
    "TASK3C_CONFIG_FILES",
    "TASK3C_EXPERIMENT_ID",
    "TASK3C_PILOT_DEFAULT",
    "Task3CBatchPlan",
    "Task3CBatchRunner",
    "Task3CError",
    "Task3CJob",
    "Task3CJobStatus",
    "Task3CProtocol",
    "AlignedOOFDataset",
    "OOFAlignmentBuilder",
    "Task3CDiagnosticReporter",
]
