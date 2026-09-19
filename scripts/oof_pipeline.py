"""Execution and provenance orchestration for nested OOF expert runs.

The active ``scripts/train.py`` path intentionally remains full-population
training.  This module is a separate development path that requires a resolved
fold context, routes data through :class:`FoldAwareDataModule`, and stores every
checkpoint/prediction beside immutable provenance.

It does not fit a router or access the CIFAR-100 test split.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Mapping

import torch

from data.nested_oof import (
    NestedOOFFoldManager,
    OOFArtifactValidationError,
    OOFPredictionArtifact,
    OOFPredictionRecord,
    OOFProtocolError,
)
from data.oof_datamodule import FoldAwareDataModule
from scripts.base_trainer import CheckpointValidationError, validate_checkpoint_metadata
from scripts.config import TrainingConfig
from scripts.trainers import build_trainer


OOF_RUN_SCHEMA_VERSION = "nested_oof_run.v1"
_EXPERT_NAMES = {
    "ce": "CE",
    "logit_adjusted": "LAL",
    "balanced_softmax": "BalancedSoftmax",
    "mixup": "Mixup",
}


class OOFArtifactError(OOFProtocolError):
    """Raised when an OOF artifact is unsafe, incomplete, or inconsistent."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise OOFArtifactError(f"value is not JSON serializable: {value!r}") from exc


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise OOFArtifactError(f"cannot hash artifact: {path}") from exc
    return digest.hexdigest()


def _safe_component(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value or not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise OOFArtifactError(
            f"{name} must be a non-empty path-safe identifier, got {value!r}"
        )
    return value


@contextmanager
def _deterministic_inference_threads():
    """Avoid CPU reduction-order drift when an artifact is collected twice."""
    previous = torch.get_num_threads()
    if previous != 1:
        torch.set_num_threads(1)
    try:
        yield
    finally:
        if previous != 1:
            torch.set_num_threads(previous)


@dataclass(frozen=True)
class OOFRunSpec:
    """Explicit selection of one expert/fold/seed execution."""

    experiment_id: str
    expert: str
    training_seed: int
    outer_fold_id: int
    inner_fold_id: int | None = None
    device: str | None = None
    epochs: int | None = None
    max_batches: int | None = None

    def resolve(self, manager: NestedOOFFoldManager) -> "OOFRunContext":
        """Resolve IDs to immutable memberships from ``manager``."""
        _safe_component(self.experiment_id, name="experiment_id")
        if self.expert not in _EXPERT_NAMES:
            raise OOFArtifactError(
                f"unknown OOF expert {self.expert!r}; expected one of "
                f"{sorted(_EXPERT_NAMES)}"
            )
        expert_name = _EXPERT_NAMES[self.expert]
        if expert_name not in manager.expert_order:
            raise OOFArtifactError(
                f"expert {expert_name!r} is absent from the manifest expert ordering"
            )
        if isinstance(self.training_seed, bool) or not isinstance(self.training_seed, int):
            raise OOFArtifactError("training_seed must be an integer")
        if self.training_seed < 0:
            raise OOFArtifactError("training_seed must be non-negative")
        if self.epochs is not None and self.epochs < 1:
            raise OOFArtifactError("epochs must be positive when supplied")
        if self.max_batches is not None and self.max_batches < 1:
            raise OOFArtifactError("max_batches must be positive when supplied")

        try:
            outer = manager.outer_fold(self.outer_fold_id)
        except OOFProtocolError as exc:
            raise OOFArtifactError(str(exc)) from exc

        if self.inner_fold_id is None:
            training_indices = outer.expert_training_indices
            prediction_indices = outer.evaluation_indices
            training_class_counts = outer.expert_training_class_counts
            prediction_class_counts = outer.evaluation_class_counts
            membership_hash = outer.expert_training_membership_hash
            role = "outer_evaluation"
        else:
            try:
                inner = manager.inner_fold(self.outer_fold_id, self.inner_fold_id)
            except OOFProtocolError as exc:
                raise OOFArtifactError(str(exc)) from exc
            training_indices = inner.expert_training_indices
            prediction_indices = inner.prediction_indices
            training_class_counts = inner.expert_training_class_counts
            prediction_class_counts = inner.prediction_class_counts
            membership_hash = inner.expert_training_membership_hash
            role = "inner_oof"

        if set(training_indices) & set(prediction_indices):
            raise OOFArtifactError("resolved run has overlapping training/prediction IDs")
        if set(prediction_indices) & set(outer.evaluation_indices) and self.inner_fold_id is not None:
            raise OOFArtifactError(
                "inner prediction population unexpectedly contains outer evaluation IDs"
            )

        return OOFRunContext(
            spec=self,
            manager=manager,
            expert_key=self.expert,
            expert_name=expert_name,
            role=role,
            training_indices=tuple(training_indices),
            prediction_indices=tuple(prediction_indices),
            training_class_counts=tuple(training_class_counts),
            prediction_class_counts=tuple(prediction_class_counts),
            training_membership_hash=membership_hash,
        )


@dataclass(frozen=True)
class OOFRunContext:
    """Immutable, validated fold membership for one expert run."""

    spec: OOFRunSpec
    manager: NestedOOFFoldManager
    expert_key: str
    expert_name: str
    role: str
    training_indices: tuple[int, ...]
    prediction_indices: tuple[int, ...]
    training_class_counts: tuple[int, ...]
    prediction_class_counts: tuple[int, ...]
    training_membership_hash: str

    @property
    def outer_fold_id(self) -> int:
        return int(self.spec.outer_fold_id)

    @property
    def inner_fold_id(self) -> int | None:
        return self.spec.inner_fold_id

    @property
    def training_seed(self) -> int:
        return int(self.spec.training_seed)

    @property
    def run_id(self) -> str:
        inner = "outer_eval" if self.inner_fold_id is None else f"inner_{self.inner_fold_id}"
        return (
            f"expert_{self.expert_name}/seed_{self.training_seed}/"
            f"outer_{self.outer_fold_id}/{inner}"
        )


class OOFArtifactStore:
    """Own the versioned artifact layout and reject incompatible reuse."""

    def __init__(
        self,
        *,
        root: str | Path,
        manager: NestedOOFFoldManager,
        experiment_id: str,
        canonical_checkpoint_dir: str | Path = "checkpoints",
    ) -> None:
        _safe_component(experiment_id, name="experiment_id")
        self.root = Path(root).resolve()
        self.manager = manager
        self.experiment_id = experiment_id
        self.canonical_checkpoint_dir = Path(canonical_checkpoint_dir).resolve()
        if self.root == self.canonical_checkpoint_dir or self.canonical_checkpoint_dir in self.root.parents:
            raise OOFArtifactError(
                "OOF artifact root may not be the canonical checkpoint directory "
                "or a child of it"
            )
        self.base_dir = self.root / experiment_id
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.manager.validate_membership()

    @property
    def manifest_path(self) -> Path:
        return self.base_dir / "fold_manifest.json"

    def run_dir(self, context: OOFRunContext) -> Path:
        if context.manager is not self.manager:
            raise OOFArtifactError("run context belongs to a different fold manager")
        return self.base_dir / context.run_id

    def checkpoint_path(self, context: OOFRunContext) -> Path:
        return self.run_dir(context) / "checkpoints" / (
            f"{context.expert_name}_seed{context.training_seed}_final.pt"
        )

    def prediction_path(self, context: OOFRunContext) -> Path:
        return self.run_dir(context) / "predictions.json"

    def metadata_path(self, context: OOFRunContext) -> Path:
        return self.run_dir(context) / "run_metadata.json"

    def config_path(self, context: OOFRunContext) -> Path:
        return self.run_dir(context) / "resolved_config.json"

    def log_path(self, context: OOFRunContext) -> Path:
        return self.run_dir(context) / "execution.log"

    def write_manifest(self) -> Path:
        self._write_once(self.manifest_path, self.manager.manifest().to_json())
        return self.manifest_path

    def prepare_run(
        self,
        context: OOFRunContext,
        resolved_config: Mapping[str, Any],
    ) -> Path:
        """Create or verify an exact run directory before model construction."""
        self._validate_context(context)
        self.write_manifest()
        run_dir = self.run_dir(context)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        config_json = _canonical_json(dict(resolved_config))
        self._write_once(self.config_path(context), config_json + "\n")

        metadata = self._static_metadata(context, dict(resolved_config))
        metadata_path = self.metadata_path(context)
        if metadata_path.exists():
            try:
                existing = json.loads(metadata_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise OOFArtifactError(
                    f"cannot read existing OOF run metadata: {metadata_path}"
                ) from exc
            for key, value in metadata.items():
                if key == "status":
                    continue
                if existing.get(key) != value:
                    raise OOFArtifactError(
                        f"existing run metadata disagrees for {key!r}; refusing to "
                        "mix incompatible configurations or fold memberships"
                    )
        else:
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        if not self.log_path(context).exists():
            self.log_path(context).write_text("OOF run prepared\n")
        return run_dir

    def record_checkpoint(
        self,
        context: OOFRunContext,
        checkpoint_path: str | Path,
        checkpoint_sha256: str | None = None,
    ) -> str:
        """Validate and record a final checkpoint tied to this exact run."""
        self._validate_context(context)
        path = Path(checkpoint_path).resolve()
        run_dir = self.run_dir(context).resolve()
        if run_dir not in path.parents:
            raise OOFArtifactError(
                "OOF checkpoint must be written inside its dedicated run directory"
            )
        if not path.exists():
            raise OOFArtifactError(f"OOF checkpoint does not exist: {path}")
        actual_sha = _sha256_file(path)
        if checkpoint_sha256 is not None and checkpoint_sha256 != actual_sha:
            raise OOFArtifactError("checkpoint SHA-256 does not match its contents")
        metadata = self._read_metadata(context)
        resolved_config = json.loads(self.config_path(context).read_text())
        expected_epoch = None
        schedule = resolved_config.get("schedule")
        if isinstance(schedule, Mapping) and "epochs" in schedule:
            expected_epoch = int(schedule["epochs"])
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
            validate_checkpoint_metadata(
                state,
                path=path,
                expected_expert=context.expert_name,
                expected_seed=context.training_seed,
                expected_epoch=expected_epoch,
                require_final=True,
            )
        except (CheckpointValidationError, OSError, RuntimeError) as exc:
            raise OOFArtifactError(f"checkpoint provenance validation failed: {exc}") from exc

        existing = metadata.get("checkpoint")
        checkpoint_record = {
            "path": str(path.relative_to(run_dir)),
            "sha256": actual_sha,
            "epoch": int(state["epoch"]),
        }
        if existing is not None and existing != checkpoint_record:
            raise OOFArtifactError("run already records a different checkpoint")
        metadata["checkpoint"] = checkpoint_record
        metadata["status"] = "training_complete"
        self._write_metadata(context, metadata)
        self._append_log(context, f"checkpoint validated: {checkpoint_record['path']}\n")
        return actual_sha

    def write_predictions(
        self,
        context: OOFRunContext,
        artifact: OOFPredictionArtifact,
    ) -> Path:
        """Validate and persist a complete single-run prediction artifact."""
        self._validate_context(context)
        artifact.validate(self.manager, require_complete=False)
        metadata = self._read_metadata(context)
        checkpoint = metadata.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise OOFArtifactError("cannot write predictions before checkpoint validation")
        self._validate_run_artifact(context, artifact, checkpoint)
        serialized = artifact.to_json()
        path = self.prediction_path(context)
        self._write_once(path, serialized)
        metadata["status"] = "complete"
        metadata["prediction"] = {
            "path": str(path.relative_to(self.run_dir(context))),
            "sha256": _sha256_file(path),
            "record_count": len(artifact.records),
        }
        self._write_metadata(context, metadata)
        self._append_log(context, f"predictions validated: {len(artifact.records)} records\n")
        return path

    def load_predictions(self, context: OOFRunContext) -> OOFPredictionArtifact:
        path = self.prediction_path(context)
        if not path.exists():
            raise OOFArtifactError(f"OOF prediction artifact is missing: {path}")
        try:
            artifact = OOFPredictionArtifact.from_json(path.read_text())
        except OOFArtifactValidationError as exc:
            raise OOFArtifactError(f"invalid OOF prediction artifact: {exc}") from exc
        metadata = self._read_metadata(context)
        checkpoint = metadata.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise OOFArtifactError("prediction artifact has no validated checkpoint")
        artifact.validate(self.manager, require_complete=False)
        self._validate_run_artifact(context, artifact, checkpoint)
        return artifact

    def _static_metadata(
        self, context: OOFRunContext, resolved_config: Mapping[str, Any]
    ) -> dict[str, Any]:
        manifest_json = self.manager.manifest().to_json()
        return {
            "schema_version": OOF_RUN_SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "role": context.role,
            "expert_key": context.expert_key,
            "expert_name": context.expert_name,
            "training_seed": context.training_seed,
            "outer_fold_id": context.outer_fold_id,
            "inner_fold_id": context.inner_fold_id,
            "canonical_training_index_sha256": (
                self.manager.canonical_training_index_sha256
            ),
            "manifest_sha256": _sha256_text(manifest_json),
            "training_indices": list(context.training_indices),
            "prediction_indices": list(context.prediction_indices),
            "training_membership_sha256": context.training_membership_hash,
            "training_class_counts": list(context.training_class_counts),
            "prediction_class_counts": list(context.prediction_class_counts),
            "resolved_config_sha256": _sha256_text(
                _canonical_json(dict(resolved_config))
            ),
            "status": "prepared",
        }

    def _validate_run_artifact(
        self,
        context: OOFRunContext,
        artifact: OOFPredictionArtifact,
        checkpoint: Mapping[str, Any],
    ) -> None:
        records = artifact.records
        if len(records) != len(context.prediction_indices):
            raise OOFArtifactValidationError(
                "OOF prediction artifact is incomplete for this run: "
                f"got {len(records)}, expected {len(context.prediction_indices)}"
            )
        expected_indices = list(context.prediction_indices)
        actual_indices = [record.sample_index for record in records]
        if actual_indices != expected_indices:
            if len(set(actual_indices)) != len(actual_indices):
                reason = "contains duplicate sample IDs"
            else:
                reason = "sample IDs are missing or misordered"
            raise OOFArtifactValidationError(
                f"OOF prediction artifact {reason}; records must follow held-out order"
            )
        config_hash = self._read_metadata(context)["resolved_config_sha256"]
        for record in records:
            if record.outer_fold_id != context.outer_fold_id:
                raise OOFArtifactValidationError("prediction record has a mismatched outer fold ID")
            if record.inner_fold_id != context.inner_fold_id:
                raise OOFArtifactValidationError("prediction record has a mismatched inner fold ID")
            if record.expert_id != context.expert_name:
                raise OOFArtifactValidationError("prediction record has a mismatched expert identity")
            if record.training_seed != context.training_seed:
                raise OOFArtifactValidationError("prediction record has a mismatched training seed")
            if record.expert_training_membership_hash != context.training_membership_hash:
                raise OOFArtifactValidationError("prediction record has a mismatched training membership")
            if record.checkpoint_sha256 != checkpoint["sha256"]:
                raise OOFArtifactValidationError("prediction record has a mismatched checkpoint hash")
            if record.resolved_config_sha256 != config_hash:
                raise OOFArtifactValidationError("prediction record has a mismatched resolved configuration")

    def _validate_context(self, context: OOFRunContext) -> None:
        if context.manager is not self.manager:
            raise OOFArtifactError("run context belongs to a different fold manager")
        try:
            expected = context.spec.resolve(self.manager)
        except OOFProtocolError as exc:
            raise OOFArtifactError(f"run context specification is invalid: {exc}") from exc
        if context != expected:
            raise OOFArtifactError(
                "run context fields are inconsistent with its explicit run specification"
            )
        if tuple(context.training_indices) != tuple(
            self._expected_training(context)
        ) or tuple(context.prediction_indices) != tuple(self._expected_prediction(context)):
            raise OOFArtifactError("run context membership is inconsistent with the manifest")

    def _expected_training(self, context: OOFRunContext) -> tuple[int, ...]:
        outer = self.manager.outer_fold(context.outer_fold_id)
        if context.inner_fold_id is None:
            return outer.expert_training_indices
        return self.manager.inner_fold(context.outer_fold_id, context.inner_fold_id).expert_training_indices

    def _expected_prediction(self, context: OOFRunContext) -> tuple[int, ...]:
        outer = self.manager.outer_fold(context.outer_fold_id)
        if context.inner_fold_id is None:
            return outer.evaluation_indices
        return self.manager.inner_fold(context.outer_fold_id, context.inner_fold_id).prediction_indices

    def _read_metadata(self, context: OOFRunContext) -> dict[str, Any]:
        path = self.metadata_path(context)
        if not path.exists():
            raise OOFArtifactError(f"OOF run metadata is missing: {path}")
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise OOFArtifactError(f"OOF run metadata is invalid: {path}") from exc
        if payload.get("schema_version") != OOF_RUN_SCHEMA_VERSION:
            raise OOFArtifactError("unsupported OOF run metadata schema")
        return payload

    def _write_metadata(self, context: OOFRunContext, metadata: Mapping[str, Any]) -> None:
        self.metadata_path(context).write_text(
            json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n"
        )

    @staticmethod
    def _write_once(path: Path, content: str) -> None:
        if path.exists():
            try:
                existing = path.read_text()
            except OSError as exc:
                raise OOFArtifactError(f"cannot read existing artifact: {path}") from exc
            if existing != content:
                raise OOFArtifactError(
                    f"refusing to overwrite an incompatible existing artifact: {path}"
                )
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    def _append_log(self, context: OOFRunContext, message: str) -> None:
        with self.log_path(context).open("a") as handle:
            handle.write(message)


@dataclass(frozen=True)
class OOFTrainingResult:
    trainer: Any
    checkpoint_path: Path
    checkpoint_sha256: str
    history: tuple[dict[str, Any], ...]


class OOFTrainingOrchestrator:
    """Train exactly one fold expert through the existing trainer registry."""

    def __init__(self, trainer_builder: Callable[..., Any] = build_trainer) -> None:
        self.trainer_builder = trainer_builder

    def train(
        self,
        *,
        context: OOFRunContext,
        config: TrainingConfig,
        data_module: FoldAwareDataModule,
        store: OOFArtifactStore,
        resolved_config: Mapping[str, Any] | None = None,
    ) -> OOFTrainingResult:
        if config.expert != context.expert_key:
            raise OOFArtifactError(
                f"config expert {config.expert!r} does not match run expert "
                f"{context.expert_key!r}"
            )
        try:
            config.validate("OOF pipeline")
        except (ValueError, TypeError) as exc:
            raise OOFArtifactError(f"invalid OOF training configuration: {exc}") from exc
        data_module.validate_context(context)
        config_payload = (
            config.to_dict() if resolved_config is None else dict(resolved_config)
        )
        store.prepare_run(context, config_payload)
        checkpoint_path = store.checkpoint_path(context)
        class_counts = data_module.class_counts(context)
        if checkpoint_path.exists():
            metadata = store._read_metadata(context)
            checkpoint = metadata.get("checkpoint")
            if isinstance(checkpoint, Mapping):
                checkpoint_sha = store.record_checkpoint(context, checkpoint_path)
                trainer = self.trainer_builder(
                    config,
                    class_counts=class_counts,
                    device=config.resolved_device,
                )
                trainer.load_checkpoint(checkpoint_path)
                return OOFTrainingResult(
                    trainer=trainer,
                    checkpoint_path=checkpoint_path,
                    checkpoint_sha256=checkpoint_sha,
                    history=tuple(),
                )
            raise OOFArtifactError(
                "an unrecorded checkpoint exists in the run directory; refusing "
                "to overwrite it"
            )

        # Membership validation above happens before this registry call, so an
        # accidental full-data loader cannot silently construct a model.
        trainer = self.trainer_builder(
            config,
            class_counts=class_counts,
            device=config.resolved_device,
        )
        history = trainer.train(data_module.training_loader(context))
        history_path = store.run_dir(context) / "training_history.json"
        trainer.save_history(path=str(history_path))
        if not checkpoint_path.exists():
            raise OOFArtifactError(
                "trainer completed without writing the expected final checkpoint "
                f"{checkpoint_path}"
            )
        checkpoint_sha = store.record_checkpoint(context, checkpoint_path)
        return OOFTrainingResult(
            trainer=trainer,
            checkpoint_path=checkpoint_path,
            checkpoint_sha256=checkpoint_sha,
            history=tuple(history),
        )


class OOFPredictionCollector:
    """Extract deterministic logits for one declared held-out population."""

    def __init__(
        self,
        *,
        manager: NestedOOFFoldManager,
        data_module: FoldAwareDataModule,
    ) -> None:
        self.manager = manager
        self.data_module = data_module

    def collect(
        self,
        *,
        context: OOFRunContext,
        trainer: Any,
        checkpoint_path: str | Path,
        checkpoint_sha256: str,
        resolved_config: Mapping[str, Any],
    ) -> OOFPredictionArtifact:
        if context.manager is not self.manager:
            raise OOFArtifactError("prediction context belongs to a different manager")
        self.data_module.validate_context(context)
        path = Path(checkpoint_path).resolve()
        if not path.exists():
            raise OOFArtifactError(f"prediction checkpoint is missing: {path}")
        actual_sha = _sha256_file(path)
        if actual_sha != checkpoint_sha256:
            raise OOFArtifactError("prediction checkpoint hash does not match its contents")
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
            validate_checkpoint_metadata(
                state,
                path=path,
                expected_expert=context.expert_name,
                expected_seed=context.training_seed,
                require_final=True,
            )
        except (CheckpointValidationError, OSError, RuntimeError) as exc:
            raise OOFArtifactError(f"prediction checkpoint is incompatible: {exc}") from exc

        if getattr(trainer, "expert_name", None) != context.expert_name:
            raise OOFArtifactError("trainer expert identity does not match the run context")
        if int(getattr(trainer, "seed", -1)) != context.training_seed:
            raise OOFArtifactError("trainer seed does not match the run context")
        try:
            trainer.load_checkpoint(path)
        except (CheckpointValidationError, RuntimeError) as exc:
            raise OOFArtifactError(
                f"checkpoint cannot be loaded by the requested trainer: {exc}"
            ) from exc
        trainer.model.eval()
        device = torch.device(getattr(trainer, "device", "cpu"))
        records: list[OOFPredictionRecord] = []
        observed_indices: list[int] = []
        with _deterministic_inference_threads(), torch.no_grad():
            for batch in self.data_module.prediction_loader(context):
                if len(batch) != 3:
                    raise OOFArtifactError(
                        "prediction loader must return image, label, and original sample ID"
                    )
                images, labels, sample_indices = batch
                images = images.to(device)
                output = trainer.model(images)
                logits = output[0] if isinstance(output, (tuple, list)) else output
                if not torch.is_tensor(logits) or logits.ndim != 2:
                    raise OOFArtifactError("expert prediction output must be a 2-D logits tensor")
                if logits.shape[1] != self.manager.num_classes:
                    raise OOFArtifactError(
                        f"expert returned {logits.shape[1]} classes, expected "
                        f"{self.manager.num_classes}"
                    )
                if logits.shape[0] != len(sample_indices) or logits.shape[0] != len(labels):
                    raise OOFArtifactError("prediction batch tensors are misaligned")
                for row, sample_index, label in zip(
                    logits.detach().cpu().numpy(),
                    sample_indices.tolist(),
                    labels.tolist(),
                ):
                    sample_index = int(sample_index)
                    observed_indices.append(sample_index)
                    records.append(
                        OOFPredictionRecord.create(
                            sample_index=sample_index,
                            training_label=int(label),
                            outer_fold_id=context.outer_fold_id,
                            inner_fold_id=context.inner_fold_id,
                            expert_id=context.expert_name,
                            training_seed=context.training_seed,
                            expert_training_membership_hash=context.training_membership_hash,
                            checkpoint_path=str(path),
                            checkpoint_sha256=checkpoint_sha256,
                            resolved_config=dict(resolved_config),
                            logits=row,
                        )
                    )

        if observed_indices != list(context.prediction_indices):
            raise OOFArtifactError(
                "prediction loader did not yield the declared held-out IDs in order"
            )
        artifact = OOFPredictionArtifact(
            expert_order=self.manager.expert_order,
            num_classes=self.manager.num_classes,
            records=tuple(records),
        )
        artifact.validate(self.manager, require_complete=False)
        return artifact


@dataclass(frozen=True)
class OOFRunResult:
    context: OOFRunContext
    checkpoint_path: Path
    prediction_path: Path
    checkpoint_sha256: str
    prediction_artifact: OOFPredictionArtifact


class OOFPipeline:
    """Coordinate one explicit fold-expert run without touching active paths."""

    def __init__(
        self,
        *,
        manager: NestedOOFFoldManager,
        store: OOFArtifactStore,
        data_module_factory: Callable[..., FoldAwareDataModule] = FoldAwareDataModule,
        trainer_builder: Callable[..., Any] = build_trainer,
    ) -> None:
        self.manager = manager
        self.store = store
        self.data_module_factory = data_module_factory
        self.training = OOFTrainingOrchestrator(trainer_builder)

    def run(self, spec: OOFRunSpec, base_config: TrainingConfig) -> OOFRunResult:
        if spec.experiment_id != self.store.experiment_id:
            raise OOFArtifactError(
                "run specification experiment_id does not match the artifact store"
            )
        context = spec.resolve(self.manager)
        if base_config.expert != context.expert_key:
            raise OOFArtifactError(
                f"base config expert {base_config.expert!r} does not match "
                f"requested expert {context.expert_key!r}"
            )
        config = base_config.replace(
            seed=context.training_seed,
            device=spec.device,
            epochs=spec.epochs,
            checkpoint_dir=str(self.store.run_dir(context) / "checkpoints"),
        )
        resolved_config = config.to_dict()
        resolved_config["resolved_device"] = config.resolved_device
        data_module = self.data_module_factory(
            manager=self.manager,
            root=config.data.root,
            imbalance_ratio=config.data.imbalance_ratio,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            pin_memory=config.data.pin_memory,
            seed=context.training_seed,
            max_batches=spec.max_batches,
        )
        training = self.training.train(
            context=context,
            config=config,
            data_module=data_module,
            store=self.store,
            resolved_config=resolved_config,
        )
        artifact = OOFPredictionCollector(
            manager=self.manager, data_module=data_module
        ).collect(
            context=context,
            trainer=training.trainer,
            checkpoint_path=training.checkpoint_path,
            checkpoint_sha256=training.checkpoint_sha256,
            resolved_config=resolved_config,
        )
        prediction_path = self.store.write_predictions(context, artifact)
        return OOFRunResult(
            context=context,
            checkpoint_path=training.checkpoint_path,
            prediction_path=prediction_path,
            checkpoint_sha256=training.checkpoint_sha256,
            prediction_artifact=artifact,
        )


@dataclass(frozen=True)
class ResourceReport:
    """Small environment report for pilot planning, not a benchmark result."""

    device: str
    gpu_name: str | None
    gpu_vram_gib: float | None
    cpu_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "gpu_name": self.gpu_name,
            "gpu_vram_gib": self.gpu_vram_gib,
            "cpu_count": self.cpu_count,
        }


def collect_resource_report() -> ResourceReport:
    import os

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return ResourceReport(
            device="cuda",
            gpu_name=torch.cuda.get_device_name(0),
            gpu_vram_gib=props.total_memory / (1024**3),
            cpu_count=os.cpu_count() or 1,
        )
    return ResourceReport(
        device="cpu",
        gpu_name=None,
        gpu_vram_gib=None,
        cpu_count=os.cpu_count() or 1,
    )
