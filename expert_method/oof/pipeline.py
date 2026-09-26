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
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Mapping

import torch

from data.nested_oof import (
    FoldManifest,
    NestedOOFFoldManager,
    OOFArtifactValidationError,
    OOFPredictionArtifact,
    OOFPredictionRecord,
    OOFProtocolError,
)
from data.oof_datamodule import FoldAwareDataModule
from scripts.analysis import ImmutableArtifactWriter
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


class OOFArtifactMissingError(OOFArtifactError):
    """Raised when a resumable run is missing a required artifact file."""


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


class AtomicMetadataWriter:
    """Persist mutable OOF run metadata without exposing partial JSON.

    Run metadata is the one intentionally mutable artifact: preparation records
    ``prepared``, checkpoint validation records ``training_complete`` and
    prediction persistence records ``complete``.  The replacement is staged in
    the destination directory, flushed to disk, and installed with
    :func:`os.replace`, so a failed write leaves the previous metadata file
    intact.  JSON rendering intentionally matches the historical writer byte
    for byte.
    """

    def __init__(self, *, error_type: type[Exception] = OOFArtifactError) -> None:
        self.error_type = error_type

    def write(self, path: Path, metadata: Mapping[str, Any]) -> None:
        rendered = json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                suffix=path.suffix or ".tmp",
                dir=path.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            temporary = None
        except OSError as exc:
            raise self.error_type(
                f"cannot atomically write OOF run metadata: {path}"
            ) from exc
        finally:
            if temporary is not None and temporary.exists():
                try:
                    temporary.unlink()
                except OSError:
                    pass


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
        read_only: bool = False,
        base_dir: str | Path | None = None,
    ) -> None:
        _safe_component(experiment_id, name="experiment_id")
        self.root = Path(root).resolve()
        self.manager = manager
        self.experiment_id = experiment_id
        self.read_only = bool(read_only)
        self.canonical_checkpoint_dir = Path(canonical_checkpoint_dir).resolve()
        if self.root == self.canonical_checkpoint_dir or self.canonical_checkpoint_dir in self.root.parents:
            raise OOFArtifactError(
                "OOF artifact root may not be the canonical checkpoint directory "
                "or a child of it"
            )
        self.base_dir = (
            (self.root / experiment_id) if base_dir is None else Path(base_dir).resolve()
        )
        if self.base_dir != self.root and self.root not in self.base_dir.parents:
            raise OOFArtifactError("OOF artifact base directory must be inside its root")
        if not self.read_only:
            self.base_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_writer = AtomicMetadataWriter(error_type=OOFArtifactError)
        self._immutable_writer = ImmutableArtifactWriter(error_type=OOFArtifactError)
        self.manager.validate_membership()

    def _require_writable(self) -> None:
        if self.read_only:
            raise OOFArtifactError("this OOF artifact store is read-only")

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
        self._require_writable()
        self._immutable_writer.write_text_once(
            self.manifest_path, self.manager.manifest().to_json()
        )
        return self.manifest_path

    def validate_run_provenance(
        self,
        context: OOFRunContext,
        *,
        expected_config: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Read and validate immutable run provenance without writing files."""
        self._validate_context(context)
        run_dir = self.run_dir(context).resolve()
        if not run_dir.is_dir():
            raise OOFArtifactMissingError(f"OOF run directory is missing: {run_dir}")

        metadata = self._read_metadata(context)
        if metadata.get("experiment_id") != self.experiment_id:
            raise OOFArtifactError("OOF run metadata has a mismatched experiment ID")
        if metadata.get("status") not in {"prepared", "training_complete", "complete"}:
            raise OOFArtifactError(
                f"OOF run metadata has an unsupported status {metadata.get('status')!r}"
            )

        manifest_path = self.manifest_path
        if not manifest_path.exists():
            raise OOFArtifactMissingError(f"OOF fold manifest is missing: {manifest_path}")
        try:
            manifest = FoldManifest.from_json(manifest_path.read_text())
            manifest.validate()
        except (OSError, OOFProtocolError) as exc:
            raise OOFArtifactError(f"OOF fold manifest is invalid: {manifest_path}") from exc
        if manifest.to_dict() != self.manager.manifest().to_dict():
            raise OOFArtifactError(
                "OOF fold manifest disagrees with the current frozen fold manager"
            )

        resolved_config = self._read_resolved_config(context)
        if expected_config is not None and _canonical_json(resolved_config) != _canonical_json(
            dict(expected_config)
        ):
            raise OOFArtifactError(
                "completed OOF run has a resolved configuration different from the "
                "requested frozen configuration"
            )

        expected_static = self._static_metadata(context, resolved_config)
        for key, value in expected_static.items():
            if key == "status":
                continue
            if metadata.get(key) != value:
                raise OOFArtifactError(
                    f"OOF metadata disagrees for {key!r}"
                )
        return metadata, resolved_config

    def validate_final_checkpoint(
        self,
        context: OOFRunContext,
        *,
        metadata: Mapping[str, Any] | None = None,
        resolved_config: Mapping[str, Any] | None = None,
        require_record: bool = False,
    ) -> tuple[Path, dict[str, Any], str]:
        """Validate the expected final checkpoint without changing metadata.

        When ``require_record`` is false, a final checkpoint at the canonical
        run-relative path may be validated before an interrupted metadata update
        records it.  This is the only checkpoint that prediction recovery may
        adopt; no canonical full-data checkpoint is considered.
        """
        self._validate_context(context)
        if metadata is None:
            metadata = self._read_metadata(context)
        if resolved_config is None:
            resolved_config = self._read_resolved_config(context)

        checkpoint_record = metadata.get("checkpoint")
        expected_path = self.checkpoint_path(context).resolve()
        if checkpoint_record is None:
            if require_record:
                raise OOFArtifactError("completed OOF run has no checkpoint record")
            checkpoint_path = expected_path
        elif not isinstance(checkpoint_record, Mapping):
            raise OOFArtifactError("OOF checkpoint metadata is malformed")
        else:
            recorded_value = checkpoint_record.get("path")
            expected_relative = self.checkpoint_path(context).relative_to(
                self.run_dir(context)
            )
            if recorded_value != str(expected_relative):
                raise OOFArtifactError(
                    "OOF checkpoint metadata does not reference the expected final "
                    "checkpoint path"
                )
            checkpoint_path = self._recorded_path(
                self.run_dir(context).resolve(),
                recorded_value,
                name="checkpoint",
            )

        state, actual_sha = self._validate_checkpoint_file(
            context,
            checkpoint_path,
            resolved_config,
        )
        if isinstance(checkpoint_record, Mapping):
            if checkpoint_record.get("sha256") != actual_sha:
                raise OOFArtifactError(
                    "OOF checkpoint hash does not match its bytes"
                )
            if checkpoint_record.get("epoch") != int(state["epoch"]):
                raise OOFArtifactError(
                    "OOF checkpoint epoch disagrees with metadata"
                )
        return checkpoint_path, state, actual_sha

    def prepare_run(
        self,
        context: OOFRunContext,
        resolved_config: Mapping[str, Any],
    ) -> Path:
        """Create or verify an exact run directory before model construction."""
        self._require_writable()
        self._validate_context(context)
        self.write_manifest()
        run_dir = self.run_dir(context)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        config_json = _canonical_json(dict(resolved_config))
        self._immutable_writer.write_text_once(
            self.config_path(context), config_json + "\n"
        )

        metadata = self._static_metadata(context, dict(resolved_config))
        metadata_path = self.metadata_path(context)
        if metadata_path.exists():
            try:
                existing = json.loads(metadata_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise OOFArtifactError(
                    f"cannot read existing OOF run metadata: {metadata_path}"
                ) from exc
            if not isinstance(existing, Mapping):
                raise OOFArtifactError(
                    f"existing OOF run metadata must contain a mapping: {metadata_path}"
                )
            for key, value in metadata.items():
                if key == "status":
                    continue
                if existing.get(key) != value:
                    raise OOFArtifactError(
                        f"existing run metadata disagrees for {key!r}; refusing to "
                        "mix incompatible configurations or fold memberships"
                    )
        else:
            self._metadata_writer.write(metadata_path, metadata)
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
        self._require_writable()
        self._validate_context(context)
        path = Path(checkpoint_path).resolve()
        run_dir = self.run_dir(context).resolve()
        if run_dir not in path.parents:
            raise OOFArtifactError(
                "OOF checkpoint must be written inside its dedicated run directory"
            )
        if path != self.checkpoint_path(context).resolve():
            raise OOFArtifactError(
                "OOF checkpoint must use the expected final checkpoint path"
            )
        resolved_config = self._read_resolved_config(context)
        _state, actual_sha = self._validate_checkpoint_file(
            context,
            path,
            resolved_config,
        )
        if checkpoint_sha256 is not None and checkpoint_sha256 != actual_sha:
            raise OOFArtifactError("checkpoint SHA-256 does not match its contents")
        metadata = self._read_metadata(context)

        existing = metadata.get("checkpoint")
        checkpoint_record = {
            "path": str(path.relative_to(run_dir)),
            "sha256": actual_sha,
            "epoch": int(_state["epoch"]),
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
        self._require_writable()
        self._validate_context(context)
        artifact.validate(self.manager, require_complete=False)
        metadata = self._read_metadata(context)
        checkpoint = metadata.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise OOFArtifactError("cannot write predictions before checkpoint validation")
        resolved_config = self._read_resolved_config(context)
        _checkpoint_path, _checkpoint_state, checkpoint_sha = self.validate_final_checkpoint(
            context,
            metadata=metadata,
            resolved_config=resolved_config,
            require_record=True,
        )
        if checkpoint.get("sha256") != checkpoint_sha:
            raise OOFArtifactError("prediction artifact has a mismatched checkpoint hash")
        self._validate_run_artifact(context, artifact, checkpoint)
        serialized = artifact.to_json()
        path = self.prediction_path(context)
        self._immutable_writer.write_text_once(path, serialized)
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
        self._validate_context(context)
        path = self.prediction_path(context)
        if not path.exists():
            raise OOFArtifactMissingError(f"OOF prediction artifact is missing: {path}")
        try:
            artifact = OOFPredictionArtifact.from_json(path.read_text())
        except OOFArtifactValidationError as exc:
            raise OOFArtifactError(f"invalid OOF prediction artifact: {exc}") from exc
        metadata = self._read_metadata(context)
        resolved_config = self._read_resolved_config(context)
        _checkpoint_path, _checkpoint_state, checkpoint_sha = self.validate_final_checkpoint(
            context,
            metadata=metadata,
            resolved_config=resolved_config,
            require_record=True,
        )
        checkpoint = metadata["checkpoint"]
        if checkpoint.get("sha256") != checkpoint_sha:
            raise OOFArtifactError("prediction artifact has a mismatched checkpoint hash")
        prediction_record = metadata.get("prediction")
        if prediction_record is not None:
            if not isinstance(prediction_record, Mapping):
                raise OOFArtifactError("prediction metadata is malformed")
            run_dir = self.run_dir(context).resolve()
            recorded_path = self._recorded_path(
                run_dir, prediction_record.get("path"), name="prediction"
            )
            if recorded_path != path.resolve():
                raise OOFArtifactError(
                    "prediction metadata points to a different artifact path"
                )
            if prediction_record.get("sha256") != _sha256_file(path):
                raise OOFArtifactError("prediction metadata hash does not match its bytes")
        artifact.validate(self.manager, require_complete=False)
        self._validate_run_artifact(context, artifact, checkpoint)
        if isinstance(prediction_record, Mapping) and prediction_record.get(
            "record_count"
        ) != len(artifact.records):
            raise OOFArtifactError("prediction metadata record count does not match artifact")
        return artifact

    def validate_completed_run(
        self,
        context: OOFRunContext,
        *,
        expected_config: Mapping[str, Any] | None = None,
    ) -> "OOFCompletedRun":
        """Validate a completed run without changing any artifact file.

        ``load_predictions`` validates the prediction schema and its linkage to
        run metadata.  Batch execution also needs a read-only completeness
        check that verifies the config file, checkpoint bytes, checkpoint
        metadata, prediction bytes, and the recorded status.  Keeping that
        check here gives the batch runner a safe way to skip work after a
        Kaggle session restart and lets it validate the Task 3B pilot without
        rewriting its metadata.
        """
        metadata, resolved_config = self.validate_run_provenance(
            context,
            expected_config=expected_config,
        )
        run_dir = self.run_dir(context).resolve()
        if metadata.get("status") != "complete":
            raise OOFArtifactError(
                "OOF run is not complete; recorded status is "
                f"{metadata.get('status')!r}"
            )

        checkpoint_record = metadata.get("checkpoint")
        if not isinstance(checkpoint_record, Mapping):
            raise OOFArtifactError("completed OOF run has no checkpoint record")
        checkpoint_path, _checkpoint_state, _actual_checkpoint_sha = (
            self.validate_final_checkpoint(
                context,
                metadata=metadata,
                resolved_config=resolved_config,
                require_record=True,
            )
        )

        prediction_record = metadata.get("prediction")
        if not isinstance(prediction_record, Mapping):
            raise OOFArtifactError("completed OOF run has no prediction record")
        prediction_path = self._recorded_path(
            run_dir, prediction_record.get("path"), name="prediction"
        )
        actual_prediction_sha = _sha256_file(prediction_path)
        if prediction_record.get("sha256") != actual_prediction_sha:
            raise OOFArtifactError("completed OOF prediction hash does not match its bytes")
        try:
            artifact = OOFPredictionArtifact.from_json(prediction_path.read_text())
        except (OSError, OOFArtifactValidationError) as exc:
            raise OOFArtifactError(
                f"completed OOF prediction artifact is invalid: {prediction_path}"
            ) from exc
        artifact.validate(self.manager, require_complete=False)
        self._validate_run_artifact(context, artifact, checkpoint_record)
        if prediction_record.get("record_count") != len(artifact.records):
            raise OOFArtifactError("prediction record count disagrees with metadata")

        return OOFCompletedRun(
            context=context,
            run_dir=run_dir,
            resolved_config=resolved_config,
            metadata=metadata,
            checkpoint_path=checkpoint_path,
            prediction_path=prediction_path,
            artifact=artifact,
        )

    @staticmethod
    def _recorded_path(run_dir: Path, value: Any, *, name: str) -> Path:
        if not isinstance(value, str) or not value:
            raise OOFArtifactError(f"completed OOF {name} path is missing")
        path = Path(value)
        if path.is_absolute():
            raise OOFArtifactError(f"completed OOF {name} path must be relative")
        resolved = (run_dir / path).resolve()
        if run_dir not in resolved.parents:
            raise OOFArtifactError(f"completed OOF {name} path escapes its run directory")
        if not resolved.is_file():
            raise OOFArtifactMissingError(
                f"completed OOF {name} file is missing: {resolved}"
            )
        return resolved

    def _read_resolved_config(self, context: OOFRunContext) -> dict[str, Any]:
        config_path = self.config_path(context)
        if not config_path.exists():
            raise OOFArtifactMissingError(f"OOF resolved config is missing: {config_path}")
        try:
            resolved_config = json.loads(config_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise OOFArtifactError(f"OOF resolved config is invalid: {config_path}") from exc
        if not isinstance(resolved_config, Mapping):
            raise OOFArtifactError("OOF resolved config must contain a mapping")
        return dict(resolved_config)

    def _validate_checkpoint_file(
        self,
        context: OOFRunContext,
        path: Path,
        resolved_config: Mapping[str, Any],
    ) -> tuple[dict[str, Any], str]:
        expected_path = self.checkpoint_path(context).resolve()
        path = path.resolve()
        if path != expected_path:
            raise OOFArtifactError(
                "OOF checkpoint must use the expected final checkpoint path"
            )
        if not path.is_file():
            raise OOFArtifactMissingError(f"OOF checkpoint is missing: {path}")
        actual_sha = _sha256_file(path)
        expected_epoch = None
        schedule = resolved_config.get("schedule")
        if isinstance(schedule, Mapping) and "epochs" in schedule:
            try:
                expected_epoch = int(schedule["epochs"])
            except (TypeError, ValueError) as exc:
                raise OOFArtifactError(
                    "OOF resolved config contains an invalid schedule epoch"
                ) from exc
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
            self._validate_model_state_compatibility(
                state,
                path=path,
                resolved_config=resolved_config,
            )
        except OOFArtifactError:
            raise
        except Exception as exc:  # noqa: BLE001 - corrupt torch files are untrusted input
            raise OOFArtifactError(
                f"OOF checkpoint provenance validation failed: {exc}"
            ) from exc
        return state, actual_sha

    @staticmethod
    def _validate_model_state_compatibility(
        state: Mapping[str, Any],
        *,
        path: Path,
        resolved_config: Mapping[str, Any],
    ) -> None:
        """Check the frozen ResNet-32 state structure without touching data."""
        model_config = resolved_config.get("model")
        if not isinstance(model_config, Mapping):
            return
        if model_config.get("arch") != "resnet32" or model_config.get("num_classes") != 100:
            return
        from models.resnet32 import ResNet32

        try:
            model = ResNet32(num_classes=100)
            model.load_state_dict(state["model_state_dict"], strict=True)
        except Exception as exc:  # noqa: BLE001 - state structure is untrusted input
            raise OOFArtifactError(
                f"checkpoint model state is incompatible with ResNet-32/100 at {path}: {exc}"
            ) from exc

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
        if not isinstance(payload, Mapping):
            raise OOFArtifactError(f"OOF run metadata must contain a mapping: {path}")
        if payload.get("schema_version") != OOF_RUN_SCHEMA_VERSION:
            raise OOFArtifactError("unsupported OOF run metadata schema")
        return payload

    def _write_metadata(self, context: OOFRunContext, metadata: Mapping[str, Any]) -> None:
        self._require_writable()
        self._metadata_writer.write(self.metadata_path(context), metadata)

    def _append_log(self, context: OOFRunContext, message: str) -> None:
        self._require_writable()
        with self.log_path(context).open("a") as handle:
            handle.write(message)


@dataclass(frozen=True)
class OOFTrainingResult:
    trainer: Any
    checkpoint_path: Path
    checkpoint_sha256: str
    history: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class OOFCompletedRun:
    """Read-only view of a fully validated single-run OOF artifact."""

    context: OOFRunContext
    run_dir: Path
    resolved_config: Mapping[str, Any]
    metadata: Mapping[str, Any]
    checkpoint_path: Path
    prediction_path: Path
    artifact: OOFPredictionArtifact


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
        metrics_sink: Callable[[Mapping[str, Any]], None] | None = None,
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
            # A Kaggle session can stop after ``torch.save`` and before the
            # metadata update.  The exact run path plus the checkpoint's own
            # expert/seed/final-epoch metadata is sufficient to validate and
            # adopt that checkpoint; a corrupt or incompatible file still
            # fails in ``record_checkpoint`` and is never silently reused.
            if checkpoint is None or isinstance(checkpoint, Mapping):
                checkpoint_sha = store.record_checkpoint(context, checkpoint_path)
                build_kwargs: dict[str, Any] = {
                    "class_counts": class_counts,
                    "device": config.resolved_device,
                }
                if metrics_sink is not None:
                    build_kwargs["metrics_sink"] = metrics_sink
                trainer = self.trainer_builder(config, **build_kwargs)
                trainer.load_checkpoint(checkpoint_path)
                return OOFTrainingResult(
                    trainer=trainer,
                    checkpoint_path=checkpoint_path,
                    checkpoint_sha256=checkpoint_sha,
                    history=tuple(),
                )
            raise OOFArtifactError("existing OOF checkpoint metadata is malformed")

        # Membership validation above happens before this registry call, so an
        # accidental full-data loader cannot silently construct a model.
        build_kwargs = {
            "class_counts": class_counts,
            "device": config.resolved_device,
        }
        if metrics_sink is not None:
            build_kwargs["metrics_sink"] = metrics_sink
        trainer = self.trainer_builder(config, **build_kwargs)
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
        provenance_checkpoint_path: str | Path | None = None,
        checkpoint_sha256: str,
        resolved_config: Mapping[str, Any],
    ) -> OOFPredictionArtifact:
        if context.manager is not self.manager:
            raise OOFArtifactError("prediction context belongs to a different manager")
        self.data_module.validate_context(context)
        path = Path(checkpoint_path).resolve()
        record_checkpoint_path = str(
            Path(provenance_checkpoint_path).resolve()
            if provenance_checkpoint_path is not None
            else path
        )
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
                            checkpoint_path=record_checkpoint_path,
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

    def run(
        self,
        spec: OOFRunSpec,
        base_config: TrainingConfig,
        *,
        metrics_sink: Callable[[Mapping[str, Any]], None] | None = None,
        phase_callback: Callable[[str], None] | None = None,
        provenance_run_dir: str | Path | None = None,
    ) -> OOFRunResult:
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
        if phase_callback is not None:
            phase_callback("preparing_data")
        config = base_config.replace(
            seed=context.training_seed,
            device=spec.device,
            epochs=spec.epochs,
            checkpoint_dir=str(self.store.run_dir(context) / "checkpoints"),
        )
        provenance_dir = (
            self.store.run_dir(context)
            if provenance_run_dir is None
            else Path(provenance_run_dir)
        )
        provenance_config = base_config.replace(
            seed=context.training_seed,
            device=spec.device,
            epochs=spec.epochs,
            checkpoint_dir=str(provenance_dir / "checkpoints"),
        )
        resolved_config = provenance_config.to_dict()
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
        if phase_callback is not None:
            phase_callback("training")
        training = self.training.train(
            context=context,
            config=config,
            data_module=data_module,
            store=self.store,
            resolved_config=resolved_config,
            metrics_sink=metrics_sink,
        )
        if phase_callback is not None:
            phase_callback("predicting")
        artifact = OOFPredictionCollector(
            manager=self.manager, data_module=data_module
        ).collect(
            context=context,
            trainer=training.trainer,
            checkpoint_path=training.checkpoint_path,
            provenance_checkpoint_path=(provenance_dir / "checkpoints" / training.checkpoint_path.name),
            checkpoint_sha256=training.checkpoint_sha256,
            resolved_config=resolved_config,
        )
        prediction_path = self.store.write_predictions(context, artifact)
        if phase_callback is not None:
            phase_callback("writing_predictions")
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
