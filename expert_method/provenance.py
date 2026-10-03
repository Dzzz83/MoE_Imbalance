"""Immutable provenance records for evaluations of frozen study artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import time
from typing import Any, Mapping


_EVALUATION_SCHEMA = "expert_method.evaluation_run.v1"
_EVALUATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_STAGES = {"evaluate": "outputs", "report": "report"}


def canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def _require_digest(value: Any, name: str, *, lengths: set[int] = {64}) -> str:
    if not isinstance(value, str) or len(value) not in lengths:
        raise ValueError(f"invalid {name} in frozen study identity")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"invalid {name} in frozen study identity") from exc
    return value.lower()


@dataclass(frozen=True)
class StudyTrainingIdentity:
    """Training identity copied from a validated immutable study freeze."""

    study_id: str
    source_commit: str
    freeze_sha256: str
    plan_sha256: str
    study_config_sha256: str
    study_config_yaml_sha256: str
    protocol_config_sha256: str
    fold_manifest_sha256: str
    job_inventory_sha256: str

    @classmethod
    def from_freeze(cls, freeze: Mapping[str, Any]) -> "StudyTrainingIdentity":
        if freeze.get("source_tree_dirty") is not False:
            raise ValueError("frozen training source was recorded as dirty")
        return cls(
            study_id=str(freeze["study_id"]),
            source_commit=_require_digest(freeze.get("source_commit"), "training commit", lengths={40, 64}),
            freeze_sha256=_require_digest(freeze.get("freeze_sha256"), "freeze SHA-256"),
            plan_sha256=_require_digest(freeze.get("plan_sha256"), "plan SHA-256"),
            study_config_sha256=_require_digest(freeze.get("study_config_sha256"), "study config SHA-256"),
            study_config_yaml_sha256=_require_digest(freeze.get("study_config_yaml_sha256"), "study YAML SHA-256"),
            protocol_config_sha256=_require_digest(freeze.get("protocol_config_sha256"), "protocol config SHA-256"),
            fold_manifest_sha256=_require_digest(freeze.get("fold_manifest_sha256"), "fold manifest SHA-256"),
            job_inventory_sha256=_require_digest(freeze.get("job_inventory_sha256"), "job inventory SHA-256"),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "study_id": self.study_id,
            "training_commit": self.source_commit,
            "freeze_sha256": self.freeze_sha256,
            "plan_sha256": self.plan_sha256,
            "study_config_sha256": self.study_config_sha256,
            "study_config_yaml_sha256": self.study_config_yaml_sha256,
            "protocol_config_sha256": self.protocol_config_sha256,
            "fold_manifest_sha256": self.fold_manifest_sha256,
            "job_inventory_sha256": self.job_inventory_sha256,
        }


class EvaluationRunStore:
    """Create append-only per-run records and atomically publish stage output."""

    def __init__(self, study_root: str | Path, evaluation_id: str) -> None:
        if not isinstance(evaluation_id, str) or not _EVALUATION_ID.fullmatch(evaluation_id):
            raise ValueError("evaluation ID must be 1-64 letters, digits, dots, underscores, or hyphens")
        if evaluation_id in {".", ".."}:
            raise ValueError("evaluation ID is unsafe")
        self.evaluation_id = evaluation_id
        self.study_root = Path(study_root).expanduser().absolute()
        self.runs_root = self.study_root / "evaluation_runs"
        self.run_dir = self.runs_root / evaluation_id

    def begin(self, stage: str, record: Mapping[str, Any]) -> Path:
        if stage not in _STAGES:
            raise ValueError(f"unsupported evaluation run stage: {stage}")
        self._require_safe_roots(require_run=stage == "report")
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._require_safe_roots(require_run=stage == "report")
        if stage == "evaluate":
            try:
                self.run_dir.mkdir()
            except FileExistsError as exc:
                raise ValueError(f"evaluation ID already exists and cannot be overwritten: {self.evaluation_id}") from exc
        elif not self.run_dir.is_dir() or self.run_dir.is_symlink():
            raise ValueError(f"evaluation run does not exist: {self.evaluation_id}")
        self._require_safe_roots(require_run=True)
        start_path = self.run_dir / f"{stage}.start.json"
        self._write_new(start_path, {"schema_version": _EVALUATION_SCHEMA, "stage": stage, **dict(record)})
        return start_path

    def stage_directory(self, stage: str) -> Path:
        self._require_safe_roots(require_run=True)
        if stage not in _STAGES or not self.run_dir.is_dir() or self.run_dir.is_symlink():
            raise ValueError("cannot stage output outside a started evaluation run")
        return Path(tempfile.mkdtemp(prefix=f".{stage}-staging-", dir=self.run_dir))

    def publish(
        self,
        stage: str,
        staged_directory: str | Path,
        *,
        expected_start_sha256: str,
    ) -> Path:
        self._require_safe_roots(require_run=True)
        if stage not in _STAGES:
            raise ValueError(f"unsupported evaluation run stage: {stage}")
        self.assert_start_unchanged(stage, expected_start_sha256)
        staged = Path(staged_directory).absolute()
        if staged.parent != self.run_dir or not staged.is_dir() or staged.is_symlink():
            raise ValueError("staged evaluation output is outside its run directory")
        output = self.run_dir / _STAGES[stage]
        if output.exists() or output.is_symlink():
            raise ValueError(f"evaluation output already exists: {output}")
        os.rename(staged, output)
        self._fsync_directory(self.run_dir)
        return output

    def complete(
        self,
        stage: str,
        record: Mapping[str, Any],
        output_dir: str | Path,
        *,
        started_monotonic: float,
        expected_start_sha256: str,
    ) -> Path:
        self._require_safe_roots(require_run=True)
        if stage not in _STAGES:
            raise ValueError(f"unsupported evaluation run stage: {stage}")
        output = Path(output_dir).absolute()
        expected_output = self.run_dir / _STAGES[stage]
        if output != expected_output or not output.is_dir() or output.is_symlink():
            raise ValueError("published evaluation output has an unexpected location")
        _start, start_bytes = self._read_start_snapshot(stage)
        start_sha256 = hashlib.sha256(start_bytes).hexdigest()
        if start_sha256 != expected_start_sha256:
            raise ValueError(f"{stage} start record changed during the command")
        self.validate_completion_record(
            stage, record, expected_start_sha256=expected_start_sha256
        )
        provenance = dict(record)
        provenance["finished_at"] = datetime.now(timezone.utc).isoformat()
        provenance["duration_seconds"] = max(0.0, time.monotonic() - started_monotonic)
        payload: dict[str, Any] = {
            "schema_version": _EVALUATION_SCHEMA,
            "stage": stage,
            "evaluation_id": self.evaluation_id,
            "status": "succeeded",
            "output_directory": _STAGES[stage],
            "output_files": self.hash_tree(output),
            "start_record_sha256": start_sha256,
            "provenance": provenance,
        }
        payload["record_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        self.assert_start_unchanged(stage, expected_start_sha256)
        complete_path = self.run_dir / f"{stage}.complete.json"
        self._write_new(complete_path, payload)
        return complete_path

    def fail(self, stage: str, *, error_type: str, message: str, finished_at: str) -> None:
        if stage not in _STAGES:
            return
        try:
            self._require_safe_roots(require_run=True)
        except (OSError, ValueError):
            return
        complete_path = self.run_dir / f"{stage}.complete.json"
        failure_path = self.run_dir / f"{stage}.failed.json"
        if complete_path.exists() or failure_path.exists():
            return
        self._write_new(failure_path, {
            "schema_version": _EVALUATION_SCHEMA,
            "stage": stage,
            "evaluation_id": self.evaluation_id,
            "status": "failed",
            "error_type": error_type,
            "message": message,
            "finished_at": finished_at,
        })

    def read_complete(self, stage: str) -> dict[str, Any]:
        self._require_safe_roots(require_run=True)
        if stage not in _STAGES or not self.run_dir.is_dir() or self.run_dir.is_symlink():
            raise ValueError("evaluation run directory is missing or unsafe")
        path = self.run_dir / f"{stage}.complete.json"
        if path.is_symlink():
            raise ValueError(f"{stage} completion record cannot be a symlink")
        try:
            raw = path.read_bytes()
            payload = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"successful {stage} completion record is missing or invalid") from exc
        start, start_bytes = self._read_start_snapshot(stage)
        recorded_hash = payload.get("record_sha256") if isinstance(payload, dict) else None
        unhashed = dict(payload) if isinstance(payload, dict) else {}
        unhashed.pop("record_sha256", None)
        provenance = payload.get("provenance") if isinstance(payload, dict) else None
        if (
            not isinstance(payload, dict)
            or raw != canonical_json_bytes(payload)
            or payload.get("schema_version") != _EVALUATION_SCHEMA
            or payload.get("stage") != stage
            or payload.get("evaluation_id") != self.evaluation_id
            or payload.get("status") != "succeeded"
            or payload.get("output_directory") != _STAGES[stage]
            or not isinstance(provenance, dict)
            or provenance.get("evaluation_id") != self.evaluation_id
            or not isinstance(provenance.get("study_id"), str)
            or not provenance.get("study_id")
            or recorded_hash != hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest()
            or payload.get("start_record_sha256") != hashlib.sha256(start_bytes).hexdigest()
            or any(provenance.get(field) != start.get(field) for field in (
                "study_id", "training_identity", "evaluator_identity", "scientific_settings",
                "command", "python_executable", "working_directory", "configuration_inputs",
                "runtime_resolution", "runtime_versions",
            ))
            or not isinstance(provenance.get("input_artifacts"), Mapping)
            or len(provenance.get("input_artifacts", {})) != 300
            or not isinstance(provenance.get("input_manifests"), Mapping)
            or not provenance.get("input_manifests")
            or not isinstance(provenance.get("fold_memberships"), Mapping)
            or len(provenance.get("fold_memberships", {})) != 15
            or not isinstance(provenance.get("training_population"), Mapping)
            or not isinstance(provenance.get("started_at"), str)
            or not isinstance(provenance.get("finished_at"), str)
            or not isinstance(provenance.get("duration_seconds"), (int, float))
            or not math.isfinite(float(provenance.get("duration_seconds", -1)))
            or float(provenance.get("duration_seconds", -1)) < 0
            or not isinstance(provenance.get("runtime_versions"), Mapping)
            or not provenance.get("runtime_versions")
            or not isinstance(provenance.get("python_executable"), str)
            or not provenance.get("python_executable")
            or not isinstance(provenance.get("working_directory"), str)
            or not provenance.get("working_directory")
            or (
                stage == "report"
                and (
                    not isinstance(provenance.get("input_evaluation_outputs"), Mapping)
                    or len(provenance.get("input_evaluation_outputs", {})) != 30
                    or not isinstance(provenance.get("input_evaluation_records"), Mapping)
                    or len(provenance.get("input_evaluation_records", {})) != 2
                )
            )
        ):
            raise ValueError(f"successful {stage} completion record has an invalid identity")
        output = self.run_dir / _STAGES[stage]
        if not output.is_dir() or output.is_symlink() or self.hash_tree(output) != payload.get("output_files"):
            raise ValueError(f"{stage} outputs differ from their immutable completion record")
        return payload

    def sidecar_hashes(self, stage: str) -> dict[str, str]:
        if stage not in _STAGES:
            raise ValueError(f"unsupported evaluation run stage: {stage}")
        result: dict[str, str] = {}
        for name in (f"{stage}.start.json", f"{stage}.complete.json"):
            path = self.run_dir / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"evaluation provenance sidecar is missing or unsafe: {name}")
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    def assert_start_unchanged(self, stage: str, expected_sha256: str) -> None:
        _, raw = self._read_start_snapshot(stage)
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError(f"{stage} start record changed during the command")

    def require_safe_run(self) -> None:
        """Require that cleanup or writes still target this run's original path."""
        self._require_safe_roots(require_run=True)

    def validate_completion_record(
        self,
        stage: str,
        record: Mapping[str, Any],
        *,
        expected_start_sha256: str,
    ) -> None:
        """Validate complete provenance against the exact pinned start record."""
        start, raw = self._read_start_snapshot(stage)
        if hashlib.sha256(raw).hexdigest() != expected_start_sha256:
            raise ValueError(f"{stage} start record changed during the command")
        if record.get("evaluation_id") != self.evaluation_id:
            raise ValueError(f"{stage} completion changed its evaluation ID")
        for field in (
            "study_id", "training_identity", "evaluator_identity", "scientific_settings",
            "command", "python_executable", "working_directory", "configuration_inputs",
            "runtime_resolution", "runtime_versions",
        ):
            if field not in start or record.get(field) != start.get(field):
                raise ValueError(f"{stage} completion changed its start-time {field}")
        if (
            not isinstance(record.get("input_artifacts"), Mapping)
            or len(record["input_artifacts"]) != 300
            or not isinstance(record.get("input_manifests"), Mapping)
            or not record["input_manifests"]
            or not isinstance(record.get("fold_memberships"), Mapping)
            or len(record["fold_memberships"]) != 15
            or not isinstance(record.get("training_population"), Mapping)
            or not isinstance(record.get("started_at"), str)
            or not isinstance(record.get("python_executable"), str)
            or not record["python_executable"]
            or not isinstance(record.get("working_directory"), str)
            or not record["working_directory"]
            or not isinstance(record.get("runtime_versions"), Mapping)
            or not record["runtime_versions"]
        ):
            raise ValueError(f"{stage} completion record is missing required study provenance")
        if stage == "report" and (
            not isinstance(record.get("input_evaluation_outputs"), Mapping)
            or len(record["input_evaluation_outputs"]) != 30
            or not isinstance(record.get("input_evaluation_records"), Mapping)
            or len(record["input_evaluation_records"]) != 2
        ):
            raise ValueError("report completion is missing its successful evaluation inputs")

    @staticmethod
    def hash_tree(root: Path) -> dict[str, dict[str, Any]]:
        files: dict[str, dict[str, Any]] = {}
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"evaluation output contains a symlink: {path}")
            if path.is_file():
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                files[path.relative_to(root).as_posix()] = {
                    "sha256": digest.hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
            elif not path.is_dir():
                raise ValueError(f"evaluation output contains an unsupported filesystem entry: {path}")
        if not files:
            raise ValueError("evaluation stage produced no files")
        return files

    @staticmethod
    def _write_new(path: Path, record: Mapping[str, Any]) -> None:
        payload = dict(record)
        if path.name.endswith(".start.json"):
            payload["record_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
        encoded = canonical_json_bytes(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            EvaluationRunStore._fsync_directory(path.parent)
        except BaseException:
            try:
                path.unlink()
            except OSError:
                pass
            raise

    def _read_start(self, stage: str) -> dict[str, Any]:
        return self._read_start_snapshot(stage)[0]

    def _read_start_snapshot(self, stage: str) -> tuple[dict[str, Any], bytes]:
        path = self.run_dir / f"{stage}.start.json"
        descriptor = -1
        try:
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError(f"{stage} start record is not a regular file")
            with os.fdopen(descriptor, "rb") as handle:
                descriptor = -1
                raw = handle.read()
            payload = json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"{stage} start record is missing or invalid") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
        if not isinstance(payload, dict) or raw != canonical_json_bytes(payload):
            raise ValueError(f"{stage} start record is not canonical JSON")
        recorded_hash = payload.get("record_sha256")
        unhashed = dict(payload)
        unhashed.pop("record_sha256", None)
        if (
            recorded_hash != hashlib.sha256(canonical_json_bytes(unhashed)).hexdigest()
            or payload.get("schema_version") != _EVALUATION_SCHEMA
            or payload.get("stage") != stage
            or payload.get("evaluation_id") != self.evaluation_id
            or not isinstance(payload.get("training_identity"), Mapping)
            or not isinstance(payload.get("evaluator_identity"), Mapping)
            or not isinstance(payload.get("scientific_settings"), Mapping)
            or not isinstance(payload.get("configuration_inputs"), Mapping)
            or not isinstance(payload.get("command"), list)
            or not isinstance(payload.get("python_executable"), str)
            or not payload.get("python_executable")
            or not isinstance(payload.get("working_directory"), str)
            or not payload.get("working_directory")
            or not isinstance(payload.get("started_at"), str)
        ):
            raise ValueError(f"{stage} start record has an invalid identity")
        return payload, raw

    def _require_safe_roots(self, *, require_run: bool) -> None:
        if self.study_root.is_symlink() or self.runs_root.is_symlink():
            raise ValueError("evaluation output root cannot be a symlink")
        if self.study_root.resolve(strict=False) != self.study_root:
            raise ValueError("evaluation output root resolves through a symlink")
        if self.runs_root.resolve(strict=False) != self.runs_root:
            raise ValueError("evaluation runs directory resolves through a symlink")
        if require_run and (
            not self.run_dir.is_dir()
            or self.run_dir.is_symlink()
            or self.run_dir.resolve() != self.run_dir
        ):
            raise ValueError("evaluation run directory is missing or unsafe")

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


__all__ = ["EvaluationRunStore", "StudyTrainingIdentity", "canonical_json_bytes"]
