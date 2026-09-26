"""Durable per-job attempt workspaces for interrupted OOF training runs."""

from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import threading
import traceback
from typing import Any, Iterator, Mapping, TextIO
import uuid


ATTEMPT_SCHEMA = "expert_method.job_attempt.v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot serialize metrics value {type(value).__name__}")


@dataclass
class AttemptWorkspace:
    """One unique, retained execution workspace for a canonical job."""

    path: Path
    attempt_id: str
    study_id: str
    job_id: str
    stage: str
    device: str
    started_at: str

    @classmethod
    def create(
        cls,
        *,
        run_root: str | Path,
        study_id: str,
        job_id: str,
        stage: str,
        expert_key: str,
        training_seed: int,
        outer_fold_id: int,
        inner_fold_id: int | None,
        device: str,
    ) -> "AttemptWorkspace":
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        attempt_id = f"{timestamp}-{uuid.uuid4().hex[:12]}"
        fold_name = "outer_eval" if inner_fold_id is None else f"inner_{inner_fold_id}"
        path = (
            Path(run_root).expanduser() / study_id / "attempts" / stage / expert_key
            / f"seed_{training_seed}" / f"outer_{outer_fold_id}" / fold_name / attempt_id
        )
        path.mkdir(parents=True, exist_ok=False)
        (path / "execution.log").touch(exist_ok=False)
        workspace = cls(
            path=path,
            attempt_id=attempt_id,
            study_id=study_id,
            job_id=job_id,
            stage=stage,
            device=device,
            started_at=utc_now(),
        )
        workspace.update(status="running", phase="initializing")
        return workspace

    @property
    def status_path(self) -> Path:
        return self.path / "attempt.json"

    @property
    def metrics_path(self) -> Path:
        return self.path / "metrics.jsonl"

    @property
    def execution_log_path(self) -> Path:
        return self.path / "execution.log"

    @property
    def resolved_config_path(self) -> Path:
        return self.path / "resolved_config.json"

    def update(
        self,
        *,
        status: str,
        phase: str,
        locations: Mapping[str, str] | None = None,
        failure: Mapping[str, str] | None = None,
        finished: bool = False,
    ) -> None:
        record: dict[str, Any] = {
            "schema_version": ATTEMPT_SCHEMA,
            "study_id": self.study_id,
            "job_id": self.job_id,
            "attempt_id": self.attempt_id,
            "stage": self.stage,
            "status": status,
            "phase": phase,
            "started_at": self.started_at,
            "updated_at": utc_now(),
            "environment": _environment(self.device),
            "locations": {
                "attempt": str(self.path),
                "status": str(self.status_path),
                "execution_log": str(self.execution_log_path),
                "resolved_config": str(self.resolved_config_path),
                "metrics": str(self.metrics_path),
                **dict(locations or {}),
            },
            "failure": dict(failure) if failure is not None else None,
        }
        if finished:
            record["finished_at"] = utc_now()
        _atomic_json(self.status_path, record)

    def record_failure(self, *, phase: str, error: BaseException) -> dict[str, str]:
        failure = {
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
        }
        self.update(status="failed", phase=phase, failure=failure, finished=True)
        self.append_execution(
            f"\n[attempt failed during {phase}] {failure['exception_type']}: {failure['message']}\n"
            f"{failure['traceback']}"
        )
        return failure

    def append_execution(self, text: str) -> None:
        with self.execution_log_path.open("a", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

    def write_resolved_config(self, config: Mapping[str, Any]) -> None:
        _atomic_json(self.resolved_config_path, config)

    def metrics_sink(self, metrics: Mapping[str, Any]) -> None:
        line = json.dumps(dict(metrics), sort_keys=True, separators=(",", ":"), default=_json_default)
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    @contextmanager
    def capture_output(self) -> Iterator[None]:
        with self.execution_log_path.open("a", encoding="utf-8", buffering=1) as handle:
            lock = threading.Lock()
            stdout_tee = _TeeTextIO(sys.stdout, handle, lock)
            stderr_tee = _TeeTextIO(sys.stderr, handle, lock)
            with redirect_stdout(stdout_tee), redirect_stderr(stderr_tee):
                yield


class _TeeTextIO:
    """Write training output to the terminal and the durable attempt log."""

    def __init__(self, terminal: TextIO, log: TextIO, lock: threading.Lock) -> None:
        self.terminal = terminal
        self.log = log
        self._lock = lock

    @property
    def encoding(self) -> str:
        return getattr(self.terminal, "encoding", None) or "utf-8"

    def write(self, value: str) -> int:
        with self._lock:
            written = self.terminal.write(value)
            self.terminal.flush()
            self.log.write(value)
            self.log.flush()
            return written

    def flush(self) -> None:
        with self._lock:
            self.terminal.flush()
            self.log.flush()

    def isatty(self) -> bool:
        return bool(getattr(self.terminal, "isatty", lambda: False)())


def summarize_attempts(run_root: str | Path, study_id: str) -> dict[str, Any]:
    """Count retained running/failed attempts without reading canonical jobs."""
    base = Path(run_root).expanduser() / study_id / "attempts"
    records: list[dict[str, str]] = []
    if base.is_dir() and not base.is_symlink():
        for path in sorted(base.rglob("attempt.json")):
            if path.is_symlink() or not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, Mapping) or payload.get("schema_version") != ATTEMPT_SCHEMA:
                continue
            records.append({
                "attempt_id": str(payload.get("attempt_id", "")),
                "job_id": str(payload.get("job_id", "")),
                "status": str(payload.get("status", "unknown")),
                "phase": str(payload.get("phase", "unknown")),
                "path": str(path.parent),
            })
    return {
        "running": sum(row["status"] == "running" for row in records),
        "failed": sum(row["status"] == "failed" for row in records),
        "succeeded": sum(row["status"] == "succeeded" for row in records),
        "locations": records,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(dict(payload), sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _fsync_dir(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _environment(device: str) -> dict[str, Any]:
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        torch_version = torch.__version__
    except Exception as exc:  # diagnostic metadata must not mask the training result
        cuda_available = False
        torch_version = f"unavailable: {type(exc).__name__}"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": platform.node(),
        "pid": os.getpid(),
        "device": device,
        "torch": torch_version,
        "cuda_available": cuda_available,
    }
