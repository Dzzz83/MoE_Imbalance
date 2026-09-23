"""Shared JSON/NPZ, hashing, provenance, and immutable-write primitives."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import numpy as np


class ArtifactError(ValueError):
    """Raised when a generic artifact operation cannot be completed safely."""


def _raise(error_type: type[Exception], message: str, cause: Exception | None = None) -> None:
    if cause is None:
        raise error_type(message)
    raise error_type(message) from cause


def to_jsonable(value: Any) -> Any:
    """Convert supported NumPy containers to the existing JSON representation."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible values with the OOF canonical convention."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        _raise(ArtifactError, f"value is not JSON serializable: {value!r}", exc)
        raise AssertionError("unreachable") from exc


def serialize_json(
    value: Any,
    *,
    indent: int = 2,
    sort_keys: bool = True,
    convert_numpy: bool = True,
) -> str:
    """Render JSON using the existing analysis-artifact formatting."""
    payload = to_jsonable(value) if convert_numpy else value
    try:
        return json.dumps(payload, indent=indent, sort_keys=sort_keys) + "\n"
    except (TypeError, ValueError) as exc:
        _raise(ArtifactError, f"value is not JSON serializable: {value!r}", exc)
        raise AssertionError("unreachable") from exc


def load_json_object(
    path: str | Path,
    *,
    name: str = "JSON artifact",
    error_type: type[Exception] = ArtifactError,
) -> dict[str, Any]:
    """Load a JSON object without changing the stored representation."""
    artifact_path = Path(path)
    try:
        payload = json.loads(artifact_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        _raise(error_type, f"cannot load {name}: {artifact_path}", exc)
        raise AssertionError("unreachable") from exc
    if not isinstance(payload, dict):
        _raise(error_type, f"{name} must contain a JSON object: {artifact_path}")
    return payload


def load_npz_arrays(
    path: str | Path,
    *,
    name: str = "NPZ artifact",
    error_type: type[Exception] = ArtifactError,
) -> dict[str, np.ndarray]:
    """Load all NPZ arrays with object deserialization disabled."""
    artifact_path = Path(path)
    try:
        with np.load(artifact_path, allow_pickle=False) as archive:
            return {key: np.array(archive[key]) for key in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        _raise(error_type, f"cannot load {name}: {artifact_path}", exc)
        raise AssertionError("unreachable") from exc


def sha256_bytes(value: bytes) -> str:
    """Return the SHA-256 digest of raw bytes."""
    return hashlib.sha256(value).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    """Hash an array's contiguous raw bytes, matching existing diagnostics."""
    return sha256_bytes(np.ascontiguousarray(value).tobytes())


def sha256_file(
    path: str | Path,
    *,
    error_type: type[Exception] = ArtifactError,
    description: str = "artifact",
) -> str:
    """Return a file's SHA-256 digest using chunked reads."""
    artifact_path = Path(path)
    digest = hashlib.sha256()
    try:
        with artifact_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        _raise(error_type, f"cannot hash {description}: {artifact_path}", exc)
        raise AssertionError("unreachable") from exc
    return digest.hexdigest()


def validate_sha256(
    value: Any,
    *,
    name: str = "SHA-256 hash",
    error_type: type[Exception] = ArtifactError,
) -> str:
    """Validate and normalize a hexadecimal SHA-256 string."""
    if not isinstance(value, str) or len(value) != 64:
        _raise(error_type, f"{name} must be a 64-character SHA-256 hash")
    try:
        int(value, 16)
    except ValueError as exc:
        _raise(error_type, f"{name} is not hexadecimal SHA-256", exc)
    return value.lower()


def verify_file_hash(
    path: str | Path,
    expected_sha256: Any,
    *,
    name: str = "artifact",
    error_type: type[Exception] = ArtifactError,
) -> str:
    """Require a file to match an expected SHA-256 digest and return the actual hash."""
    expected = validate_sha256(
        expected_sha256,
        name=f"{name} hash",
        error_type=error_type,
    )
    actual = sha256_file(path, error_type=error_type, description=name)
    if actual != expected:
        _raise(error_type, f"{name} hash does not match: {Path(path)}")
    return actual


def repository_relative(path: str | Path, project_root: str | Path) -> str:
    """Return a repository-relative path when possible, otherwise an absolute path."""
    artifact_path = Path(path)
    root = Path(project_root)
    try:
        return str(artifact_path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(artifact_path.resolve())


def git_commit(
    project_root: str | Path,
    *,
    error_type: type[Exception] = ArtifactError,
) -> str:
    """Return the current repository commit for provenance records."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(project_root),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        _raise(error_type, "cannot record the source Git commit", exc)
        raise AssertionError("unreachable") from exc
    commit = completed.stdout.strip()
    if not commit:
        _raise(error_type, "source Git commit is empty")
    return commit


def write_text_once(
    path: str | Path,
    content: str,
    *,
    error_type: type[Exception] = ArtifactError,
) -> None:
    """Write exact text once; identical reruns are accepted."""
    artifact_path = Path(path)
    if artifact_path.exists():
        try:
            existing = artifact_path.read_text()
        except OSError as exc:
            _raise(error_type, f"cannot inspect existing output: {artifact_path}", exc)
            raise AssertionError("unreachable") from exc
        if existing != content:
            _raise(
                error_type,
                f"refusing to overwrite an incompatible output: {artifact_path}",
            )
        return
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(content)


def write_texts_once(
    files: Mapping[str | Path, str],
    *,
    error_type: type[Exception] = ArtifactError,
) -> None:
    """Preflight and then write a group of immutable text artifacts."""
    normalized = {Path(path): content for path, content in files.items()}
    for artifact_path, content in normalized.items():
        if artifact_path.exists():
            try:
                existing = artifact_path.read_text()
            except OSError as exc:
                _raise(error_type, f"cannot inspect existing output: {artifact_path}", exc)
                raise AssertionError("unreachable") from exc
            if existing != content:
                _raise(
                    error_type,
                    f"refusing to overwrite an incompatible output: {artifact_path}",
                )
    for artifact_path, content in normalized.items():
        if not artifact_path.exists():
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(content)


def write_json_once(
    path: str | Path,
    payload: Any,
    *,
    error_type: type[Exception] = ArtifactError,
) -> None:
    """Serialize and write one immutable JSON artifact."""
    try:
        rendered = serialize_json(payload)
    except ArtifactError as exc:
        _raise(error_type, str(exc), exc)
        raise AssertionError("unreachable") from exc
    write_text_once(path, rendered, error_type=error_type)


def write_npz_once(
    path: str | Path,
    arrays: Mapping[str, np.ndarray],
    *,
    error_type: type[Exception] = ArtifactError,
) -> None:
    """Write a compressed NPZ once and reject incompatible reruns."""
    artifact_path = Path(path)
    normalized = {str(key): np.asarray(value) for key, value in arrays.items()}
    if artifact_path.exists():
        mismatch: str | None = None
        try:
            with np.load(artifact_path, allow_pickle=False) as existing:
                if set(existing.files) != set(normalized):
                    mismatch = f"existing output arrays differ: {artifact_path}"
                else:
                    for key, value in normalized.items():
                        existing_array = existing[key]
                        if np.issubdtype(value.dtype, np.inexact):
                            matches = np.array_equal(existing_array, value, equal_nan=True)
                        else:
                            matches = np.array_equal(existing_array, value)
                        if not matches:
                            mismatch = f"existing output array differs for {key}: {artifact_path}"
                            break
        except (OSError, ValueError, KeyError) as exc:
            _raise(error_type, f"cannot validate existing NumPy output: {artifact_path}", exc)
            raise AssertionError("unreachable") from exc
        if mismatch is not None:
            _raise(error_type, mismatch)
        return

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npz", dir=artifact_path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        np.savez_compressed(temporary, **normalized)
        temporary.replace(artifact_path)
    except OSError as exc:
        _raise(error_type, f"cannot write NumPy output: {artifact_path}", exc)
        raise AssertionError("unreachable") from exc
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


__all__ = [
    "ArtifactError",
    "canonical_json",
    "git_commit",
    "load_json_object",
    "load_npz_arrays",
    "repository_relative",
    "serialize_json",
    "sha256_array",
    "sha256_bytes",
    "sha256_file",
    "to_jsonable",
    "validate_sha256",
    "verify_file_hash",
    "write_json_once",
    "write_npz_once",
    "write_text_once",
    "write_texts_once",
]
