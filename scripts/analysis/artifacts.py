"""Read and write immutable JSON/NumPy analysis artifacts.

This module intentionally depends only on the Python standard library and
NumPy.  Task-specific modules may wrap :class:`ArtifactError` with their own
exception type while retaining the old private helper names used by focused
tests.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import numpy as np


class ArtifactError(ValueError):
    """Raised when an artifact cannot be read or written safely."""


def _jsonable(value: Any) -> Any:
    """Convert common NumPy containers into deterministic JSON values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _raise(error_type: type[Exception], message: str, cause: BaseException | None = None) -> None:
    error = error_type(message)
    if cause is None:
        raise error
    raise error from cause


class ArtifactReader:
    """Read and hash immutable analysis inputs.

    ``error_type`` lets each task preserve its public error taxonomy without
    copying the filesystem implementation into every diagnostic module.
    """

    def __init__(self, error_type: type[Exception] = ArtifactError) -> None:
        self.error_type = error_type

    def read_json(self, path: str | Path, *, name: str = "JSON artifact") -> dict[str, Any]:
        artifact_path = Path(path)
        try:
            payload = json.loads(artifact_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            _raise(self.error_type, f"cannot load {name}: {artifact_path}", exc)
        if not isinstance(payload, dict):
            _raise(self.error_type, f"{name} must be a JSON object")
        return payload

    def read_npz(self, path: str | Path, *, name: str = "NumPy artifact") -> dict[str, np.ndarray]:
        artifact_path = Path(path)
        try:
            with np.load(artifact_path, allow_pickle=False) as archive:
                return {key: np.array(archive[key]) for key in archive.files}
        except (OSError, ValueError) as exc:
            _raise(self.error_type, f"cannot load {name}: {artifact_path}", exc)
        raise AssertionError("unreachable")

    def sha256_file(self, path: str | Path, *, description: str = "artifact") -> str:
        artifact_path = Path(path)
        digest = hashlib.sha256()
        try:
            with artifact_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            _raise(self.error_type, f"cannot hash {description}: {artifact_path}", exc)
        return digest.hexdigest()

    def git_commit(self, project_root: str | Path, *, required: bool = False) -> str | None:
        root = Path(project_root)
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            if required:
                _raise(self.error_type, "cannot record the source Git commit", exc)
            return None
        commit = result.stdout.strip()
        if not commit:
            if required:
                _raise(self.error_type, "source Git commit is empty")
            return None
        return commit


class ImmutableArtifactWriter:
    """Write artifacts once, atomically, and idempotently.

    Existing files are compared byte-for-byte (or array-for-array for NPZ)
    and are never overwritten.  New files are first written in the destination
    directory and then installed with a same-directory hard link.  The hard
    link is an atomic no-clobber operation, so concurrent writers cannot
    replace a winner after an existence check.
    """

    def __init__(self, error_type: type[Exception] = ArtifactError) -> None:
        self.error_type = error_type

    def _existing_text(self, path: Path, expected: str) -> bool:
        try:
            existing = path.read_text()
        except OSError as exc:
            _raise(self.error_type, f"cannot inspect existing output: {path}", exc)
        if existing != expected:
            _raise(self.error_type, f"refusing to overwrite an incompatible output: {path}")
        return True

    def _install_no_clobber(self, temporary: Path, destination: Path) -> bool:
        """Install ``temporary`` without replacing an existing destination.

        Both paths are created in the destination directory, so ``os.link``
        is atomic on the same filesystem and fails with ``FileExistsError``
        instead of overwriting a file installed by a competing writer.
        """
        try:
            os.link(temporary, destination)
        except FileExistsError:
            return False
        except OSError as exc:
            _raise(self.error_type, f"cannot install output atomically: {destination}", exc)
        return True

    def _atomic_text(self, path: Path, rendered: str) -> None:
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
            if not self._install_no_clobber(temporary, path):
                # A competing writer won the link race.  Compare its complete
                # output and reject the write if it is not identical.
                self._existing_text(path, rendered)
                return
        except OSError as exc:
            _raise(self.error_type, f"cannot write output: {path}", exc)
        finally:
            if temporary is not None and temporary.exists():
                try:
                    temporary.unlink()
                except OSError:
                    pass

    def write_text_once(self, path: str | Path, rendered: str) -> None:
        artifact_path = Path(path)
        if artifact_path.exists():
            self._existing_text(artifact_path, rendered)
            return
        self._atomic_text(artifact_path, rendered)

    def write_json_once(self, path: str | Path, payload: Mapping[str, Any]) -> None:
        rendered = json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n"
        self.write_text_once(path, rendered)

    @staticmethod
    def _arrays_match(path: Path, arrays: Mapping[str, np.ndarray]) -> bool:
        with np.load(path, allow_pickle=False) as existing:
            if set(existing.files) != set(arrays):
                return False
            for key, value in arrays.items():
                existing_array = existing[key]
                if np.issubdtype(value.dtype, np.inexact):
                    matches = np.array_equal(existing_array, value, equal_nan=True)
                else:
                    matches = np.array_equal(existing_array, value)
                if not matches:
                    return False
        return True

    def write_npz_once(self, path: str | Path, arrays: Mapping[str, np.ndarray]) -> None:
        artifact_path = Path(path)
        if artifact_path.exists():
            try:
                matches = self._arrays_match(artifact_path, arrays)
            except (OSError, ValueError) as exc:
                _raise(
                    self.error_type,
                    f"cannot validate existing NumPy output: {artifact_path}",
                    exc,
                )
            if not matches:
                _raise(self.error_type, f"existing output arrays differ: {artifact_path}")
            return

        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", suffix=".npz", dir=artifact_path.parent, delete=False
            ) as handle:
                temporary = Path(handle.name)
            np.savez_compressed(temporary, **arrays)
        except (OSError, ValueError) as exc:
            _raise(self.error_type, f"cannot write NumPy output: {artifact_path}", exc)
        try:
            if not self._install_no_clobber(temporary, artifact_path):
                # A competing writer won the link race.  Compare its complete
                # archive and reject the write if any array differs.
                try:
                    matches = self._arrays_match(artifact_path, arrays)
                except (OSError, ValueError) as exc:
                    _raise(
                        self.error_type,
                        f"cannot validate existing NumPy output: {artifact_path}",
                        exc,
                    )
                if not matches:
                    _raise(self.error_type, f"existing output arrays differ: {artifact_path}")
                return
        except OSError as exc:
            _raise(self.error_type, f"cannot install NumPy output: {artifact_path}", exc)
        finally:
            if temporary is not None and temporary.exists():
                try:
                    temporary.unlink()
                except OSError:
                    pass


__all__ = ["ArtifactError", "ArtifactReader", "ImmutableArtifactWriter"]
