"""Spawn-safe minimal workers for canonical promotion lock tests."""

from __future__ import annotations

import json
from pathlib import Path
import traceback
from typing import Any, Mapping


class MinimalArtifactStore:
    """Small store fake that exposes only the promotion service contract."""

    def __init__(self, run_dir: str | Path) -> None:
        self._run_dir = Path(run_dir)

    def run_dir(self, _context: object) -> Path:
        return self._run_dir

    def validate_completed_run(
        self, _context: object, *, expected_config: Mapping[str, Any]
    ) -> dict[str, Any]:
        payload_path = self._run_dir / "validated.json"
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        if payload != dict(expected_config):
            raise ValueError("fake completed payload does not match its expected config")
        return {"run_dir": self._run_dir}


def promote_in_spawned_process(
    source: str,
    destination: str,
    expected_config: Mapping[str, Any],
    start_barrier: Any,
    result_queue: Any,
) -> None:
    """Run one promotion and report errors instead of leaking a worker."""
    try:
        from expert_method.workflow import _promote_attempt_run

        start_barrier.wait(timeout=10)
        promoted_existing = _promote_attempt_run(
            attempt_store=MinimalArtifactStore(source),
            canonical_store=MinimalArtifactStore(destination),
            context="job-1",
            expected_config=expected_config,
            attempt=None,
        )
        result_queue.put(("ok", promoted_existing))
    except BaseException:  # communicate even process-level exceptions to parent
        result_queue.put(("error", traceback.format_exc()))
