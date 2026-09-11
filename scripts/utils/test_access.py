"""
Test-set access logging.

The routing candidate set is frozen in `docs/routing-preregistration.md` before
any test-set number is seen. That discipline is only real if peeking is
*visible*, so every evaluation entry point appends an entry here first.

The log exists to make access auditable, never to block work: if the log cannot
be written, the caller continues and the failure is reported, not raised.
"""

from __future__ import annotations

import datetime as _dt
import subprocess
from pathlib import Path


class TestAccessLog:
    """Append-only record of every read of the CIFAR-100 test set."""

    HEADER = (
        "# Test-Set Access Log\n\n"
        "> Append-only. Every entry is written automatically by the evaluation\n"
        "> entry points the moment they load the test set. The candidate routing\n"
        "> rules were frozen in `docs/routing-preregistration.md`; this log exists\n"
        "> so that any access *after* that freeze is visible.\n\n"
        "| timestamp (UTC) | git | command | note |\n"
        "|:--|:--|:--|:--|\n"
    )

    def __init__(self, path: str | Path = 'docs/test-access-log.md') -> None:
        self.path = Path(path)

    def entries(self) -> list[str]:
        """Return the table rows currently recorded (empty when no log exists)."""
        if not self.path.exists():
            return []
        rows = []
        for line in self.path.read_text().splitlines():
            if line.startswith('|') and not line.startswith('|:') and 'timestamp' not in line:
                rows.append(line)
        return rows

    def record(self, command: str, note: str = '') -> bool:
        """Append one entry. Returns True when written, False when it could not be.

        Never raises: an unwritable log must not break an evaluation run.
        """
        stamp = _dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        row = f"| {stamp} | {self._git_hash()} | `{command}` | {note} |\n"
        try:
            if not self.path.exists():
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(self.HEADER)
            with open(self.path, 'a') as f:
                f.write(row)
            return True
        except OSError:
            return False

    @staticmethod
    def _git_hash() -> str:
        try:
            out = subprocess.run(
                ['git', 'rev-parse', '--short', 'HEAD'],
                capture_output=True, text=True, timeout=5,
            )
            return out.stdout.strip() or 'n/a'
        except Exception:  # noqa: BLE001 - git may be absent; never break a run
            return 'n/a'
