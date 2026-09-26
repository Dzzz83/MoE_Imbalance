#!/usr/bin/env python3
"""Deprecated compatibility wrapper for the config-driven expert-matrix CLI."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from expert_method.legacy import build_matrix_parser as build_parser
from expert_method.legacy import matrix_main


def main(argv: Sequence[str] | None = None) -> int:
    """Retain historical flags while dispatching through package services."""
    return matrix_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
