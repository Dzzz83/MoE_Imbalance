#!/usr/bin/env python3
"""Deprecated compatibility wrapper for lock/evaluate/report services."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Keep historical helper imports available to notebooks and focused tests;
# actual stage orchestration now lives in ``expert_method.analysis``.
from expert_method.analysis import (  # noqa: E402
    expected_outer_source_hashes as _expected_outer_source_hashes,
    job_id_lookup as _job_id_lookup,
    read_frozen_matrix_identity as _read_frozen_matrix_identity,
    reference_source_hashes as _reference_source_hashes,
    source_hashes as _source_hashes,
    validate_complete_job_references as _validate_complete_job_references,
    validate_fold_lock_membership as _validate_fold_lock_membership,
    validate_outer_stage_prerequisites as _validate_outer_stage_prerequisites,
)
from expert_method.legacy import (  # noqa: E402
    analysis_main,
    build_analysis_parser as build_parser,
    parse_matrix_reuse_root as _parse_reuse_root,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Retain historical flags while dispatching through package services."""
    return analysis_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
