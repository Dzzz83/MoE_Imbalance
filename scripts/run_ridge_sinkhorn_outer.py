#!/usr/bin/env python3
"""Evaluate the single locked Ridge/Sinkhorn candidate on outer fold 0."""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from scripts.ridge_sinkhorn_outer import run_outer  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--artifact-root", default="artifacts/oof")
    parser.add_argument("--development-directory", default="artifacts/oof/ridge_sinkhorn_v3")
    args = parser.parse_args(argv)
    try:
        result = run_outer(**vars(args))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Ridge/Sinkhorn outer error: {exc}", file=sys.stderr)
        return 2
    print(f"candidate: {result['candidate_id']}")
    print(f"expansion gate passed: {result['expansion_gate_passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
