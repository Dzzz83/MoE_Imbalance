"""Human-facing CLI for read-only inner-study diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from .artifacts import InnerArtifactReader
from .contracts import DiagnosticsError
from .runner import StudyDiagnosticsRunner


_DEFAULT_CONFIG = "configs/studies/ridge_sinkhorn_3seed_v1.yaml"


def _reuse_roots(values: Sequence[str]) -> dict[str, Path]:
    """Parse unique NAME=PATH roots with clear argument-local errors."""
    roots: dict[str, Path] = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise DiagnosticsError(f"invalid --reuse-root {value!r}; expected NAME=PATH")
        if name in roots:
            raise DiagnosticsError(f"duplicate --reuse-root name: {name}")
        roots[name] = Path(path).expanduser()
    return roots


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m expert_method.diagnostics",
        description=(
            "Validate and analyze saved inner-study predictions and locked router states. "
            "This command never reads outer predictions, outer labels, or CIFAR test data."
        ),
    )
    parser.add_argument("--config", default=_DEFAULT_CONFIG, help="frozen study YAML")
    parser.add_argument("--artifact-root", required=True, help="read-only result snapshot root")
    parser.add_argument(
        "--reuse-root", action="append", default=[], metavar="NAME=PATH",
        help="read-only historical OOF root; repeat once per named root",
    )
    parser.add_argument("--stage", choices=("inner",), default="inner")
    parser.add_argument("--output-root", help="new or identical-rerun output directory")
    parser.add_argument(
        "--validate-only", action="store_true",
        help="validate identities and saved payloads (15 locks, 240 jobs); write no analysis output",
    )
    parser.add_argument("--no-figures", action="store_true", help="write tables and report without PNG/SVG figures")
    parser.add_argument(
        "--limit-pairs", type=int, metavar="N",
        help="run an initial N-pair smoke check; omit for the complete 15-pair analysis",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run validation or diagnostics and print a compact JSON result."""
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.validate_only and not args.output_root:
        parser.error("--output-root is required unless --validate-only is set")
    try:
        reader = InnerArtifactReader(
            config_path=args.config,
            artifact_root=args.artifact_root,
            reuse_roots=_reuse_roots(args.reuse_root),
        )
        runner = StudyDiagnosticsRunner(
            reader,
            output_root=args.output_root or ".",
            command_arguments=list(sys.argv[1:] if argv is None else argv),
            make_figures=not args.no_figures,
        )
        result = runner.validate_only() if args.validate_only else runner.run(limit_pairs=args.limit_pairs)
    except DiagnosticsError as exc:
        print(f"diagnostics error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised with module invocation
    raise SystemExit(main())
