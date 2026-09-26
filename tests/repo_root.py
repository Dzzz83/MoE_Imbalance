"""Find the repository root independently of test-file nesting depth."""

from pathlib import Path


REPO_ROOT = next(
    path
    for path in Path(__file__).resolve().parents
    if (path / "pyproject.toml").is_file() and (path / "expert_method").is_dir()
)
