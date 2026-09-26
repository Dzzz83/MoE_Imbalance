"""Legacy import paths remain aliases to the canonical package engines."""

from importlib import import_module

import pytest


@pytest.mark.parametrize(
    ("compatibility_path", "canonical_path", "symbols"),
    (
        (
            "scripts.oof_pipeline",
            "expert_method.oof.pipeline",
            ("OOFPipeline", "_sha256_file"),
        ),
        (
            "scripts.ridge_sinkhorn_matrix",
            "expert_method.ridge_sinkhorn.matrix",
            ("OOFMatrixPlanner", "_read_bundle"),
        ),
        (
            "scripts.ridge_sinkhorn_3seed_study",
            "expert_method.ridge_sinkhorn.three_seed_study",
            ("StudyArtifactRepository", "_sha256_text"),
        ),
    ),
)
def test_legacy_module_paths_reexport_public_and_private_symbols(
    compatibility_path: str, canonical_path: str, symbols: tuple[str, ...]
) -> None:
    compatibility = import_module(compatibility_path)
    canonical = import_module(canonical_path)

    for name in symbols:
        assert getattr(compatibility, name) is getattr(canonical, name)
