"""Dependency-light infrastructure for analysis artifacts.

The analysis jobs consume NumPy/JSON artifacts and must remain importable in
environments that do not initialize the training stack.  Shared filesystem
and provenance behavior lives in :mod:`scripts.analysis.artifacts`; task
modules keep their task-specific validation and error types at the boundary.
"""

from .artifacts import ArtifactError, ArtifactReader, ImmutableArtifactWriter

__all__ = ["ArtifactError", "ArtifactReader", "ImmutableArtifactWriter"]
