"""Role-checked data access for the Ridge/Sinkhorn development study.

The public loader validates the complete Task 3C aligned artifact through the
existing restricted analysis loader, then exposes separate router-fit and
router-selection views.  It never opens outer-evaluation prediction artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from data.nested_oof import NestedOOFFoldManager, OOFProtocolError
from scripts.task3c_oof import AlignedOOFDataset
from scripts.task3e_fixed import (
    EXPERT_ORDER,
    PERMITTED_ANALYSIS_INNER_FOLDS,
    RESERVED_ROUTER_SELECTION_INNER_FOLDS,
    RestrictedAnalysisDataset,
    load_restricted_analysis_dataset,
)


RIDGE_SINKHORN_FIT_INNER_FOLDS = PERMITTED_ANALYSIS_INNER_FOLDS
RIDGE_SINKHORN_SELECTION_INNER_FOLDS = RESERVED_ROUTER_SELECTION_INNER_FOLDS
RIDGE_SINKHORN_OUTER_FOLD = 0


class RidgeSinkhornDataError(OOFProtocolError):
    """Raised when Ridge/Sinkhorn data roles or provenance are invalid."""


class DataRole(str, Enum):
    """Permitted roles for rows in the development partition."""

    FIT = "fit"
    SELECTION = "selection"


def _readonly_copy(values: np.ndarray) -> np.ndarray:
    array = np.array(values, copy=True)
    array.setflags(write=False)
    return array


def _validate_source_files(
    source_files: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    required = {
        "aligned_oof_arrays",
        "aligned_oof_metadata",
        "task3c_batch_manifest",
        "task3c_fold_manifest",
    }
    if set(source_files) != required:
        raise RidgeSinkhornDataError(
            "validated source files must identify the aligned arrays and all Task 3C sidecars"
        )

    checked: dict[str, dict[str, Any]] = {}
    for name, entry in source_files.items():
        if not isinstance(entry, Mapping) or "path" not in entry or "sha256" not in entry:
            raise RidgeSinkhornDataError(f"source provenance is malformed for {name}")
        digest = entry["sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise RidgeSinkhornDataError(f"source SHA-256 is malformed for {name}")
        try:
            int(digest, 16)
        except (TypeError, ValueError) as exc:
            raise RidgeSinkhornDataError(
                f"source SHA-256 is not hexadecimal for {name}"
            ) from exc
        checked[name] = {"path": Path(entry["path"]), "sha256": digest.lower()}
    return checked


def _verify_source_files_unchanged(
    source_files: Mapping[str, Mapping[str, Any]],
) -> None:
    """Close the gap between the public provenance check and the second load."""
    for name, entry in source_files.items():
        path = Path(entry["path"])
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as exc:
            raise RidgeSinkhornDataError(
                f"cannot recheck validated Task 3C source file: {path}"
            ) from exc
        if digest.hexdigest() != entry["sha256"]:
            raise RidgeSinkhornDataError(
                f"Task 3C source file changed while loading: {name}"
            )


@dataclass(frozen=True)
class RidgeSinkhornRoleDataset:
    """Immutable array view restricted to one declared development role."""

    role: DataRole
    sample_indices: np.ndarray
    labels: np.ndarray
    inner_fold_ids: np.ndarray
    outer_fold_ids: np.ndarray
    logits: np.ndarray
    expert_names: tuple[str, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        try:
            role = DataRole(self.role)
        except ValueError as exc:
            raise RidgeSinkhornDataError(f"unknown Ridge/Sinkhorn data role: {self.role!r}") from exc

        indices = np.asarray(self.sample_indices)
        labels = np.asarray(self.labels)
        inner_ids = np.asarray(self.inner_fold_ids)
        outer_ids = np.asarray(self.outer_fold_ids)
        logits = np.asarray(self.logits)
        vectors = (indices, labels, inner_ids, outer_ids)
        if any(array.ndim != 1 for array in vectors):
            raise RidgeSinkhornDataError("role indices and labels must be one-dimensional")
        if len(indices) == 0 or any(len(array) != len(indices) for array in vectors[1:]):
            raise RidgeSinkhornDataError("role arrays must be nonempty and aligned")
        if any(not np.issubdtype(array.dtype, np.integer) for array in vectors):
            raise RidgeSinkhornDataError("role indices, labels, and fold IDs must be integers")
        if len(np.unique(indices)) != len(indices):
            raise RidgeSinkhornDataError("role sample IDs contain duplicates")
        expected_folds = (
            RIDGE_SINKHORN_FIT_INNER_FOLDS
            if role is DataRole.FIT
            else RIDGE_SINKHORN_SELECTION_INNER_FOLDS
        )
        if tuple(sorted(int(value) for value in np.unique(inner_ids))) != tuple(
            sorted(expected_folds)
        ):
            raise RidgeSinkhornDataError(
                f"{role.value} data must contain exactly inner folds {expected_folds}"
            )
        if np.any(outer_ids != RIDGE_SINKHORN_OUTER_FOLD):
            raise RidgeSinkhornDataError("development role contains a nonzero outer fold")
        if tuple(self.expert_names) != EXPERT_ORDER:
            raise RidgeSinkhornDataError(
                "expert ordering must be CE, LAL, BalancedSoftmax, Mixup"
            )
        if logits.ndim != 3 or logits.shape[:2] != (len(indices), len(EXPERT_ORDER)):
            raise RidgeSinkhornDataError("role logits must have shape (samples, 4, classes)")
        if (
            not np.issubdtype(logits.dtype, np.number)
            or np.iscomplexobj(logits)
            or not np.isfinite(logits).all()
        ):
            raise RidgeSinkhornDataError("role logits must be finite real numeric values")
        if logits.shape[2] < 1 or np.any(labels < 0) or np.any(labels >= logits.shape[2]):
            raise RidgeSinkhornDataError("role labels are outside the logit class range")

        object.__setattr__(self, "role", role)
        object.__setattr__(self, "sample_indices", _readonly_copy(indices))
        object.__setattr__(self, "labels", _readonly_copy(labels))
        object.__setattr__(self, "inner_fold_ids", _readonly_copy(inner_ids))
        object.__setattr__(self, "outer_fold_ids", _readonly_copy(outer_ids))
        object.__setattr__(self, "logits", _readonly_copy(logits))
        object.__setattr__(self, "expert_names", tuple(self.expert_names))
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def num_samples(self) -> int:
        return int(len(self.sample_indices))

    @classmethod
    def from_restricted_fit(
        cls, dataset: RestrictedAnalysisDataset
    ) -> "RidgeSinkhornRoleDataset":
        """Wrap the existing restricted fit view without broadening its role."""
        return cls(
            role=DataRole.FIT,
            sample_indices=dataset.sample_indices,
            labels=dataset.labels,
            inner_fold_ids=dataset.inner_fold_ids,
            outer_fold_ids=dataset.outer_fold_ids,
            logits=dataset.logits,
            expert_names=dataset.expert_names,
            metadata=dataset.metadata,
        )

    @classmethod
    def from_aligned_selection(
        cls,
        aligned: AlignedOOFDataset,
        manager: NestedOOFFoldManager,
    ) -> "RidgeSinkhornRoleDataset":
        """Validate full aligned development data, then expose inner fold 0."""
        try:
            aligned.validate(manager)
        except (OOFProtocolError, ValueError) as exc:
            raise RidgeSinkhornDataError(
                "aligned OOF data fails fold or label validation"
            ) from exc
        selected = aligned.select_inner_folds(RIDGE_SINKHORN_SELECTION_INNER_FOLDS)
        expected = set(manager.outer_fold(RIDGE_SINKHORN_OUTER_FOLD).router_development.selection_indices)
        actual = set(int(value) for value in selected.sample_indices.tolist())
        if actual != expected:
            raise RidgeSinkhornDataError(
                "selection samples are not exactly the frozen inner-fold-0 partition"
            )
        outer_evaluation = set(
            manager.outer_fold(RIDGE_SINKHORN_OUTER_FOLD).evaluation_indices
        )
        if actual & outer_evaluation:
            raise RidgeSinkhornDataError(
                "selection data overlaps the reserved outer-evaluation population"
            )
        return cls(
            role=DataRole.SELECTION,
            sample_indices=selected.sample_indices,
            labels=selected.labels,
            inner_fold_ids=selected.inner_fold_ids,
            outer_fold_ids=selected.outer_fold_ids,
            logits=selected.logits,
            expert_names=selected.expert_names,
            metadata=selected.metadata,
        )


@dataclass(frozen=True)
class RidgeSinkhornDevelopmentData:
    """Disjoint fit and selection views plus verified Task 3C provenance."""

    fit: RidgeSinkhornRoleDataset
    selection: RidgeSinkhornRoleDataset
    source_files: Mapping[str, Mapping[str, Any]]

    def __post_init__(self) -> None:
        if self.fit.role is not DataRole.FIT or self.selection.role is not DataRole.SELECTION:
            raise RidgeSinkhornDataError("development data roles are not fit/selection")
        if self.fit.expert_names != self.selection.expert_names:
            raise RidgeSinkhornDataError("fit and selection expert orderings differ")
        if self.fit.logits.shape[2] != self.selection.logits.shape[2]:
            raise RidgeSinkhornDataError("fit and selection class dimensions differ")
        fit_indices = set(int(value) for value in self.fit.sample_indices.tolist())
        selection_indices = set(
            int(value) for value in self.selection.sample_indices.tolist()
        )
        if fit_indices & selection_indices:
            raise RidgeSinkhornDataError("router fit and selection samples overlap")
        object.__setattr__(self, "source_files", _validate_source_files(self.source_files))


def load_ridge_sinkhorn_development_data(
    source_directory: str | Path,
    manager: NestedOOFFoldManager,
) -> RidgeSinkhornDevelopmentData:
    """Load validated Task 3C OOF data as disjoint fit and selection roles.

    The existing restricted loader checks the frozen manager, aligned labels
    and memberships, source checkpoint provenance, batch manifest, and fold
    manifest.  Its fit-only result is preserved as the new loader's fit view.
    The aligned development artifact is reopened only to form the separately
    role-checked inner-fold-0 selection view; no outer-evaluation predictions
    or CIFAR-100 test data are read.
    """
    source_directory = Path(source_directory)
    try:
        restricted_fit, source_files = load_restricted_analysis_dataset(
            source_directory, manager
        )
        checked_sources = _validate_source_files(source_files)
        aligned = AlignedOOFDataset.load(source_directory, manager=manager)
        _verify_source_files_unchanged(checked_sources)
        fit = RidgeSinkhornRoleDataset.from_restricted_fit(restricted_fit)
        selection = RidgeSinkhornRoleDataset.from_aligned_selection(aligned, manager)
        if set(int(value) for value in fit.sample_indices.tolist()) != set(
            manager.outer_fold(RIDGE_SINKHORN_OUTER_FOLD).router_development.fit_indices
        ):
            raise RidgeSinkhornDataError(
                "fit samples are not exactly inner folds 1–3"
            )
        return RidgeSinkhornDevelopmentData(
            fit=fit,
            selection=selection,
            source_files=checked_sources,
        )
    except RidgeSinkhornDataError:
        raise
    except (OOFProtocolError, OSError, ValueError) as exc:
        raise RidgeSinkhornDataError(
            f"cannot load role-checked Ridge/Sinkhorn data from {source_directory}"
        ) from exc


__all__ = [
    "DataRole",
    "RIDGE_SINKHORN_FIT_INNER_FOLDS",
    "RIDGE_SINKHORN_OUTER_FOLD",
    "RIDGE_SINKHORN_SELECTION_INNER_FOLDS",
    "RidgeSinkhornDataError",
    "RidgeSinkhornDevelopmentData",
    "RidgeSinkhornRoleDataset",
    "load_ridge_sinkhorn_development_data",
]
