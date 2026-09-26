"""Synthetic protocol tests for Ridge/Sinkhorn role checked OOF loading."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from pathlib import Path as _TestPath
import sys as _test_sys
_TEST_PACKAGE_DIR = str(_TestPath(__file__).resolve().parent.parent)
if _TEST_PACKAGE_DIR not in _test_sys.path:
    _test_sys.path.insert(0, _TEST_PACKAGE_DIR)
from repo_root import REPO_ROOT
if str(REPO_ROOT) not in _test_sys.path:
    _test_sys.path.insert(0, str(REPO_ROOT))
_PROJECT_ROOT = str(REPO_ROOT)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from data.nested_oof import NestedOOFFoldManager  # noqa: E402
from scripts.ridge_sinkhorn_data import (  # noqa: E402
    DataRole,
    RidgeSinkhornDataError,
    RidgeSinkhornRoleDataset,
    load_ridge_sinkhorn_development_data,
)
from scripts.task3c_oof import AlignedOOFDataset  # noqa: E402
from scripts.task3e_fixed import (  # noqa: E402
    EXPERT_ORDER,
    RestrictedAnalysisDataset,
)


def _small_manager() -> NestedOOFFoldManager:
    labels = np.repeat(np.arange(4, dtype=np.int64), 10)
    indices = np.arange(100, 100 + len(labels), dtype=np.int64)
    return NestedOOFFoldManager(
        indices,
        labels,
        seed=42,
        num_classes=4,
        expert_order=EXPERT_ORDER,
    )


def _aligned(manager: NestedOOFFoldManager) -> AlignedOOFDataset:
    labels_by_index = dict(zip(manager.canonical_indices, manager.training_labels))
    rows: list[tuple[int, int, int]] = []
    for index in sorted(manager.outer_fold(0).expert_training_indices):
        inner_id = next(
            inner.inner_fold_id
            for inner in manager.outer_fold(0).inner_folds
            if index in inner.prediction_indices
        )
        rows.append((index, inner_id, labels_by_index[index]))
    indices = np.asarray([row[0] for row in rows], dtype=np.int64)
    inner_ids = np.asarray([row[1] for row in rows], dtype=np.int64)
    labels = np.asarray([row[2] for row in rows], dtype=np.int64)
    rng = np.random.default_rng(23)
    logits = rng.normal(size=(len(indices), len(EXPERT_ORDER), manager.num_classes))
    return AlignedOOFDataset(
        sample_indices=indices,
        outer_fold_ids=np.zeros(len(indices), dtype=np.int64),
        inner_fold_ids=inner_ids,
        labels=labels,
        logits=logits,
        expert_names=EXPERT_ORDER,
        metadata={"synthetic_fixture": True},
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_source_files(directory: Path) -> dict[str, dict[str, object]]:
    arrays_path = directory / "aligned_oof.npz"
    metadata_path = directory / "aligned_oof_metadata.json"
    batch_path = directory / "batch_manifest.json"
    fold_path = directory / "fold_manifest.json"
    batch_path.write_text("synthetic batch sidecar\n")
    fold_path.write_text("synthetic fold sidecar\n")
    return {
        "aligned_oof_arrays": {"path": arrays_path, "sha256": _sha256(arrays_path)},
        "aligned_oof_metadata": {"path": metadata_path, "sha256": _sha256(metadata_path)},
        "task3c_batch_manifest": {"path": batch_path, "sha256": _sha256(batch_path)},
        "task3c_fold_manifest": {"path": fold_path, "sha256": _sha256(fold_path)},
    }


def _role_dataset(
    aligned: AlignedOOFDataset,
    role: DataRole,
    fold_ids: tuple[int, ...],
) -> RidgeSinkhornRoleDataset:
    selected = aligned.select_inner_folds(fold_ids)
    return RidgeSinkhornRoleDataset(
        role=role,
        sample_indices=selected.sample_indices,
        labels=selected.labels,
        inner_fold_ids=selected.inner_fold_ids,
        outer_fold_ids=selected.outer_fold_ids,
        logits=selected.logits,
        expert_names=selected.expert_names,
        metadata=selected.metadata,
    )


def test_role_dataset_is_immutable_and_rejects_the_wrong_fold_role():
    manager = _small_manager()
    aligned = _aligned(manager)
    fit = _role_dataset(aligned, DataRole.FIT, (1, 2, 3))
    selection = _role_dataset(aligned, DataRole.SELECTION, (0,))

    assert fit.role is DataRole.FIT
    assert set(np.unique(fit.inner_fold_ids)) == {1, 2, 3}
    assert selection.role is DataRole.SELECTION
    assert set(np.unique(selection.inner_fold_ids)) == {0}
    assert set(fit.sample_indices).isdisjoint(set(selection.sample_indices))
    assert not fit.logits.flags.writeable
    with pytest.raises(ValueError):
        fit.labels[0] = 99

    with pytest.raises(RidgeSinkhornDataError, match="fit data"):
        _role_dataset(aligned, DataRole.FIT, (0, 1, 2, 3))
    with pytest.raises(RidgeSinkhornDataError, match="selection data"):
        _role_dataset(aligned, DataRole.SELECTION, (1, 2, 3))


def test_selection_view_checks_full_fold_membership_and_canonical_labels():
    manager = _small_manager()
    aligned = _aligned(manager)
    selection = RidgeSinkhornRoleDataset.from_aligned_selection(aligned, manager)
    expected = set(manager.outer_fold(0).router_development.selection_indices)
    assert set(selection.sample_indices.tolist()) == expected
    assert selection.role is DataRole.SELECTION

    wrong_labels = aligned.labels.copy()
    wrong_labels[0] = (wrong_labels[0] + 1) % manager.num_classes
    bad_labels = AlignedOOFDataset(
        sample_indices=aligned.sample_indices,
        outer_fold_ids=aligned.outer_fold_ids,
        inner_fold_ids=aligned.inner_fold_ids,
        labels=wrong_labels,
        logits=aligned.logits,
        expert_names=aligned.expert_names,
        metadata=aligned.metadata,
    )
    with pytest.raises(RidgeSinkhornDataError, match="fold or label validation"):
        RidgeSinkhornRoleDataset.from_aligned_selection(bad_labels, manager)

    wrong_membership = aligned.sample_indices.copy()
    wrong_membership[0] = manager.outer_fold(0).evaluation_indices[0]
    bad_membership = AlignedOOFDataset(
        sample_indices=wrong_membership,
        outer_fold_ids=aligned.outer_fold_ids,
        inner_fold_ids=aligned.inner_fold_ids,
        labels=aligned.labels,
        logits=aligned.logits,
        expert_names=aligned.expert_names,
        metadata=aligned.metadata,
    )
    with pytest.raises(RidgeSinkhornDataError, match="fold or label validation"):
        RidgeSinkhornRoleDataset.from_aligned_selection(bad_membership, manager)


def test_loader_uses_existing_provenance_gate_and_returns_disjoint_role_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manager = _small_manager()
    aligned = _aligned(manager)
    aligned.save(tmp_path, manager=manager)
    source_files = _valid_source_files(tmp_path)
    restricted_fit = RestrictedAnalysisDataset.from_aligned(
        aligned, manager, enforce_frozen_population=False
    )
    calls: list[tuple[Path, NestedOOFFoldManager]] = []

    def validate_with_existing_seam(directory, supplied_manager):
        calls.append((Path(directory), supplied_manager))
        return restricted_fit, source_files

    monkeypatch.setattr(
        "scripts.ridge_sinkhorn_data.load_restricted_analysis_dataset",
        validate_with_existing_seam,
    )

    development = load_ridge_sinkhorn_development_data(tmp_path, manager)

    assert calls == [(tmp_path, manager)]
    assert set(np.unique(development.fit.inner_fold_ids)) == {1, 2, 3}
    assert set(np.unique(development.selection.inner_fold_ids)) == {0}
    assert set(development.fit.sample_indices).isdisjoint(
        set(development.selection.sample_indices)
    )
    assert set(development.fit.sample_indices) == set(
        manager.outer_fold(0).router_development.fit_indices
    )
    assert set(development.selection.sample_indices) == set(
        manager.outer_fold(0).router_development.selection_indices
    )
    assert set(development.source_files) == set(source_files)
    assert len(development.source_files) == 4


def test_loader_detects_sidecar_changes_between_validation_and_array_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    manager = _small_manager()
    aligned = _aligned(manager)
    aligned.save(tmp_path, manager=manager)
    source_files = _valid_source_files(tmp_path)
    restricted_fit = RestrictedAnalysisDataset.from_aligned(
        aligned, manager, enforce_frozen_population=False
    )

    def validate_then_change_sidecar(directory, supplied_manager):
        (tmp_path / "batch_manifest.json").write_text("changed after validation\n")
        return restricted_fit, source_files

    monkeypatch.setattr(
        "scripts.ridge_sinkhorn_data.load_restricted_analysis_dataset",
        validate_then_change_sidecar,
    )

    with pytest.raises(RidgeSinkhornDataError, match="changed while loading"):
        load_ridge_sinkhorn_development_data(tmp_path, manager)


def test_restricted_fit_view_cannot_be_reconstructed_with_selection_rows():
    manager = _small_manager()
    aligned = _aligned(manager)
    selected = aligned.select_inner_folds((0,))
    with pytest.raises(RidgeSinkhornDataError, match="fit data"):
        RidgeSinkhornRoleDataset(
            role=DataRole.FIT,
            sample_indices=selected.sample_indices,
            labels=selected.labels,
            inner_fold_ids=selected.inner_fold_ids,
            outer_fold_ids=selected.outer_fold_ids,
            logits=selected.logits,
            expert_names=selected.expert_names,
            metadata=selected.metadata,
        )
