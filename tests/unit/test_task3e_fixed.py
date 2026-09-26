"""Synthetic regression tests for Task 3E-A.

These fixtures use only hand-constructed logits and fold arrays.  They never
load CIFAR data, checkpoints, or any Task 3C artifact.
"""

from __future__ import annotations

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
from scripts.task3c_oof import AlignedOOFDataset  # noqa: E402
from scripts.task3e_fixed import (  # noqa: E402
    EXPERT_ORDER,
    FixedWeightAnalyzer,
    RestrictedAnalysisDataset,
    Task3EError,
    generate_fixed_weight_candidates,
    pareto_frontier,
    verify_uniform_baseline,
)


def _small_restricted_dataset() -> RestrictedAnalysisDataset:
    labels = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    logits = np.zeros((6, 4, 3), dtype=np.float64)
    logits[0] = np.array([[0, 4, 0], [0, 0, 3], [0, 0, 2], [0, 0, 1]])
    logits[1] = np.array([[4, 0, 0], [0, 3, 0], [0, 2, 0], [0, 1, 0]])
    logits[2] = np.full((4, 3), [4, 0, 0], dtype=np.float64)
    logits[3] = logits[0]
    logits[4] = logits[1]
    logits[5] = logits[2]
    return RestrictedAnalysisDataset.from_arrays(
        sample_indices=np.arange(6),
        labels=labels,
        inner_fold_ids=np.array([1, 2, 3, 1, 2, 3]),
        logits=logits,
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


def _aligned_for_manager(manager: NestedOOFFoldManager) -> AlignedOOFDataset:
    labels_by_index = dict(zip(manager.canonical_indices, manager.training_labels))
    rows = []
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
    logits = np.zeros((len(indices), 4, manager.num_classes), dtype=np.float64)
    return AlignedOOFDataset(
        sample_indices=indices,
        outer_fold_ids=np.zeros(len(indices), dtype=np.int64),
        inner_fold_ids=inner_ids,
        labels=labels,
        logits=logits,
        expert_names=EXPERT_ORDER,
        metadata={},
    )


def test_candidate_grid_is_exact_stable_and_contains_references():
    candidates = generate_fixed_weight_candidates()
    again = generate_fixed_weight_candidates()

    assert len(candidates) == 35
    assert candidates == again
    assert [candidate.candidate_id for candidate in candidates] == [
        f"fixed_{index:03d}" for index in range(35)
    ]
    assert len({candidate.units for candidate in candidates}) == 35
    for candidate in candidates:
        assert all(0 <= unit <= 4 for unit in candidate.units)
        assert sum(candidate.units) == 4
        assert all(weight in {0.0, 0.25, 0.5, 0.75, 1.0} for weight in candidate.weights)
        assert sum(candidate.weights) == 1.0

    by_weights = {candidate.weights for candidate in candidates}
    assert (0.25, 0.25, 0.25, 0.25) in by_weights
    assert {
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    } <= by_weights
    assert (0.0, 1 / 3, 1 / 3, 1 / 3) not in by_weights


def test_fixed_weight_evaluation_uses_direct_weighted_logits_and_canonical_metrics():
    dataset = _small_restricted_dataset()
    analyzer = FixedWeightAnalyzer(dataset, np.array([100, 20, 5]))
    candidates = generate_fixed_weight_candidates()
    uniform = next(candidate for candidate in candidates if candidate.units == (1, 1, 1, 1))
    ce_only = next(candidate for candidate in candidates if candidate.units == (4, 0, 0, 0))

    matrix = analyzer.weight_matrix(uniform)
    assert matrix.shape == (dataset.num_samples, 4)
    assert np.array_equal(matrix, np.repeat(np.asarray(uniform.weights)[None, :], 6, axis=0))
    assert np.unique(matrix, axis=0).shape[0] == 1

    row = analyzer.evaluate_candidate(uniform)
    expected_logits = np.einsum("e,nec->nc", np.asarray(uniform.weights), dataset.logits)
    expected_predictions = expected_logits.argmax(axis=1)
    assert np.array_equal(
        expected_predictions,
        np.argmax(np.einsum("ne,nec->nc", matrix, dataset.logits), axis=1),
    )
    assert row["ordinary_accuracy"] == float(np.mean(expected_predictions == dataset.labels))
    assert row["balanced_accuracy"] == pytest.approx(
        np.mean([np.mean(expected_predictions[dataset.labels == c] == c) for c in range(3)])
    )
    assert ce_only.weights == (1.0, 0.0, 0.0, 0.0)


def test_analysis_deltas_and_joint_improvement_are_relative_to_uniform():
    analyzer = FixedWeightAnalyzer(_small_restricted_dataset(), np.array([100, 20, 5]))
    analysis = analyzer.analyze()
    uniform = analysis["uniform_baseline"]
    assert analysis["candidate_count"] == 35
    for row in analysis["candidates"]:
        assert row["delta_ba"] == pytest.approx(
            row["balanced_accuracy"] - uniform["balanced_accuracy"]
        )
        assert row["delta_head"] == pytest.approx(
            row["head_accuracy"] - uniform["head_accuracy"]
        )
        assert row["delta_medium"] == pytest.approx(
            row["medium_accuracy"] - uniform["medium_accuracy"]
        )
        assert row["delta_tail"] == pytest.approx(
            row["tail_accuracy"] - uniform["tail_accuracy"]
        )
        assert row["joint_improvement"] == (
            row["delta_ba"] > 0.0 and row["delta_tail"] > 0.0
        )


def test_pareto_frontier_preserves_non_dominated_order_and_exact_ties():
    results = [
        {"candidate_id": "a", "balanced_accuracy": 0.50, "tail_accuracy": 0.20},
        {"candidate_id": "b", "balanced_accuracy": 0.40, "tail_accuracy": 0.10},
        {"candidate_id": "c", "balanced_accuracy": 0.45, "tail_accuracy": 0.25},
        {"candidate_id": "d", "balanced_accuracy": 0.50, "tail_accuracy": 0.20},
        {"candidate_id": "e", "balanced_accuracy": 0.55, "tail_accuracy": 0.15},
        {"candidate_id": "f", "balanced_accuracy": 0.40, "tail_accuracy": 0.30},
    ]
    assert pareto_frontier(results) == ("a", "c", "d", "e", "f")


def test_uniform_baseline_verification_uses_all_required_metrics():
    actual = {
        "ordinary_accuracy": 0.5,
        "balanced_accuracy": 0.4,
        "head_accuracy": 0.6,
        "medium_accuracy": 0.3,
        "tail_accuracy": 0.1,
    }
    stored = {
        "accuracy": 0.5,
        "ba": 0.4,
        "head": 0.6,
        "medium": 0.3,
        "tail": 0.1,
    }
    verification = verify_uniform_baseline(actual, stored)
    assert verification["matches"] is True
    assert max(verification["absolute_differences"].values()) == 0.0
    with pytest.raises(Task3EError, match="uniform baseline"):
        verify_uniform_baseline({**actual, "balanced_accuracy": 0.401}, stored)


def test_restricted_dataset_rejects_inner_fold_zero_and_invalid_ordering():
    dataset = _small_restricted_dataset()
    with pytest.raises(Task3EError, match="inner-fold-0"):
        RestrictedAnalysisDataset.from_arrays(
            sample_indices=dataset.sample_indices,
            labels=dataset.labels,
            inner_fold_ids=np.array([0, 1, 2, 1, 2, 3]),
            logits=dataset.logits,
        )
    with pytest.raises(Task3EError, match="expert ordering"):
        RestrictedAnalysisDataset.from_arrays(
            sample_indices=dataset.sample_indices,
            labels=dataset.labels,
            inner_fold_ids=dataset.inner_fold_ids,
            logits=dataset.logits,
            expert_names=("LAL", "CE", "BalancedSoftmax", "Mixup"),
        )


def test_analyzer_refuses_complete_aligned_dataset_directly():
    manager = _small_manager()
    aligned = _aligned_for_manager(manager)
    with pytest.raises(Task3EError, match="RestrictedAnalysisDataset"):
        FixedWeightAnalyzer(aligned, np.ones(manager.num_classes, dtype=np.int64))


def test_from_aligned_checks_exact_membership_even_when_counts_match():
    manager = _small_manager()
    aligned = _aligned_for_manager(manager)
    restricted = RestrictedAnalysisDataset.from_aligned(
        aligned, manager, enforce_frozen_population=False
    )
    assert set(restricted.sample_indices.tolist()) == set(
        manager.outer_fold(0).router_development.fit_indices
    )
    assert not set(restricted.sample_indices.tolist()) & set(
        manager.outer_fold(0).router_development.selection_indices
    )

    corrupted_indices = aligned.sample_indices.copy()
    replacement = next(
        index
        for index in manager.outer_fold(0).router_development.selection_indices
        if index != corrupted_indices[0]
    )
    corrupted_indices[0] = replacement
    corrupted = AlignedOOFDataset(
        sample_indices=corrupted_indices,
        outer_fold_ids=aligned.outer_fold_ids,
        inner_fold_ids=aligned.inner_fold_ids,
        labels=aligned.labels,
        logits=aligned.logits,
        expert_names=aligned.expert_names,
        metadata={},
    )
    with pytest.raises(Task3EError, match="aligned OOF artifact"):
        RestrictedAnalysisDataset.from_aligned(
            corrupted, manager, enforce_frozen_population=False
        )


def test_outer_evaluation_population_is_rejected():
    manager = _small_manager()
    aligned = _aligned_for_manager(manager)
    corrupted_indices = aligned.sample_indices.copy()
    corrupted_indices[0] = manager.outer_fold(0).evaluation_indices[0]
    corrupted = AlignedOOFDataset(
        sample_indices=corrupted_indices,
        outer_fold_ids=aligned.outer_fold_ids,
        inner_fold_ids=aligned.inner_fold_ids,
        labels=aligned.labels,
        logits=aligned.logits,
        expert_names=aligned.expert_names,
        metadata={},
    )
    with pytest.raises(Task3EError, match="aligned OOF artifact"):
        RestrictedAnalysisDataset.from_aligned(
            corrupted, manager, enforce_frozen_population=False
        )


def test_corrupted_aligned_array_hash_is_rejected(tmp_path: Path):
    arrays_path = tmp_path / "aligned_oof.npz"
    np.savez_compressed(
        arrays_path,
        sample_indices=np.array([1], dtype=np.int64),
        outer_fold_ids=np.array([0], dtype=np.int64),
        inner_fold_ids=np.array([1], dtype=np.int64),
        labels=np.array([0], dtype=np.int64),
        logits=np.zeros((1, 4, 1), dtype=np.float32),
    )
    (tmp_path / "aligned_oof_metadata.json").write_text(
        '{"schema_version": "task3c_aligned_oof.v1", '
        '"arrays_sha256": "' + '0' * 64 + '", '
        '"expert_names": ["CE", "LAL", "BalancedSoftmax", "Mixup"]}'
    )
    with pytest.raises(ValueError, match="array hash"):
        AlignedOOFDataset.load(tmp_path)
