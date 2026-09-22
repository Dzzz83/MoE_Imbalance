"""Synthetic tests for the Task 3F-B Mixup preference diagnostics."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.task3f_mixup_diagnostics import (  # noqa: E402
    Task3FBDiagnosticError,
    compute_group_statistics,
    decompose_ridge_scores,
    renormalize_mixup_weights,
    tail_gain_loss_accounting,
    validate_artifact_alignment,
)


def _class_counts() -> np.ndarray:
    # Classes 0/1 are Head, class 2 is Medium, and class 3 is Tail.
    return np.array([100, 100, 20, 5], dtype=np.int64)


def test_alignment_rejects_reserved_fold_and_mismatched_sample_ids():
    sample_ids = np.array([10, 11, 12], dtype=np.int64)
    folds = np.array([1, 2, 3], dtype=np.int64)
    labels = np.array([0, 1, 2], dtype=np.int64)
    validate_artifact_alignment(
        expected_sample_indices=sample_ids,
        expected_inner_fold_ids=folds,
        expected_labels=labels,
        artifact_sample_indices=sample_ids.copy(),
        artifact_inner_fold_ids=folds.copy(),
        artifact_labels=labels.copy(),
    )

    with pytest.raises(Task3FBDiagnosticError, match="reserved inner-fold-0"):
        validate_artifact_alignment(
            expected_sample_indices=sample_ids,
            expected_inner_fold_ids=folds,
            expected_labels=labels,
            artifact_sample_indices=sample_ids,
            artifact_inner_fold_ids=np.array([0, 2, 3]),
            artifact_labels=labels,
        )

    with pytest.raises(Task3FBDiagnosticError, match="sample IDs"):
        validate_artifact_alignment(
            expected_sample_indices=sample_ids,
            expected_inner_fold_ids=folds,
            expected_labels=labels,
            artifact_sample_indices=np.array([10, 12, 13]),
            artifact_inner_fold_ids=folds,
            artifact_labels=labels,
        )


def test_group_statistics_use_canonical_groups_and_report_mixup_tail_weight():
    labels = np.array([0, 1, 2, 3], dtype=np.int64)
    weights = np.array(
        [
            [0.4, 0.2, 0.2, 0.2],
            [0.2, 0.3, 0.1, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.2, 0.5],
        ],
        dtype=np.float64,
    )
    predicted_scores = weights - 0.25
    actual_targets = predicted_scores + 0.1

    report = compute_group_statistics(
        labels,
        weights,
        predicted_scores,
        actual_targets,
        _class_counts(),
    )

    assert report["head"]["sample_count"] == 2
    assert report["medium"]["sample_count"] == 1
    assert report["tail"]["sample_count"] == 1
    assert report["tail"]["highest_weight_fraction"]["Mixup"] == pytest.approx(1.0)
    assert report["tail"]["mean_weight"]["Mixup"] == pytest.approx(0.5)
    assert report["tail"]["mean_actual_target"]["Mixup"] == pytest.approx(0.35)


def test_ridge_score_decomposition_exposes_standardized_feature_terms():
    features = np.array([[3.0, 5.0], [1.0, 1.0]], dtype=np.float64)
    intercept = np.array([10.0, -2.0, 4.0, 8.0], dtype=np.float64)
    coefficients = np.array(
        [[2.0, -1.0], [0.5, 3.0], [1.0, 2.0], [-2.0, 0.5]],
        dtype=np.float64,
    )
    scaler_mean = np.array([1.0, 3.0], dtype=np.float64)
    scaler_scale = np.array([2.0, 1.0], dtype=np.float64)

    decomposition = decompose_ridge_scores(
        features,
        intercept,
        coefficients,
        scaler_mean,
        scaler_scale,
    )

    assert np.allclose(decomposition["standardized_features"], [[1.0, 2.0], [0.0, -2.0]])
    assert np.allclose(
        decomposition["feature_term"],
        [[0.0, 6.5, 5.0, -1.0], [2.0, -6.0, -4.0, -1.0]],
    )
    assert np.allclose(
        decomposition["predicted_scores"],
        [[10.0, 4.5, 9.0, 7.0], [12.0, -8.0, 0.0, 7.0]],
    )
    assert np.allclose(
        decomposition["predicted_scores"],
        decomposition["intercept"] + decomposition["feature_term"],
    )


def test_tail_gain_loss_accounting_keeps_per_image_expert_and_weight_records():
    sample_ids = np.array([100, 101, 102, 103], dtype=np.int64)
    labels = np.array([0, 1, 3, 3], dtype=np.int64)
    uniform = np.array([0, 0, 0, 3], dtype=np.int64)
    ridge = np.array([0, 2, 3, 1], dtype=np.int64)
    expert_predictions = np.array(
        [[0, 0, 1, 0], [1, 1, 2, 1], [2, 2, 3, 2], [3, 3, 2, 3]],
        dtype=np.int64,
    )
    weights = np.full((4, 4), 0.25, dtype=np.float64)

    report = tail_gain_loss_accounting(
        sample_ids,
        labels,
        uniform,
        ridge,
        expert_predictions,
        weights,
        _class_counts(),
    )

    assert report["tail_sample_count"] == 2
    assert report["gained_count"] == 1
    assert report["lost_count"] == 1
    assert report["both_correct_count"] == 0
    assert report["both_wrong_count"] == 0
    assert report["gained_cases"][0]["sample_id"] == 102
    assert report["gained_cases"][0]["expert_predictions"] == [2, 2, 3, 2]
    assert report["lost_cases"][0]["sample_id"] == 103


def test_mixup_weight_factor_one_preserves_predictions_and_zero_is_valid():
    weights = np.array(
        [[0.2, 0.3, 0.1, 0.4], [0.25, 0.25, 0.25, 0.25]],
        dtype=np.float64,
    )
    same = renormalize_mixup_weights(weights, 1.0)
    reduced = renormalize_mixup_weights(weights, 0.0)

    assert np.array_equal(same, weights)
    assert np.allclose(reduced[:, 3], 0.0)
    assert np.allclose(reduced.sum(axis=1), 1.0)

    with pytest.raises(Task3FBDiagnosticError, match="non-negative"):
        renormalize_mixup_weights(weights, -0.25)
