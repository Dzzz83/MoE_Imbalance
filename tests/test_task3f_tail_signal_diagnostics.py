"""Synthetic tests for Task 3F-C Tail-signal diagnostics."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.task3f_tail_signal_diagnostics import (  # noqa: E402
    Task3FCTailDiagnosticError,
    _write_json_once,
    assign_class_groups,
    confidence_diagnostics,
    disagreement_diagnostics,
    predicted_group_signal_report,
    ridge_weight_pattern_diagnostics,
    validate_signal_alignment,
)


def _class_counts() -> np.ndarray:
    # Class 0 is Head, class 1 is Medium, and classes 2/3 are Tail.
    return np.array([100, 20, 5, 5], dtype=np.int64)


def _predictions() -> np.ndarray:
    return np.array(
        [
            [0, 0, 0, 0],  # all agree on Head
            [2, 2, 2, 1],  # LAL/BS share correct Tail, Mixup disagrees
            [3, 3, 1, 3],  # CE/LAL/Mixup share Tail, BS disagrees
            [1, 1, 1, 2],  # LAL/BS share correct Medium, Mixup disagrees
            [2, 1, 2, 2],  # Mixup/BS correct Tail, LAL disagrees
            [0, 0, 1, 0],  # mostly Head, one rebalanced disagreement
        ],
        dtype=np.int64,
    )


def test_alignment_checks_reserved_fold_and_expert_order_without_mutation():
    sample_ids = np.array([10, 11, 12], dtype=np.int64)
    folds = np.array([1, 2, 3], dtype=np.int64)
    labels = np.array([0, 1, 2], dtype=np.int64)
    predictions = np.zeros((3, 4), dtype=np.int64)
    confidences = np.full((3, 4), 0.5, dtype=np.float64)
    weights = np.full((3, 4), 0.25, dtype=np.float64)
    original_predictions = predictions.copy()

    validate_signal_alignment(
        expected_sample_indices=sample_ids,
        expected_inner_fold_ids=folds,
        expected_labels=labels,
        artifact_sample_indices=sample_ids,
        artifact_inner_fold_ids=folds,
        artifact_labels=labels,
        expert_predictions=predictions,
        confidences=confidences,
        weights=weights,
        expert_order=("CE", "LAL", "BalancedSoftmax", "Mixup"),
    )
    assert np.array_equal(predictions, original_predictions)

    with pytest.raises(Task3FCTailDiagnosticError, match="reserved inner-fold-0"):
        validate_signal_alignment(
            expected_sample_indices=sample_ids,
            expected_inner_fold_ids=folds,
            expected_labels=labels,
            artifact_sample_indices=sample_ids,
            artifact_inner_fold_ids=np.array([0, 2, 3]),
            artifact_labels=labels,
            expert_predictions=predictions,
            confidences=confidences,
            weights=weights,
            expert_order=("CE", "LAL", "BalancedSoftmax", "Mixup"),
        )

    with pytest.raises(Task3FCTailDiagnosticError, match="expert order"):
        validate_signal_alignment(
            expected_sample_indices=sample_ids,
            expected_inner_fold_ids=folds,
            expected_labels=labels,
            artifact_sample_indices=sample_ids,
            artifact_inner_fold_ids=folds,
            artifact_labels=labels,
            expert_predictions=predictions,
            confidences=confidences,
            weights=weights,
            expert_order=("LAL", "CE", "BalancedSoftmax", "Mixup"),
        )


def test_assign_class_groups_uses_canonical_boundaries():
    groups = assign_class_groups(np.array([0, 1, 2, 3]), _class_counts())
    assert groups.tolist() == ["head", "medium", "tail", "tail"]


def test_confidence_report_separates_raw_confidence_from_correctness():
    labels = np.array([0, 2, 2, 3, 1, 0], dtype=np.int64)
    predictions = _predictions()
    confidences = np.array(
        [
            [0.90, 0.80, 0.70, 0.60],
            [0.60, 0.90, 0.85, 0.95],
            [0.70, 0.80, 0.65, 0.92],
            [0.70, 0.75, 0.80, 0.90],
            [0.80, 0.65, 0.88, 0.90],
            [0.95, 0.85, 0.80, 0.75],
        ],
        dtype=np.float64,
    )

    report = confidence_diagnostics(
        confidences, predictions, labels, _class_counts()
    )

    assert report["by_true_group"]["tail"]["sample_count"] == 3
    assert report["by_true_group"]["tail"]["per_expert"]["Mixup"]["median"] == pytest.approx(0.92)
    pair = report["tail_pairwise_comparisons"]["LAL"]
    assert pair["rebalanced_correct_mixup_wrong"]["count"] == 1
    assert "correct_confidence" in report["by_true_group"]["tail"]["per_expert"]["LAL"]
    assert "incorrect_confidence" in report["by_true_group"]["tail"]["per_expert"]["Mixup"]


def test_disagreement_report_identifies_shared_rebalanced_prediction():
    labels = np.array([0, 2, 3, 1, 2, 0], dtype=np.int64)
    report = disagreement_diagnostics(_predictions(), labels, _class_counts())

    assert report["by_true_group"]["head"]["sample_count"] == 2
    assert report["by_true_group"]["tail"]["sample_count"] == 3
    shared = report["patterns"]["lal_balanced_agree_mixup_disagree"]
    assert shared["count"] == 2
    assert shared["shared_prediction_correct_count"] == 2
    assert shared["by_true_group"]["tail"]["count"] == 1
    assert report["by_true_group"]["tail"]["distinct_prediction_counts"]["1"] >= 0


@pytest.mark.parametrize("diagnostic", [confidence_diagnostics, disagreement_diagnostics])
def test_prediction_diagnostics_reject_out_of_range_class_ids(diagnostic):
    labels = np.array([0, 2, 3, 1, 2, 0], dtype=np.int64)
    predictions = _predictions().copy()
    predictions[0, 0] = 4
    if diagnostic is confidence_diagnostics:
        confidences = np.full((len(labels), 4), 0.5, dtype=np.float64)
        with pytest.raises(Task3FCTailDiagnosticError, match="outside class_counts"):
            diagnostic(confidences, predictions, labels, _class_counts())
    else:
        with pytest.raises(Task3FCTailDiagnosticError, match="outside class_counts"):
            diagnostic(predictions, labels, _class_counts())


def test_predicted_group_signal_does_not_change_when_labels_change():
    predictions = _predictions()
    labels = np.array([0, 2, 3, 1, 2, 0], dtype=np.int64)
    changed_labels = np.array([3, 3, 0, 0, 1, 2], dtype=np.int64)

    first = predicted_group_signal_report(predictions, labels, _class_counts())
    second = predicted_group_signal_report(
        predictions, changed_labels, _class_counts()
    )

    assert first["signal_definition"] == second["signal_definition"]
    assert first["per_expert"]["LAL"]["predicted_group_counts"] == second["per_expert"]["LAL"]["predicted_group_counts"]
    assert first["per_expert"]["LAL"]["predicted_tail_count"] == 2
    assert first["per_expert"]["LAL"]["actual_tail_recall"] != second["per_expert"]["LAL"]["actual_tail_recall"]


def test_ridge_weight_patterns_report_conditioned_weights_and_correctness():
    labels = np.array([0, 2, 3, 1, 2, 0], dtype=np.int64)
    weights = np.array(
        [
            [0.25, 0.25, 0.25, 0.25],
            [0.10, 0.45, 0.35, 0.10],
            [0.10, 0.40, 0.10, 0.40],
            [0.20, 0.35, 0.35, 0.10],
            [0.10, 0.20, 0.45, 0.25],
            [0.30, 0.20, 0.20, 0.30],
        ],
        dtype=np.float64,
    )
    report = ridge_weight_pattern_diagnostics(
        _predictions(), labels, weights, _class_counts()
    )

    shared_tail = report["patterns"]["lal_balanced_agree_on_tail_class"]
    assert shared_tail["count"] == 1
    assert shared_tail["mean_weight"]["LAL"] == pytest.approx(0.45)
    assert "by_expert_predicted_group" in report
    assert report["tail_correctness_patterns"]["LAL_correct_Mixup_wrong"]["count"] >= 0


def test_write_once_guard_preserves_existing_artifact(tmp_path):
    path = tmp_path / "existing.json"
    path.write_text('{"original": true}\n')

    with pytest.raises(Task3FCTailDiagnosticError, match="refusing to overwrite"):
        _write_json_once(path, {"replacement": True})

    assert path.read_text() == '{"original": true}\n'
