"""Synthetic tests for Task 3F-D combined signal diagnostics."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.task3f_combined_signal_diagnostics import (  # noqa: E402
    SIGNAL_COMBINATION_IDS,
    Task3FDCombinedSignalDiagnosticError,
    build_all_signal_masks,
    build_inference_time_signals,
    build_signal_combinations,
    compare_combined_signals,
    correctness_opportunity_diagnostics,
    detect_equivalent_signal_masks,
    ridge_weight_diagnostics,
    tail_detection_diagnostics,
)


def _class_counts() -> np.ndarray:
    # Class 0 is Head, class 1 is Medium, and classes 2/3 are Tail.
    return np.array([100, 20, 5, 5], dtype=np.int64)


def _predictions() -> np.ndarray:
    return np.array(
        [
            [0, 0, 0, 0],  # A true, B false, C false
            [2, 2, 2, 1],  # A true, B false, C true
            [3, 3, 1, 3],  # A false, B false, C false
            [1, 1, 1, 2],  # A true, B false, C false
            [2, 1, 2, 2],  # A false, B false, C false
            [0, 0, 1, 0],  # A false, B false, C false
        ],
        dtype=np.int64,
    )


def _confidences() -> np.ndarray:
    return np.array(
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


def test_signal_definitions_use_predictions_and_confidences_only():
    masks = build_inference_time_signals(
        _predictions(), _confidences(), _class_counts()
    )

    assert masks["A"].tolist() == [True, True, False, True, False, False]
    assert masks["B"].tolist() == [False, True, False, True, False, False]
    assert masks["C"].tolist() == [False, True, False, False, False, False]
    assert masks["D"].tolist() == [True, False, False, False, False, True]

    # A label change cannot affect any signal mask because labels are not an
    # argument to the construction seam.
    changed_labels = np.array([3, 3, 0, 0, 1, 2], dtype=np.int64)
    del changed_labels
    repeated = build_inference_time_signals(
        _predictions(), _confidences(), _class_counts()
    )
    assert all(mask.dtype == np.bool_ for mask in masks.values())
    assert all(np.array_equal(masks[name], repeated[name]) for name in masks)


def test_all_fifteen_combinations_are_frozen_conjunctions():
    individual = build_inference_time_signals(
        _predictions(), _confidences(), _class_counts()
    )
    combinations = build_signal_combinations(individual)

    assert tuple(combinations) == SIGNAL_COMBINATION_IDS
    assert len(combinations) == 15
    assert np.array_equal(combinations["AB"], individual["A"] & individual["B"])
    assert np.array_equal(
        combinations["ABCD"],
        individual["A"] & individual["B"] & individual["C"] & individual["D"],
    )


def test_equivalent_signal_masks_are_grouped():
    masks = {
        "A": np.array([True, False, True]),
        "AB": np.array([True, False, True]),
        "C": np.array([False, True, False]),
    }
    assert detect_equivalent_signal_masks(masks) == [["A", "AB"], ["C"]]


def test_tail_precision_recall_and_empty_group_are_counted():
    predictions = _predictions()
    masks = build_all_signal_masks(predictions, _confidences(), _class_counts())
    labels = np.array([0, 2, 3, 1, 2, 0], dtype=np.int64)
    report = tail_detection_diagnostics(masks, labels, _class_counts())

    assert report["actual_tail_count"] == 3
    a = report["reports"]["A"]
    assert a["selected_count"] == 3
    assert a["selected_true_group_counts"]["tail"] == 1
    assert a["tail_precision"] == pytest.approx(1 / 3)
    assert a["tail_recall"] == pytest.approx(1 / 3)
    empty = report["reports"]["ABCD"]
    assert empty["selected_count"] == 0
    assert empty["tail_precision"] is None
    assert empty["tail_recall"] == 0.0


def test_correctness_diagnostics_are_aligned_and_grouped():
    predictions = _predictions()
    masks = build_all_signal_masks(predictions, _confidences(), _class_counts())
    labels = np.array([0, 2, 3, 1, 2, 0], dtype=np.int64)
    report = correctness_opportunity_diagnostics(
        masks, predictions, labels, _class_counts()
    )

    a = report["reports"]["A"]
    assert a["selected_count"] == 3
    assert a["overall"]["LAL_correct_Mixup_wrong"]["count"] == 2
    assert a["overall"]["BalancedSoftmax_correct_Mixup_wrong"]["count"] == 2
    assert a["overall"]["either_rebalanced_correct_Mixup_wrong"]["count"] == 2
    assert a["by_true_group"]["tail"]["selected_count"] == 1
    assert a["by_true_group"]["tail"]["outcomes"][
        "either_rebalanced_correct_Mixup_wrong"
    ]["count"] == 1

    with pytest.raises(Task3FDCombinedSignalDiagnosticError):
        correctness_opportunity_diagnostics(
            masks, predictions, labels[:-1], _class_counts()
        )


def test_ridge_weight_diagnostics_do_not_mutate_weights():
    masks = build_all_signal_masks(_predictions(), _confidences(), _class_counts())
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
    original = weights.copy()
    report = ridge_weight_diagnostics(masks, weights)

    assert np.array_equal(weights, original)
    assert report["overall"]["selected_count"] == len(weights)
    assert report["reports"]["A"]["selected_count"] == 3
    assert report["reports"]["A"]["mean_weight"]["LAL"] == pytest.approx(
        (0.25 + 0.45 + 0.35) / 3
    )


def test_comparisons_preserve_subset_relationships_and_counts():
    predictions = _predictions()
    masks = build_all_signal_masks(predictions, _confidences(), _class_counts())
    labels = np.array([0, 2, 3, 1, 2, 0], dtype=np.int64)
    tail = tail_detection_diagnostics(masks, labels, _class_counts())
    correctness = correctness_opportunity_diagnostics(
        masks, predictions, labels, _class_counts()
    )
    comparisons = compare_combined_signals(masks, tail, correctness)

    ab = comparisons["comparisons"]["AB"]
    assert ab["all_constituent_subset_checks_pass"] is True
    assert ab["constituents"]["A"]["overlap_count"] == masks["AB"].sum()
    assert ab["constituents"]["A"]["excluded_from_constituent_count"] == (
        masks["A"].sum() - masks["AB"].sum()
    )


def test_bad_prediction_shape_is_rejected_without_expert_reordering():
    with pytest.raises(Task3FDCombinedSignalDiagnosticError, match="shape.*4"):
        build_inference_time_signals(
            np.zeros((2, 3), dtype=np.int64),
            np.zeros((2, 4), dtype=np.float64),
            _class_counts(),
        )
