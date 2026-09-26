"""Synthetic tests for the read-only Task 3F-F diagnostics."""

from __future__ import annotations

import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

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

from scripts.task3f_feature_comparison import (  # noqa: E402
    ALPHAS,
    FEATURE_SET_CONFIDENCE,
    FEATURE_SET_FULL,
    GAMMAS,
    PERMITTED_ANALYSIS_INNER_FOLDS,
    Task3FFeatureComparisonError,
    _adaptive_model_key,
    _write_json_once,
    contribution_prediction_report,
    identify_matched_model_fits,
    pairwise_contribution_ranking,
    pearson_correlation,
    prediction_change_report,
    reproduce_predictions_from_saved_weights,
    validate_artifact_alignment,
    validate_saved_task3f_a_metadata,
    weight_group_statistics,
)


def _class_counts() -> np.ndarray:
    return np.asarray([100] * 35 + [20] * 35 + [5] * 30, dtype=np.int64)


def _model_fit_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for fold in PERMITTED_ANALYSIS_INNER_FOLDS:
        for feature_set in (FEATURE_SET_CONFIDENCE, FEATURE_SET_FULL):
            for alpha in ALPHAS:
                for gamma in GAMMAS:
                    records.append(
                        {
                            "model_key": _adaptive_model_key(
                                fold, feature_set, alpha, gamma
                            ),
                            "feature_set": feature_set,
                            "alpha": alpha,
                            "gamma": gamma,
                            "validation_inner_fold": fold,
                        }
                    )
    return records


def test_alignment_requires_exact_order_and_rejects_reserved_fold_zero():
    expected_ids = np.asarray([10, 20, 30], dtype=np.int64)
    expected_folds = np.asarray([1, 2, 3], dtype=np.int64)
    labels = np.asarray([0, 1, 2], dtype=np.int64)
    validate_artifact_alignment(
        expected_sample_indices=expected_ids,
        expected_inner_fold_ids=expected_folds,
        expected_labels=labels,
        artifact_sample_indices=expected_ids,
        artifact_inner_fold_ids=expected_folds,
        artifact_labels=labels,
    )
    with pytest.raises(Task3FFeatureComparisonError, match="inner-fold-0"):
        validate_artifact_alignment(
            expected_sample_indices=expected_ids,
            expected_inner_fold_ids=expected_folds,
            expected_labels=labels,
            artifact_sample_indices=expected_ids,
            artifact_inner_fold_ids=np.asarray([0, 2, 3]),
            artifact_labels=labels,
        )
    with pytest.raises(Task3FFeatureComparisonError, match="duplicates"):
        validate_artifact_alignment(
            expected_sample_indices=expected_ids,
            expected_inner_fold_ids=expected_folds,
            expected_labels=labels,
            artifact_sample_indices=np.asarray([10, 10, 30]),
            artifact_inner_fold_ids=expected_folds,
            artifact_labels=labels,
        )
    with pytest.raises(Task3FFeatureComparisonError, match="aligned"):
        validate_artifact_alignment(
            expected_sample_indices=expected_ids,
            expected_inner_fold_ids=expected_folds,
            expected_labels=labels,
            artifact_sample_indices=np.asarray([20, 10, 30]),
            artifact_inner_fold_ids=expected_folds,
            artifact_labels=labels,
        )


def test_matched_model_identification_rejects_missing_and_duplicate_saved_fits():
    records = _model_fit_records()
    score_ids = [str(record["model_key"]) for record in records]
    matched = identify_matched_model_fits(records, score_ids)
    assert len(matched) == len(ALPHAS) * len(GAMMAS) * 2
    assert set(matched[(FEATURE_SET_FULL, 1000.0, 1.0)]) == set(
        PERMITTED_ANALYSIS_INNER_FOLDS
    )

    with pytest.raises(Task3FFeatureComparisonError, match="missing"):
        identify_matched_model_fits(records[:-1], score_ids[:-1])
    with pytest.raises(Task3FFeatureComparisonError, match="duplicate"):
        identify_matched_model_fits(records + [records[0]], score_ids)


def test_incorrect_saved_source_metadata_is_rejected():
    config = {
        "task_identifier": "Task 3F-A",
        "schema_version": "task3f_ridge.v1",
        "expert_order": ["CE", "LAL", "BalancedSoftmax", "Mixup"],
        "analyzed_sample_count": 6507,
        "permitted_analysis_inner_folds": [1, 2, 3],
        "reserved_router_selection_inner_folds": [0],
        "data_restrictions": {
            "inner_fold_zero_used": False,
            "reserved_outer_evaluation_used": False,
            "original_cifar_test_used": False,
        },
    }
    validate_saved_task3f_a_metadata(config)
    config["expert_order"] = ["LAL", "CE", "BalancedSoftmax", "Mixup"]
    with pytest.raises(Task3FFeatureComparisonError, match="expert ordering"):
        validate_saved_task3f_a_metadata(config)


def test_contribution_metrics_and_degenerate_pearson_are_safe():
    labels = np.asarray([0, 35, 70, 71, 72, 73], dtype=np.int64)
    actual = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.5],
            [0.0, 1.0, 0.0, 0.5],
            [0.0, 1.0, 0.0, 0.5],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.5, 1.0],
            [0.0, 0.0, 0.5, 1.0],
        ]
    )
    predicted = actual.copy()
    predicted[:, 0] += 1.0
    report = contribution_prediction_report(predicted, actual, labels, _class_counts())
    assert report["tail"]["sample_count"] == 4
    assert report["tail"]["experts"]["CE"]["mse"] == pytest.approx(1.0)
    assert report["pooled"]["experts"]["CE"]["mean_actual"] == pytest.approx(0.0)
    assert pearson_correlation([1.0, 1.0], [1.0, 2.0]) is None
    assert pearson_correlation([1.0], [1.0]) is None


def test_pairwise_ranking_evaluates_rebalanced_and_mixup_directions():
    labels = np.asarray([70, 71, 72, 73], dtype=np.int64)
    actual = np.asarray(
        [
            [0.0, 2.0, 0.0, 1.0],  # LAL higher
            [0.0, 0.0, 2.0, 1.0],  # BalancedSoftmax higher
            [0.0, 0.0, 1.0, 2.0],  # Mixup higher
            [0.0, 0.0, 1.0, 2.0],  # Mixup higher
        ]
    )
    predicted = np.asarray(
        [
            [0.0, 2.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 2.0, 1.0],
            [0.0, 0.0, 1.0, 2.0],
        ]
    )
    ranking = pairwise_contribution_ranking(
        predicted, actual, labels, _class_counts()
    )
    lal = ranking["pairs"]["LAL_vs_Mixup"]
    balanced = ranking["pairs"]["BalancedSoftmax_vs_Mixup"]
    assert lal["actual_rebalanced_higher_count"] == 1
    assert lal["correct_rebalanced_on_actual_rebalanced_higher"] == 1
    assert lal["actual_mixup_higher_count"] == 3
    assert lal["correct_mixup_on_actual_mixup_higher"] == 3
    assert balanced["actual_rebalanced_higher_count"] == 1
    assert balanced["incorrect_mixup_on_actual_rebalanced_higher"] == 1
    assert balanced["pairwise_ranking_denominator_non_tied_actual"] == 4


def test_saved_weights_reproduce_predictions_without_labels():
    logits = np.asarray(
        [
            [[3.0, 0.0], [0.0, 3.0], [2.0, 1.0], [1.0, 2.0]],
            [[0.0, 3.0], [3.0, 0.0], [1.0, 2.0], [2.0, 1.0]],
        ],
        dtype=np.float64,
    )
    weights = np.asarray([[0.25, 0.25, 0.25, 0.25], [0.0, 0.5, 0.0, 0.5]])
    predictions_before = reproduce_predictions_from_saved_weights(logits, weights)
    changed_labels = np.asarray([1, 0], dtype=np.int64)
    predictions_after = reproduce_predictions_from_saved_weights(logits, weights)
    assert np.array_equal(predictions_before, predictions_after)
    assert np.array_equal(changed_labels, [1, 0])


def test_prediction_changes_and_group_weight_statistics_use_canonical_groups():
    labels = np.asarray([0, 35, 70, 71], dtype=np.int64)
    sample_ids = np.asarray([100, 101, 102, 103], dtype=np.int64)
    confidence = np.asarray([1, 0, 1, 71], dtype=np.int64)
    full = np.asarray([0, 0, 1, 3], dtype=np.int64)
    changes = prediction_change_report(
        confidence, full, labels, sample_ids, _class_counts()
    )
    assert changes["by_group"]["head"]["full_corrected_confidence_missed"] == 1
    assert changes["by_group"]["tail"]["confidence_corrected_full_missed"] == 1
    assert changes["tail_prediction_records"]["gained"] == []
    assert changes["tail_prediction_records"]["lost"] == [
        {"sample_id": 103, "true_class": 71}
    ]

    weights = np.repeat(np.asarray([[0.1, 0.2, 0.3, 0.4]]), len(labels), axis=0)
    weight_report = weight_group_statistics(weights, labels, _class_counts())
    assert weight_report["head"]["sample_count"] == 1
    assert weight_report["tail"]["sample_count"] == 2
    assert weight_report["tail"]["highest_weight_count"]["Mixup"] == 2


def test_write_once_protects_incompatible_outputs(tmp_path):
    output = tmp_path / "diagnostic.json"
    _write_json_once(output, {"value": 1})
    _write_json_once(output, {"value": 1})
    with pytest.raises(Task3FFeatureComparisonError, match="overwrite"):
        _write_json_once(output, {"value": 2})
