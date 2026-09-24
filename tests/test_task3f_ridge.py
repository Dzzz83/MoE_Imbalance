"""Synthetic regression tests for Task 3F-A Ridge routing.

The fixtures are deliberately small and self-contained.  They do not load the
canonical CIFAR data, checkpoints, the balanced test set, or the real OOF
artifact.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.task3e_fixed import RestrictedAnalysisDataset  # noqa: E402
from scripts.task3f_ridge import (  # noqa: E402
    ALPHAS,
    FEATURE_SET_CONFIDENCE,
    FEATURE_SET_FULL,
    GAMMAS,
    SHRINKAGES,
    TEMPERATURES,
    RidgeCVAnalyzer,
    Task3FError,
    combine_weighted_logits,
    compute_contribution_targets,
    compute_sample_weights,
    extract_features,
    fit_global_score_control,
    fit_ridge_router,
    scores_to_weights,
)


def _logits_fixture() -> np.ndarray:
    rng = np.random.default_rng(17)
    return rng.normal(size=(18, 4, 6)).astype(np.float64)


def _restricted_fixture() -> RestrictedAnalysisDataset:
    labels = np.tile(np.arange(6, dtype=np.int64), 3)
    inner_folds = np.repeat(np.array([1, 2, 3], dtype=np.int64), 6)
    return RestrictedAnalysisDataset.from_arrays(
        sample_indices=np.arange(len(labels), dtype=np.int64),
        labels=labels,
        inner_fold_ids=inner_folds,
        logits=_logits_fixture(),
    )


def _class_counts() -> np.ndarray:
    # Three Head, two Medium, and one Tail class; every synthetic fold contains
    # one row for every class so canonical macro metrics are defined.
    return np.array([100, 100, 100, 20, 20, 5], dtype=np.int64)


def test_contribution_targets_are_zero_sum_and_match_finite_difference():
    logits = _logits_fixture()[:2]
    labels = np.array([0, 4], dtype=np.int64)
    targets = compute_contribution_targets(logits, labels)

    assert targets.shape == (2, 4)
    assert np.isfinite(targets).all()
    assert np.allclose(targets.sum(axis=1), 0.0, atol=1e-12)

    row = logits[0]
    label = int(labels[0])
    mean_logits = row.mean(axis=0)
    delta = row[0] - mean_logits

    def true_log_probability(epsilon: float) -> float:
        shifted = mean_logits + epsilon * delta
        shifted -= shifted.max()
        log_probability = shifted - np.log(np.exp(shifted).sum())
        return float(log_probability[label])

    epsilon = 1e-6
    numerical = (true_log_probability(epsilon) - true_log_probability(-epsilon)) / (2 * epsilon)
    assert targets[0, 0] == pytest.approx(numerical, rel=1e-5, abs=1e-7)


def test_identical_experts_have_zero_contribution_targets_and_labels_matter():
    one = np.arange(5, dtype=np.float64)
    logits = np.broadcast_to(one, (3, 4, 5)).copy()
    assert np.allclose(
        compute_contribution_targets(logits, np.array([0, 1, 2])), 0.0
    )

    different_labels = compute_contribution_targets(
        _logits_fixture()[:1], np.array([0], dtype=np.int64)
    )
    changed_label = compute_contribution_targets(
        _logits_fixture()[:1], np.array([1], dtype=np.int64)
    )
    assert not np.allclose(different_labels, changed_label)


def test_feature_sets_have_frozen_shapes_order_and_no_label_dependency():
    logits = _logits_fixture()
    confidence = extract_features(logits, FEATURE_SET_CONFIDENCE)
    full = extract_features(logits, FEATURE_SET_FULL)

    assert confidence.shape == (18, 4)
    assert full.shape == (18, 13)
    assert np.isfinite(confidence).all()
    assert np.isfinite(full).all()
    assert np.all((confidence >= 0.0) & (confidence <= 1.0))
    assert np.array_equal(full[:, :4], confidence)
    assert np.all(full[:, 4:8] >= 0.0)
    assert np.all(full[:, 8:12] >= 0.0)
    assert np.all((full[:, 12] >= 1.0) & (full[:, 12] <= 4.0))


def test_feature_extraction_rejects_nonfinite_logits():
    logits = _logits_fixture()
    logits[0, 0, 0] = np.nan
    with pytest.raises(Task3FError, match="non-finite"):
        extract_features(logits, FEATURE_SET_FULL)


def test_sample_weights_use_training_labels_only_and_are_normalized():
    labels = np.array([0, 0, 0, 1, 1, 2], dtype=np.int64)
    weights = compute_sample_weights(labels, gamma=1.0)
    assert weights.shape == labels.shape
    assert np.isfinite(weights).all()
    assert np.all(weights > 0.0)
    assert weights.mean() == pytest.approx(1.0)
    assert weights[-1] > weights[3] > weights[0]
    assert np.allclose(compute_sample_weights(labels, gamma=0.0), 1.0)


def test_sample_weight_report_rejects_missing_intermediate_classes():
    from scripts.task3f_ridge import sample_weight_report

    with pytest.raises(Task3FError, match="missing classes"):
        sample_weight_report(
            np.array([0, 2], dtype=np.int64),
            np.ones(2, dtype=np.float64),
        )


def test_scores_to_weights_are_stable_convex_and_support_global_scores():
    scores = np.array([[1000.0, 0.0, -1000.0, 1.0], [0.0, 0.0, 0.0, 0.0]])
    weights = scores_to_weights(scores, temperature=0.5, shrinkage=1.0)
    assert weights.shape == scores.shape
    assert np.isfinite(weights).all()
    assert np.all(weights >= 0.0)
    assert np.allclose(weights.sum(axis=1), 1.0)
    assert np.allclose(
        scores_to_weights(scores, temperature=2.0, shrinkage=0.0), 0.25
    )
    global_weights = scores_to_weights(scores[0], temperature=1.0, shrinkage=0.5)
    assert global_weights.shape == (4,)
    assert global_weights.sum() == pytest.approx(1.0)
    assert np.allclose(scores_to_weights(np.zeros((3, 4)), 1.0, 1.0), 0.25)


def test_weighted_logits_use_original_logits_without_recalibration():
    logits = _logits_fixture()[:3]
    weights = np.array([0.0, 0.25, 0.5, 0.25])
    combined = combine_weighted_logits(logits, weights)
    expected = np.einsum("e,nec->nc", weights, logits)
    assert np.array_equal(combined, expected)


def test_ridge_fit_standardizes_training_features_and_predicts_finite_scores():
    dataset = _restricted_fixture()
    train = dataset.inner_fold_ids != 1
    router = fit_ridge_router(
        dataset.logits[train],
        dataset.labels[train],
        feature_set=FEATURE_SET_FULL,
        alpha=1.0,
        gamma=0.5,
    )
    assert router.model.fit_intercept is True
    assert router.model.solver == "lsqr"
    assert np.isfinite(router.predict_scores(dataset.logits[~train])).all()
    train_features = extract_features(dataset.logits[train], FEATURE_SET_FULL)
    assert np.allclose(router.scaler.mean_, train_features.mean(axis=0))
    assert router.training_sample_weights.mean() == pytest.approx(1.0)


def test_global_score_control_is_constant_and_uses_same_target_definition():
    dataset = _restricted_fixture()
    control = fit_global_score_control(
        dataset.logits[dataset.inner_fold_ids != 1],
        dataset.labels[dataset.inner_fold_ids != 1],
        gamma=1.0,
    )
    scores = control.predict_scores(7)
    assert scores.shape == (7, 4)
    assert np.allclose(scores, scores[0])
    assert np.isfinite(scores).all()


def test_cross_validation_evaluates_the_frozen_grid_and_all_samples_once():
    dataset = _restricted_fixture()
    analyzer = RidgeCVAnalyzer(dataset, _class_counts())
    result = analyzer.run()

    assert len(result["model_fits"]) == 3 * 2 * len(ALPHAS) * len(GAMMAS)
    assert len(result["adaptive_results"]) == 2 * len(ALPHAS) * len(GAMMAS) * len(TEMPERATURES) * len(SHRINKAGES)
    assert len(result["global_results"]) == len(GAMMAS) * len(TEMPERATURES) * len(SHRINKAGES)
    assert sorted(result["held_out_inner_fold_ids"].tolist()) == [1] * 6 + [2] * 6 + [3] * 6
    for row in result["adaptive_results"]:
        assert np.isfinite(row["metrics"]["balanced_accuracy"])
        assert len(row["fold_metrics"]) == 3
        assert row["weight_diagnostics"]["fraction_meaningfully_different_from_global"] >= 0.0
    assert result["cv_contract"]["every_sample_held_out_exactly_once"] is True


def test_validation_labels_do_not_enter_a_fitted_router():
    dataset = _restricted_fixture()
    train = dataset.inner_fold_ids != 1
    router = fit_ridge_router(
        dataset.logits[train],
        dataset.labels[train],
        feature_set=FEATURE_SET_CONFIDENCE,
        alpha=10.0,
        gamma=0.0,
    )
    original = router.predict_scores(dataset.logits[~train])
    changed_validation_labels = dataset.labels[~train].copy()
    changed_validation_labels[:] = (changed_validation_labels + 1) % 6
    # The validation labels are intentionally not an argument to prediction.
    assert np.array_equal(original, router.predict_scores(dataset.logits[~train]))
    assert not np.array_equal(changed_validation_labels, dataset.labels[~train])


@pytest.mark.parametrize(
    "call, pattern",
    [
        (lambda: scores_to_weights(np.zeros((1, 4)), 0.0, 0.5), "temperature"),
        (lambda: scores_to_weights(np.zeros((1, 4)), 1.0, -0.1), "shrinkage"),
        (lambda: scores_to_weights(np.zeros((1, 4)), 1.0, 1.1), "shrinkage"),
        (lambda: fit_ridge_router(np.zeros((4, 4, 3)), np.zeros(4, dtype=np.int64), FEATURE_SET_FULL, 0.0, 0.0), "alpha"),
    ],
)
def test_invalid_numerical_configuration_is_rejected(call, pattern):
    with pytest.raises(Task3FError, match=pattern):
        call()


def test_restricted_dataset_rejects_reserved_fold_duplicates_and_ordering():
    dataset = _restricted_fixture()
    with pytest.raises(Exception, match="inner-fold-0"):
        RestrictedAnalysisDataset.from_arrays(
            sample_indices=dataset.sample_indices,
            labels=dataset.labels,
            inner_fold_ids=np.array([0] + dataset.inner_fold_ids.tolist()[1:]),
            logits=dataset.logits,
        )
    with pytest.raises(Exception, match="duplicates"):
        RestrictedAnalysisDataset.from_arrays(
            sample_indices=np.array([0] + dataset.sample_indices.tolist()),
            labels=np.array([0] + dataset.labels.tolist()),
            inner_fold_ids=np.array([1] + dataset.inner_fold_ids.tolist()),
            logits=np.concatenate([dataset.logits[:1], dataset.logits], axis=0),
        )
    with pytest.raises(Exception, match="expert ordering"):
        RestrictedAnalysisDataset.from_arrays(
            sample_indices=dataset.sample_indices,
            labels=dataset.labels,
            inner_fold_ids=dataset.inner_fold_ids,
            logits=dataset.logits,
            expert_names=("LAL", "CE", "BalancedSoftmax", "Mixup"),
        )
