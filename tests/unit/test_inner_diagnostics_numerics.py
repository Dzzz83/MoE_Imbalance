"""Synthetic tests for array-only inner diagnostic numerics."""

from __future__ import annotations

import json

import numpy as np
import pytest

from expert_method.diagnostics.complementarity import (
    analyze_complementarity,
    classify_predictions,
)
from expert_method.diagnostics.contracts import DiagnosticsError, EXPERT_NAMES
from expert_method.diagnostics.router import analyze_router_weights
from expert_method.diagnostics.sinkhorn import sinkhorn_stage_diagnostics
from expert_method.diagnostics.stability import compare_profiles, selective_gate
from scripts.ridge_sinkhorn_ot import apply_log_bias


def _logits_for_predictions(predictions: np.ndarray, num_classes: int) -> np.ndarray:
    predictions = np.asarray(predictions, dtype=np.int64)
    logits = np.zeros((*predictions.shape, num_classes), dtype=np.float64)
    rows, experts = np.indices(predictions.shape)
    logits[rows, experts, predictions] = 1.0
    return logits


def test_classifier_preserves_argmax_ties_and_marks_absent_classes_null():
    # The first logit class wins a tie, while missing classes do not receive
    # zero recall in a macro denominator.
    logits = np.zeros((2, 4, 3), dtype=np.float64)
    labels = np.array([0, 2])
    predictions = logits.argmax(axis=2)[:, 0]
    report = classify_predictions(labels, predictions, np.array([100, 20, 5]))

    assert predictions.tolist() == [0, 0]
    assert report["ordinary_accuracy"] == 0.5
    assert report["balanced_accuracy"] is None
    assert report["missing_class_ids"] == [1]
    assert report["head_accuracy"] == 1.0
    assert report["medium_accuracy"] is None
    assert report["tail_accuracy"] == 0.0
    assert report["per_class"][1]["recall"] is None


def test_complementarity_reports_pairwise_contingency_oracle_ties_and_frequency():
    labels = np.array([0, 0, 1, 2])
    predictions = np.array(
        [
            [0, 1, 1, 2],
            [1, 0, 0, 1],
            [1, 0, 1, 2],
            [2, 2, 0, 2],
        ],
        dtype=np.int64,
    )
    report = analyze_complementarity(
        _logits_for_predictions(predictions, 3),
        labels,
        np.array([100, 20, 5]),
    )

    ce_lal = next(row for row in report["pairwise"] if row["pair"] == "CE|LAL")
    assert ce_lal["correctness_contingency"] == [[1, 2], [1, 0]]
    assert ce_lal["a_only_correct_count"] == 2
    assert ce_lal["b_only_correct_count"] == 1
    assert sum(ce_lal["correctness_contingency"][i][j] for i in range(2) for j in range(2)) == 4

    agreement = report["expert_pairwise_agreement_counts"]
    disagreement = report["expert_pairwise_disagreement_counts"]
    assert len(agreement) == len(disagreement) == 4
    assert agreement[0][0] == 4
    assert disagreement[0][0] == 0
    assert agreement[0][1] == 1
    assert disagreement[0][1] == 3

    class_two = report["per_class"][2]
    assert class_two["strongest_experts"] == ["CE", "LAL", "Mixup"]
    assert class_two["best_second_recall_gap"] == 0.0
    assert class_two["sample_with_any_prediction_disagreement_fraction"] == 1.0

    oracle_gap = report["summary"]["hard_oracle_gaps_vs_strongest_individual"]
    assert oracle_gap["ordinary_accuracy"]["oracle"] == 1.0
    assert oracle_gap["ordinary_accuracy"]["strongest_individual"] == 0.75
    assert oracle_gap["ordinary_accuracy"]["oracle_minus_strongest_individual"] == 0.25
    assert len(report["frequency_associations"]) == 10
    assert all("spearman_rho" in row for row in report["frequency_associations"])
    json.dumps(report, allow_nan=False)


def test_router_profiles_include_normalized_entropy_quantiles_and_probability_predictions():
    logits = np.zeros((3, 4, 3), dtype=np.float64)
    logits[:, 0, 0] = 100.0
    logits[:, 1:, 1] = 1.0
    labels = np.ones(3, dtype=np.int64)
    uniform = np.full((3, 4), 0.25, dtype=np.float64)

    weighted_logit_prediction = np.einsum("ne,nec->nc", uniform, logits).argmax(axis=1)
    probabilities = np.exp(logits - logits.max(axis=2, keepdims=True))
    probabilities /= probabilities.sum(axis=2, keepdims=True)
    weighted_probability_prediction = (uniform[:, :, None] * probabilities).sum(axis=1).argmax(axis=1)
    assert weighted_logit_prediction.tolist() == [0, 0, 0]
    assert weighted_probability_prediction.tolist() == [1, 1, 1]

    with pytest.raises(DiagnosticsError, match="explicit predictions_by_method"):
        analyze_router_weights(
            logits,
            labels,
            {"uniform_probability": uniform},
            np.array([100, 20, 5]),
        )

    report = analyze_router_weights(
        logits,
        labels,
        {"uniform_probability": uniform},
        np.array([100, 20, 5]),
        predictions_by_method={"uniform_probability": weighted_probability_prediction},
    )
    method = report["methods"]["uniform_probability"]
    assert method["fit_set_metrics"]["ordinary_accuracy"] == 1.0
    profile = method["overall_profile"]
    assert profile["mean_weights"] == [0.25] * 4
    assert profile["allocation_entropy_bits"] == 2.0
    assert profile["normalized_allocation_entropy"] == 1.0
    assert profile["preferred_expert"] == "CE"
    assert profile["preferred_expert_ties"] == list(EXPERT_NAMES)
    assert profile["preferred_expert_tie_count"] == 4
    assert profile["weight_quantiles"]["p50"] == [0.25] * 4
    assert profile["per_row_preferred_tie_count_histogram"] == {"1": 0, "2": 0, "3": 0, "4": 3}


def test_profile_comparison_is_order_sensitive_and_constants_have_null_correlation():
    class_profiles_a = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    class_profiles_b = class_profiles_a[::-1].copy()
    reordered = compare_profiles(class_profiles_a, class_profiles_b)
    assert reordered["l1_distance"] == 2.0
    assert reordered["preferred_agreement"] == 0.0

    constant = np.full(4, 0.25)
    same = compare_profiles(constant, constant)
    assert same["l1_distance"] == 0.0
    assert same["correlation"] is None
    assert same["entropy_difference_bits"] == 0.0


def test_sinkhorn_decomposition_reports_exact_residual_kl_and_scoped_transitions():
    labels = np.array([0, 1, 2])
    logits = np.array(
        [
            [[2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 1]],
            [[0, 2, 0], [2, 0, 0], [0, 0, 2], [1, 1, 1]],
            [[0, 0, 2], [2, 0, 0], [0, 2, 0], [1, 1, 1]],
        ],
        dtype=np.float64,
    )
    raw = np.array(
        [[0.60, 0.20, 0.10, 0.10], [0.10, 0.60, 0.20, 0.10], [0.10, 0.10, 0.70, 0.10]],
        dtype=np.float64,
    )
    smoothed = np.array(
        [[0.55, 0.20, 0.15, 0.10], [0.10, 0.55, 0.20, 0.15], [0.15, 0.10, 0.60, 0.15]],
        dtype=np.float64,
    )
    prices = np.array([0.2, -0.1, 0.3, -0.4])
    prior = np.array([0.25, 0.25, 0.25, 0.25])
    adjusted = apply_log_bias(smoothed, prices)

    report = sinkhorn_stage_diagnostics(
        logits,
        labels,
        raw,
        smoothed,
        adjusted,
        prices,
        prior,
        np.array([100, 20, 5]),
        "synthetic",
    )
    assert report["decomposition"]["exact_frozen_price_decomposition"] is True
    assert report["decomposition"]["expected_adjusted_max_abs_error"] < 1e-12
    expected_row_kl = np.sum(adjusted * np.log(adjusted / smoothed), axis=1).mean()
    assert np.isclose(report["sinkhorn"]["mean_row_kl_adjusted_vs_smoothed_nats"], expected_row_kl)
    assert len(report["scoped_rows"]) == 6  # three classes and three H/M/T groups
    tail = next(row for row in report["scoped_rows"] if row["scope"] == "group" and row["scope_id"] == "tail")
    assert tail["sample_count"] == 1
    assert len(tail["transitions"]) == 3
    assert report["transitions"][2]["correctness_contingency"]["wrong_to_correct"] == report["transitions"][2]["helped_count"]

    mismatch = sinkhorn_stage_diagnostics(
        logits,
        labels,
        raw,
        smoothed,
        raw,
        prices,
        prior,
        np.array([100, 20, 5]),
        "mismatch",
    )
    assert mismatch["decomposition"]["decomposition_status"] == "mismatch"
    assert mismatch["decomposition"]["expected_adjusted_max_abs_error"] > 0.0


def test_selective_gate_uses_anchor_distance_and_preserves_the_simplex():
    anchor = np.array([1.0, 0.0, 0.0, 0.0])
    ridge = np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    sinkhorn = np.array(
        [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    weights, gates = selective_gate(ridge, sinkhorn, tau=2.0, anchor=anchor)
    assert np.allclose(gates, [0.0, 0.5, 1.0])
    assert np.allclose(weights[0], ridge[0])
    assert np.allclose(weights[1], [0.25, 0.25, 0.5, 0.0])
    assert np.allclose(weights[2], sinkhorn[2])
    assert np.allclose(weights.sum(axis=1), 1.0)
