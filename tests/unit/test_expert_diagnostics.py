"""Synthetic regression tests for the reusable expert diagnostics.

Every fixture in this module is hand-constructed.  These tests intentionally
exercise the supplied-array API only; they do not load datasets, checkpoints,
or cached experiment artifacts.
"""

from __future__ import annotations

import os
import sys

import numpy as np

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

from scripts.expert_diagnostics import ExpertDiagnostics
from scripts.router.probability import ProbabilityAverageRouter
from scripts.router.uniform import UniformRouter


def _logits_for_predictions(predictions: np.ndarray, margins: np.ndarray) -> np.ndarray:
    """Build finite logits with the requested argmax and confidence order."""
    predictions = np.asarray(predictions, dtype=np.int64)
    margins = np.asarray(margins, dtype=np.float64)
    n, experts = predictions.shape
    classes = int(max(predictions.max(), 0)) + 1
    logits = np.zeros((n, experts, classes), dtype=np.float64)
    rows = np.arange(n)[:, None]
    expert_cols = np.arange(experts)[None, :]
    logits[rows, expert_cols, predictions] = margins
    return logits


def test_four_expert_agreement_patterns_do_not_infer_a_dissenter_from_2_1_1():
    predictions = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 1],
        [0, 0, 1, 2],
        [0, 1, 2, 3],
    ])
    labels = np.array([0, 1, 0, 0, 0])

    report = ExpertDiagnostics(predictions=predictions, labels=labels).agreement_patterns()

    assert report["pattern_counts"] == {
        "4-0": 1,
        "3-1": 1,
        "2-2": 1,
        "2-1-1": 1,
        "1-1-1-1": 1,
    }
    unique = report["unique_dissenter"]
    assert unique["count"] == 1
    assert unique["denominator"] == 1
    assert unique["agreeing_group_correct_fraction"] == 0.0
    assert unique["dissenting_expert_correct_fraction"] == 1.0
    assert report["non_unique_dissenter_count"] == 4


def test_correctness_counts_are_separate_from_confidence_ranking():
    predictions = np.array([
        [0, 1, 2],  # exactly one correct, but it is least confident
        [0, 0, 1],  # multiple correct, both less confident than the wrong expert
        [1, 2, 1],  # no correct expert
    ])
    labels = np.array([0, 0, 0])
    margins = np.array([
        [1.0, 3.0, 2.0],
        [2.0, 1.0, 4.0],
        [3.0, 1.0, 2.0],
    ])
    logits = _logits_for_predictions(predictions, margins)

    report = ExpertDiagnostics(
        logits=logits, predictions=predictions, labels=labels
    ).correctness_diagnostics()

    assert report["num_correct_experts"] == [1, 2, 0]
    assert report["exactly_one_correct"]["count"] == 1
    assert report["exactly_one_correct"]["fraction"] == 1 / 3
    assert report["multiple_correct"]["count"] == 1
    assert report["multiple_correct"]["fraction"] == 1 / 3
    assert report["no_correct"]["count"] == 1
    assert report["no_correct"]["fraction"] == 1 / 3

    # The globally most-confident expert is wrong on all three samples.
    assert report["globally_most_confident"]["correct_count"] == 0
    assert report["globally_most_confident"]["correct_fraction"] == 0.0

    # Conditional on an expert being correct, the best-ranked correct expert
    # is rank 3 on the first sample and rank 2 on the second.
    ranking = report["confidence_ranking_among_correct"]
    assert ranking["denominator"] == 2
    assert ranking["best_correct_rank_counts"] == {"1": 0, "2": 1, "3": 1}
    assert ranking["best_correct_rank_fractions"] == {"1": 0.0, "2": 0.5, "3": 0.5}

    # This is the old ambiguity trap: a unique highest-confidence correct
    # expert is not the same event as exactly one correct expert.
    assert report["unique_highest_confidence_among_correct"]["count"] == 2
    assert report["unique_highest_confidence_among_correct"]["denominator"] == 2


def test_complementarity_reports_joint_and_exclusive_correctness():
    labels = np.array([0, 0, 0, 1, 1, 2])
    predictions = np.array([
        [0, 1, 2],
        [0, 0, 2],
        [1, 2, 0],
        [1, 2, 1],
        [2, 1, 0],
        [2, 0, 1],
    ])
    counts = np.array([100, 20, 5])

    report = ExpertDiagnostics(
        predictions=predictions,
        labels=labels,
        class_counts=counts,
        expert_names=["A", "B", "C"],
    ).complementarity()

    assert report["per_expert"]["A"]["correct_count"] == 4
    assert report["per_expert"]["B"]["correct_count"] == 2
    assert report["per_expert"]["C"]["correct_count"] == 2

    pair = report["pairwise"]["A|B"]
    assert pair["joint_correct_count"] == 1
    assert pair["joint_error_count"] == 1
    assert pair["joint_correct_fraction"] == 1 / 6
    assert pair["joint_error_fraction"] == 1 / 6

    assert report["exclusive_correct"]["A"]["count"] == 2
    assert report["exclusive_correct"]["B"]["count"] == 1
    assert report["exclusive_correct"]["C"]["count"] == 1
    assert report["exclusive_correct"]["A"]["fraction_given_any_expert_correct"] == 2 / 6

    # There is one class per H/M/T group here, so the expected macro recalls
    # are directly visible and independently hand-calculated.
    groups = report["head_medium_tail"]
    assert groups["head"]["expert_macro_recall"] == {
        "A": 2 / 3,
        "B": 1 / 3,
        "C": 1 / 3,
    }
    assert groups["medium"]["expert_macro_recall"] == {
        "A": 1 / 2,
        "B": 1 / 2,
        "C": 1 / 2,
    }
    assert groups["tail"]["expert_macro_recall"] == {
        "A": 1.0,
        "B": 0.0,
        "C": 0.0,
    }


def test_hard_headroom_distinguishes_sample_accuracy_from_macro_recall():
    labels = np.array([0, 0, 0, 0, 1, 2, 3])
    predictions = np.array([
        [0, 1],
        [1, 2],
        [0, 2],
        [0, 2],
        [2, 2],
        [1, 2],
        [1, 2],
    ])
    # Head has classes 0 and 1.  The oracle gets 3/4 head samples right, but
    # macro head recall is ((3/4) + 0) / 2 = 3/8.
    counts = np.array([100, 100, 20, 5])
    report = ExpertDiagnostics(
        predictions=predictions, labels=labels, class_counts=counts
    ).hard_routing_headroom()

    assert report["all_experts_wrong_fraction"] == 3 / 7
    assert report["at_least_one_expert_correct_fraction"] == 4 / 7
    assert report["hard_selection_oracle_accuracy"] == 4 / 7
    assert report["hard_selection_oracle_balanced_accuracy"] == (3 / 4 + 0.0 + 1.0 + 0.0) / 4
    assert report["hard_selection_oracle_metrics"]["head"] == 3 / 8
    assert report["hard_selection_oracle_metrics"]["medium"] == 1.0
    assert report["hard_selection_oracle_metrics"]["tail"] == 0.0
    assert "soft" in report["interpretation"]


def test_uniform_soft_mixtures_match_existing_logit_and_probability_baselines():
    logits = np.array([
        [[3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
        [[0.0, 1.0, 0.0], [0.0, 0.0, 2.0], [3.0, 0.0, 0.0]],
    ])
    labels = np.array([0, 2])
    diagnostics = ExpertDiagnostics(logits=logits, labels=labels)
    weights = np.ones((2, 3)) / 3.0

    logit_report = diagnostics.evaluate_soft_mixture(weights, "logits")
    probability_report = diagnostics.evaluate_soft_mixture(weights, "probabilities")
    names = ["A", "B", "C"]
    expected_logit = UniformRouter(names).predict_class(logits)
    expected_probability = ProbabilityAverageRouter(names).predict_class(logits)

    assert np.array_equal(logit_report["predictions"], expected_logit)
    assert np.array_equal(probability_report["predictions"], expected_probability)
    assert np.allclose(
        probability_report["probabilities"],
        ProbabilityAverageRouter(names).averaged_probabilities(logits),
    )
    assert logit_report["metrics"]["ba"] == (1.0 + 0.0) / 2
    assert probability_report["metrics"]["ba"] == (1.0 + 0.0) / 2


def test_soft_mixture_rejects_invalid_weights():
    diagnostics = ExpertDiagnostics(
        logits=np.zeros((2, 3, 4)), labels=np.array([0, 1])
    )
    invalid = [
        (np.ones((2, 2)), "shape"),
        (np.array([[1.0, -0.1, 0.1], [0.0, 0.0, 1.0]]), "non-negative"),
        (np.array([[0.5, 0.5, np.nan], [0.0, 0.0, 1.0]]), "non-finite"),
        (np.array([[0.5, 0.5, 0.5], [0.0, 0.0, 1.0]]), "sum to one"),
    ]
    for weights, message in invalid:
        try:
            diagnostics.evaluate_soft_mixture(weights)
        except ValueError as exc:
            assert message in str(exc), (message, exc)
        else:
            raise AssertionError(f"invalid weights did not raise: {message}")


def test_leave_one_out_ensemble_contribution_reports_paired_ba_and_tail_changes():
    labels = np.array([0, 1, 2, 3])
    predictions = np.array([
        [0, 1, 1],
        [1, 0, 0],
        [2, 3, 3],
        [3, 2, 2],
    ])
    margins = np.full((4, 3), 2.0)
    margins[:, 0] = 3.0
    logits = _logits_for_predictions(predictions, margins)
    counts = np.array([100, 100, 20, 5])

    report = ExpertDiagnostics(
        logits=logits, labels=labels, class_counts=counts
    ).ensemble_contribution()

    assert report["complete"]["ba"] == 0.0
    assert report["complete"]["tail"] == 0.0
    assert report["without_expert"]["E0"]["delta_ba"] == 0.0
    assert report["without_expert"]["E0"]["delta_tail"] == 0.0
    for name in ("E1", "E2"):
        row = report["without_expert"][name]
        assert row["delta_ba"] == 1.0
        assert row["delta_tail"] == 1.0
        assert row["paired_sample_accuracy"]["improved"] == 4
        assert row["paired_sample_accuracy"]["degraded"] == 0


def test_diagnostics_reject_misaligned_missing_and_nonfinite_inputs():
    valid_logits = np.zeros((2, 2, 3), dtype=np.float64)
    valid_logits[:, :, 0] = 1.0

    invalid_cases = [
        (lambda: ExpertDiagnostics(logits=np.zeros((2, 3))), "shape"),
        (
            lambda: ExpertDiagnostics(
                logits=valid_logits,
                predictions=np.zeros((2, 1), dtype=np.int64),
            ),
            "misaligned",
        ),
        (
            lambda: ExpertDiagnostics(
                logits=valid_logits,
                predictions=np.ones((2, 2), dtype=np.int64),
            ),
            "disagree",
        ),
        (
            lambda: ExpertDiagnostics(
                logits=np.array([[[np.nan, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
            ),
            "non-finite",
        ),
        (
            lambda: ExpertDiagnostics(
                predictions=np.zeros((2, 2), dtype=np.int64),
                labels=np.array([0]),
            ),
            "misaligned",
        ),
        (
            lambda: ExpertDiagnostics(
                predictions=np.zeros((2, 2), dtype=np.int64),
                labels=np.array([0, 1]),
                class_counts=np.array([100, 20, 5]),
            ),
            "missing classes",
        ),
        (
            lambda: ExpertDiagnostics(
                predictions=np.zeros((2, 2), dtype=np.int64),
                labels=np.array([0, 0]),
                class_counts=np.array([100, 20, 0]),
            ),
            "positive integer count",
        ),
    ]
    for make_case, message in invalid_cases:
        try:
            make_case()
        except ValueError as exc:
            assert message in str(exc), (message, exc)
        else:
            raise AssertionError(f"invalid input did not raise: {message}")


TESTS = [
    (
        "four-expert agreement patterns",
        test_four_expert_agreement_patterns_do_not_infer_a_dissenter_from_2_1_1,
    ),
    (
        "correctness versus confidence ambiguity",
        test_correctness_counts_are_separate_from_confidence_ranking,
    ),
    (
        "complementarity overlap",
        test_complementarity_reports_joint_and_exclusive_correctness,
    ),
    (
        "hard-routing headroom",
        test_hard_headroom_distinguishes_sample_accuracy_from_macro_recall,
    ),
    (
        "uniform soft-mixture equivalence",
        test_uniform_soft_mixtures_match_existing_logit_and_probability_baselines,
    ),
    ("soft-mixture weight validation", test_soft_mixture_rejects_invalid_weights),
    (
        "leave-one-out contribution",
        test_leave_one_out_ensemble_contribution_reports_paired_ba_and_tail_changes,
    ),
    (
        "diagnostic input validation",
        test_diagnostics_reject_misaligned_missing_and_nonfinite_inputs,
    ),
]


def main() -> int:
    passed = failed = 0
    for name, test in TESTS:
        try:
            test()
            print(f"  PASS {name}")
            passed += 1
        except Exception as exc:  # noqa: BLE001 - standalone test harness
            print(f"  FAIL {name}: {exc}")
            failed += 1
    print(f"expert diagnostics: {passed} passed, {failed} failed")
    return int(failed != 0)


if __name__ == "__main__":
    raise SystemExit(main())
