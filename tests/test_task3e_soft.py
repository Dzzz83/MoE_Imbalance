"""Synthetic regression tests for Task 3E-B.

These tests use hand-constructed logits and restricted-view fixtures only.  No
CIFAR test data, checkpoints, training, or real OOF artifact is loaded.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.task3c_oof import AlignedOOFDataset  # noqa: E402
from scripts.task3e_fixed import (  # noqa: E402
    EXPERT_ORDER,
    RestrictedAnalysisDataset,
    Task3EError,
)
from scripts.task3e_soft import (  # noqa: E402
    MARGIN_TOLERANCE,
    SoftFeasibilityAnalyzer,
    SoftMixtureOracle,
    SoftOracleResult,
)


def _logits_100() -> np.ndarray:
    return np.zeros((4, 100), dtype=np.float64)


def _oracle(num_classes: int = 100) -> SoftMixtureOracle:
    return SoftMixtureOracle(num_classes=num_classes)


def test_one_expert_and_uniform_positive_cases_are_correctable():
    logits = _logits_100()
    logits[0, 0] = 3.0
    logits[0, 1] = 0.0
    result = _oracle().solve(logits, 0)
    assert result.status == "correctable"
    assert result.verification_passed is True
    assert result.optimal_margin == pytest.approx(3.0)
    assert result.reconstructed_margin == pytest.approx(3.0)
    assert sum(result.oracle_weights) == pytest.approx(1.0)
    assert min(result.oracle_weights) >= 0.0

    uniform_logits = np.zeros((4, 100), dtype=np.float64)
    uniform_logits[:, 0] = 2.0
    uniform_logits[:, 1] = 0.0
    uniform_result = _oracle().solve(uniform_logits, 0)
    assert uniform_result.status == "correctable"
    assert uniform_result.reconstructed_margin > MARGIN_TOLERANCE


def test_all_experts_wrong_but_an_interior_mixture_is_correct():
    logits = np.full((4, 100), -10.0, dtype=np.float64)
    # For expert 0, margins against class 1 and 2 are -1 and +3.  Expert 1
    # swaps those margins.  Every expert predicts a wrong class, while the
    # equal interior mixture gives the true class both positive margins.
    logits[0, 0], logits[0, 1], logits[0, 2] = 0.0, 1.0, -3.0
    logits[1, 0], logits[1, 1], logits[1, 2] = 0.0, -3.0, 1.0
    logits[2] = logits[0]
    logits[3] = logits[1]
    assert np.array_equal(logits.argmax(axis=1), np.array([1, 2, 1, 2]))
    result = _oracle().solve(logits, 0)
    assert result.status == "correctable"
    assert result.reconstructed_margin > MARGIN_TOLERANCE
    # Duplicate experts make the optimizer's particular optimal point
    # non-unique; the mass on each opposing-margin group is fixed.
    assert result.oracle_weights[0] + result.oracle_weights[2] == pytest.approx(0.5)
    assert result.oracle_weights[1] + result.oracle_weights[3] == pytest.approx(0.5)


def test_negative_and_zero_optimal_margins_are_distinguished():
    negative = np.zeros((4, 100), dtype=np.float64)
    negative[:, 1] = 2.0
    assert _oracle().solve(negative, 0).status == "not_strictly_correctable"

    zero = np.zeros((4, 100), dtype=np.float64)
    zero[:, 1] = 0.0
    zero[:, 2] = -1.0
    result = _oracle().solve(zero, 0)
    assert result.status == "numerically_ambiguous"
    assert abs(result.reconstructed_margin) <= MARGIN_TOLERANCE


def test_invalid_logits_dimensions_and_nonfinite_values_are_explicit_failures():
    with pytest.raises(Task3EError, match="shape"):
        _oracle().solve(np.zeros((4, 99)), 0)
    with pytest.raises(Task3EError, match="NaN"):
        bad = _logits_100()
        bad[0, 0] = np.nan
        _oracle().solve(bad, 0)
    with pytest.raises(Task3EError, match="infinity"):
        bad = _logits_100()
        bad[0, 0] = np.inf
        _oracle().solve(bad, 0)


@pytest.mark.parametrize(
    "name", ["margin_tolerance", "feasibility_tolerance", "verification_tolerance"]
)
def test_soft_oracle_rejects_nonfinite_tolerances(name):
    with pytest.raises(Task3EError, match="finite positive"):
        SoftMixtureOracle(**{name: np.nan})


def test_lp_objective_direction_unrestricted_margin_and_all_99_constraints():
    logits = _logits_100()
    logits[0, 0] = 4.0
    problem = _oracle().build_problem(logits, 0)
    assert problem["objective"].tolist() == [0.0, 0.0, 0.0, 0.0, -1.0]
    assert problem["A_ub"].shape == (99, 5)
    assert len(problem["b_ub"]) == 99
    assert problem["bounds"][-1] == (None, None)
    result = _oracle().solve(logits, 0)
    assert result.status == "correctable"
    assert result.optimal_margin > 0.0


class _Result:
    def __init__(self, *, success: bool, x=None, fun=None, status=0, message="fake"):
        self.success = success
        self.x = x
        self.fun = fun
        self.status = status
        self.message = message


def test_invalid_solver_solution_is_not_counted_as_correctable():
    def invalid_solver(*args, **kwargs):
        return _Result(
            success=True,
            x=np.array([1.1, 0.0, 0.0, 0.0, 1.0]),
            fun=-1.0,
            message="invalid weights",
        )

    result = SoftMixtureOracle(num_classes=100, solver=invalid_solver).solve(
        _logits_100(), 0
    )
    assert result.status == "solver_failure"
    assert result.verification_passed is False
    assert "upper_weight_constraint_violation" in result.verification_errors


def test_solver_failure_is_separate_from_negative_infeasibility():
    def failing_solver(*args, **kwargs):
        return _Result(success=False, status=2, message="deliberate failure")

    result = SoftMixtureOracle(num_classes=100, solver=failing_solver).solve(
        _logits_100(), 0
    )
    assert result.status == "solver_failure"
    assert result.optimal_margin is None
    assert result.solver_status == 2
    assert result.solver_message == "deliberate failure"


def _restricted_fixture() -> RestrictedAnalysisDataset:
    labels = np.arange(6, dtype=np.int64)
    logits = np.zeros((6, 4, 6), dtype=np.float64)
    # Sample 0: all experts and uniform are correct.
    logits[0, :, 0] = 2.0
    # Sample 1: all experts are wrong, but the first two experts form an
    # interior correct mixture for true class 1.
    logits[1] = -10.0
    logits[1, 0, 1], logits[1, 0, 2], logits[1, 0, 3] = 0.0, 1.0, -3.0
    logits[1, 1, 1], logits[1, 1, 2], logits[1, 1, 3] = 0.0, -3.0, 1.0
    logits[1, 2] = -10.0
    logits[1, 2, 1], logits[1, 2, 4] = 0.0, 50.0
    logits[1, 3] = -10.0
    logits[1, 3, 1], logits[1, 3, 5] = 0.0, 50.0
    # Sample 2: all experts point at a common competitor.
    logits[2, :, 0] = 2.0
    logits[2, :, 2] = 0.0
    # Sample 3: exact tie for the true class and a competitor.
    logits[3, :, 3] = 0.0
    logits[3, :, 0] = 0.0
    # Samples 4 and 5 have distinct, simple positive solutions.
    logits[4, :, 4] = 1.0
    logits[5, :, 5] = 1.0
    return RestrictedAnalysisDataset.from_arrays(
        sample_indices=np.arange(6, dtype=np.int64),
        labels=labels,
        inner_fold_ids=np.array([1, 2, 3, 1, 2, 3], dtype=np.int64),
        logits=logits,
    )


def test_analyzer_aggregates_feasibility_headroom_and_tail_groups():
    dataset = _restricted_fixture()
    analyzer = SoftFeasibilityAnalyzer(
        dataset,
        np.array([100, 100, 20, 20, 5, 5], dtype=np.int64),
    )
    analysis = analyzer.analyze()
    aggregate = analysis["aggregate"]
    assert aggregate["status_counts"]["correctable"] == 4
    assert aggregate["status_counts"]["not_strictly_correctable"] == 1
    assert aggregate["status_counts"]["numerically_ambiguous"] == 1
    assert aggregate["status_counts"]["solver_failure"] == 0
    assert aggregate["soft_oracle"]["strict_correctable_count"] == 4
    assert aggregate["soft_oracle"]["lower_bound_fraction"] == pytest.approx(4 / 6)
    assert aggregate["soft_oracle"]["potential_upper_bound_fraction"] == pytest.approx(5 / 6)
    assert aggregate["all_experts_wrong_but_soft_correctable"]["count"] == 1
    assert aggregate["uniform_wrong_but_soft_correctable"]["counts_by_group"]["all"]["count"] >= 1
    assert aggregate["soft_oracle"]["strict_correctable"]["head_coverage"] == pytest.approx(1.0)
    assert aggregate["soft_oracle"]["strict_correctable"]["medium_coverage"] == pytest.approx(0.0)
    assert aggregate["soft_oracle"]["strict_correctable"]["tail_coverage"] == pytest.approx(1.0)
    assert len(analysis["tail_class_analysis"]) == 2
    assert all(row["sample_count"] == 1 for row in analysis["tail_class_analysis"])
    assert all(record["oracle_weights_are_label_dependent"] for record in analysis["per_image"])


class _SequenceOracle:
    num_classes = 6
    margin_tolerance = MARGIN_TOLERANCE

    def __init__(self, statuses):
        self.statuses = list(statuses)
        self.index = 0

    def solve(self, logits, true_label):
        status = self.statuses[self.index]
        self.index += 1
        if status == "solver_failure":
            return SoftOracleResult(
                status=status,
                optimal_margin=None,
                reconstructed_margin=None,
                oracle_weights=None,
                solver_status=2,
                solver_message="stub failure",
                verification_passed=False,
                verification_errors=("stub",),
                constraint_count=5,
            )
        margin = {
            "correctable": 1.0,
            "not_strictly_correctable": -1.0,
            "numerically_ambiguous": 0.0,
        }[status]
        return SoftOracleResult(
            status=status,
            optimal_margin=margin,
            reconstructed_margin=margin,
            oracle_weights=(1.0, 0.0, 0.0, 0.0),
            solver_status=0,
            solver_message="stub",
            verification_passed=True,
            verification_errors=(),
            constraint_count=5,
            reported_margin=margin,
            objective_value=-margin,
        )


def test_analyzer_preserves_ambiguous_and_solver_failure_accounting():
    original = _restricted_fixture()
    logits = original.logits.copy()
    # Keep the stub's unresolved cases away from a strictly positive uniform
    # margin, so the test isolates status accounting rather than the real-LP
    # numerical invariant.
    logits[4] = 0.0
    logits[5] = 0.0
    dataset = RestrictedAnalysisDataset.from_arrays(
        sample_indices=original.sample_indices,
        labels=original.labels,
        inner_fold_ids=original.inner_fold_ids,
        logits=logits,
    )
    # No uniform-positive sample is used with an unresolved status here; the
    # stub isolates aggregate accounting from the real solver.
    analyzer = SoftFeasibilityAnalyzer(
        dataset,
        np.array([100, 100, 20, 20, 5, 5], dtype=np.int64),
        oracle=_SequenceOracle(
            [
                "correctable",
                "correctable",
                "not_strictly_correctable",
                "numerically_ambiguous",
                "solver_failure",
                "correctable",
            ]
        ),
    )
    analysis = analyzer.analyze()
    counts = analysis["aggregate"]["status_counts"]
    assert counts == {
        "correctable": 3,
        "not_strictly_correctable": 1,
        "numerically_ambiguous": 1,
        "solver_failure": 1,
    }
    assert analysis["aggregate"]["soft_oracle"]["unresolved_count"] == 2
    assert analysis["aggregate"]["class_status_distribution"]["solver_failure"]["counts_by_class"][4] == 1


def test_restricted_type_and_reserved_partitions_are_rejected():
    logits = np.zeros((1, 4, 6), dtype=np.float64)
    aligned = AlignedOOFDataset(
        sample_indices=np.array([0], dtype=np.int64),
        outer_fold_ids=np.array([0], dtype=np.int64),
        inner_fold_ids=np.array([1], dtype=np.int64),
        labels=np.array([0], dtype=np.int64),
        logits=logits,
        expert_names=EXPERT_ORDER,
        metadata={},
    )
    with pytest.raises(Task3EError, match="RestrictedAnalysisDataset"):
        SoftFeasibilityAnalyzer(aligned, np.ones(6, dtype=np.int64))

    with pytest.raises(Task3EError, match="inner-fold-0"):
        RestrictedAnalysisDataset.from_arrays(
            sample_indices=np.array([0], dtype=np.int64),
            labels=np.array([0], dtype=np.int64),
            inner_fold_ids=np.array([0], dtype=np.int64),
            logits=logits,
        )

    with pytest.raises(Task3EError, match="nonzero outer fold"):
        RestrictedAnalysisDataset.from_arrays(
            sample_indices=np.array([0], dtype=np.int64),
            labels=np.array([0], dtype=np.int64),
            inner_fold_ids=np.array([1], dtype=np.int64),
            outer_fold_ids=np.array([1], dtype=np.int64),
            logits=logits,
        )
