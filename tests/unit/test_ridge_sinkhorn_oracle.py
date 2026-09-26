"""Synthetic checks for Ridge/Sinkhorn oracle and residual weight mapping."""

from __future__ import annotations

import json
import os
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

from scripts.ridge_sinkhorn_oracle import (  # noqa: E402
    FIXED_007_ANCHOR,
    OracleTargetResult,
    RidgeSinkhornOracleError,
    compute_oracle_targets,
    project_onto_simplex,
    residuals_to_weights,
    smooth_positive_kernel,
)


def _oracle_fixture() -> tuple[np.ndarray, np.ndarray]:
    # Row 0 is already correctly classified by fixed_007. Row 1 has a positive
    # anchor margin against class 1 and can reduce it by moving toward experts
    # 2 and 3.
    logits = np.array(
        [
            [
                [4.0, 0.0, -1.0],
                [3.0, 0.0, -1.0],
                [5.0, 0.0, -1.0],
                [2.0, 0.0, -1.0],
            ],
            [
                [0.0, 8.0, -1.0],
                [0.0, 10.0, -1.0],
                [0.0, -3.0, -1.0],
                [0.0, -2.0, -1.0],
            ],
        ],
        dtype=np.float64,
    )
    return logits, np.array([0, 0], dtype=np.int64)


def test_oracle_returns_exact_zero_residual_for_zero_hinge_anchor_row():
    logits, labels = _oracle_fixture()
    result = compute_oracle_targets(logits, labels, penalty=1.0)

    assert isinstance(result, OracleTargetResult)
    assert result.oracle_weights.shape == (2, 4)
    assert result.residuals.shape == (2, 4)
    assert np.array_equal(result.oracle_weights[0], np.asarray(FIXED_007_ANCHOR))
    assert np.array_equal(result.residuals[0], np.zeros(4))
    assert result.diagnostics["exact_anchor_rows"] == 1
    assert result.diagnostics["all_succeeded"] is True


def test_oracle_solution_is_deterministic_feasible_and_reports_objective_checks():
    logits, labels = _oracle_fixture()
    first = compute_oracle_targets(logits, labels, penalty=1.0)
    second = compute_oracle_targets(logits, labels, penalty=1.0)

    weights = first.oracle_weights
    assert np.array_equal(weights, second.oracle_weights)
    assert np.isfinite(weights).all()
    assert np.all(weights >= 0.0)
    assert np.allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-10)
    assert not np.allclose(first.residuals[1], 0.0)
    diagnostics = first.diagnostics
    assert diagnostics["solver"] == "scipy.optimize.minimize(method='SLSQP')"
    assert diagnostics["max_primal_residual"] <= 1e-7
    assert diagnostics["max_objective_residual"] <= 1e-7
    assert np.all(np.asarray(diagnostics["objective_residual"]) <= 1e-7)
    json.dumps(diagnostics)

    row = logits[1]
    label = labels[1]
    wrong_classes = np.arange(row.shape[1]) != label
    margins = (row[:, wrong_classes] - row[:, label, None]).T @ weights[1]
    expected_hinge = max(0.0, float(margins.max()))
    expected_objective = expected_hinge + 0.5 * np.square(
        weights[1] - np.asarray(FIXED_007_ANCHOR)
    ).sum()
    assert diagnostics["objective"][1] == pytest.approx(expected_objective, abs=1e-8)
    assert diagnostics["max_solution_margin"][1] <= 1e-7


def test_margin_sensitivity_is_explicit_and_diagnostic_records_margin():
    logits, labels = _oracle_fixture()
    zero_margin = compute_oracle_targets(logits, labels, penalty=1.0)
    unit_margin = compute_oracle_targets(logits, labels, penalty=1.0, margin=1.0)

    assert zero_margin.diagnostics["margin"] == 0.0
    assert unit_margin.diagnostics["margin"] == 1.0
    assert not np.array_equal(unit_margin.residuals[1], np.zeros(4))
    assert np.isfinite(unit_margin.oracle_weights).all()
    assert np.allclose(unit_margin.oracle_weights.sum(axis=1), 1.0, atol=1e-10)


def test_simplex_projection_is_euclidean_and_supports_vectors_and_batches():
    values = np.array(
        [
            [0.25, 0.25, 0.25, 0.25],
            [2.0, -1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
        ]
    )
    projected = project_onto_simplex(values)

    assert projected.shape == values.shape
    assert np.all(projected >= 0.0)
    assert np.allclose(projected.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    assert np.array_equal(projected[0], values[0])
    assert np.array_equal(projected[1], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(projected[2], [0.0, 1.0 / 3, 1.0 / 3, 1.0 / 3])
    assert np.array_equal(project_onto_simplex(values[1]), projected[1])


def test_residual_mapping_projects_anchor_plus_scaled_prediction():
    predicted = np.array([[1.0, -1.0, 0.0, 0.0], [-0.5, 0.0, 0.5, 0.0]])
    mapped = residuals_to_weights(predicted, scale=0.5)
    expected = project_onto_simplex(
        np.asarray(FIXED_007_ANCHOR)[None, :] + 0.5 * predicted
    )

    assert np.array_equal(mapped, expected)
    assert mapped.shape == (2, 4)
    assert np.all(mapped >= 0.0)
    assert np.allclose(mapped.sum(axis=1), 1.0)
    assert np.array_equal(
        residuals_to_weights(predicted[0], scale=0.0),
        np.asarray(FIXED_007_ANCHOR),
    )


def test_positive_kernel_smoothing_matches_frozen_formula_including_exact_zeros():
    weights = np.array([[0.0, 0.25, 0.5, 0.25], [1.0, 0.0, 0.0, 0.0]])
    epsilon = 1e-6
    smoothed = smooth_positive_kernel(weights, epsilon=epsilon)
    expected = (1.0 - epsilon) * weights + epsilon / 4.0

    assert np.array_equal(smoothed, expected)
    assert np.all(smoothed > 0.0)
    assert np.allclose(smoothed.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    "call, pattern",
    [
        (lambda: compute_oracle_targets(np.zeros((2, 3, 4)), [0, 0]), "exactly 4 experts"),
        (lambda: compute_oracle_targets(np.zeros((2, 4, 4)), [0]), "labels must have shape"),
        (lambda: compute_oracle_targets(np.zeros((2, 4, 4)), [0, 4]), r"\[0, 4\)"),
        (lambda: compute_oracle_targets(np.zeros((1, 4, 4)), [0], penalty=0.0), "penalty"),
        (lambda: compute_oracle_targets(np.zeros((1, 4, 4)), [0], margin=-1.0), "margin"),
        (lambda: compute_oracle_targets(np.zeros((1, 4, 4)), [0], anchor=[0.0, 0.2, 0.5, 0.2]), "sum to one"),
        (lambda: compute_oracle_targets(np.zeros((1, 4, 4)), [0], max_iterations=0), "max_iterations"),
        (lambda: project_onto_simplex(np.zeros((2, 5))), "shape"),
        (lambda: project_onto_simplex(np.zeros((0, 4))), "at least one sample"),
        (lambda: project_onto_simplex([0.0, 0.0, np.nan, 0.0]), "finite"),
        (lambda: residuals_to_weights(np.zeros((2, 3))), "shape"),
        (lambda: residuals_to_weights(np.zeros(4), scale=-1.0), "scale"),
        (lambda: smooth_positive_kernel(np.array([0.0, 0.2, 0.3, 0.5]), 0.0), "epsilon"),
        (lambda: smooth_positive_kernel(np.array([0.0, 0.2, 0.3, 0.4])), "sum to one"),
    ],
)
def test_invalid_inputs_are_rejected(call, pattern):
    with pytest.raises(RidgeSinkhornOracleError, match=pattern):
        call()


def test_oracle_reports_nonconvergence_instead_of_returning_unchecked_targets():
    logits = np.random.default_rng(4).normal(size=(1, 4, 100))
    labels = np.array([0], dtype=np.int64)
    with pytest.raises(RidgeSinkhornOracleError, match="oracle solver failed"):
        compute_oracle_targets(logits, labels, max_iterations=1)
