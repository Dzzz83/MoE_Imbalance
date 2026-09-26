"""Small synthetic tests for the array-only four-expert OT components."""

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

from scripts.ridge_sinkhorn_ot import (  # noqa: E402
    DEFAULT_Q,
    EXPERT_ORDER,
    OTConvergenceError,
    RidgeSinkhornOTError,
    apply_log_bias,
    apply_prior_bias,
    balanced_sinkhorn,
    fit_frozen_dual_prices,
    relaxed_sinkhorn,
)


def _kernel_fixture() -> np.ndarray:
    return np.array(
        [
            [0.70, 0.10, 0.10, 0.10],
            [0.10, 0.65, 0.15, 0.10],
            [0.15, 0.15, 0.55, 0.15],
            [0.10, 0.20, 0.20, 0.50],
            [0.45, 0.25, 0.20, 0.10],
            [0.15, 0.35, 0.30, 0.20],
        ],
        dtype=np.float64,
    )


def test_relaxed_ot_preserves_rows_and_rho_zero_returns_original_kernel():
    kernel = _kernel_fixture()
    zero = relaxed_sinkhorn(kernel, rho=0.0)
    fitted = relaxed_sinkhorn(kernel, rho=1.0)

    assert zero.expert_order == EXPERT_ORDER
    assert zero.diagnostics.converged
    assert zero.diagnostics.iterations == 0
    assert np.array_equal(zero.weights, kernel)
    assert fitted.weights.shape == kernel.shape
    assert np.isfinite(fitted.weights).all()
    assert np.all(fitted.weights >= 0.0)
    assert np.allclose(fitted.weights.sum(axis=1), 1.0, atol=1e-12)
    assert fitted.diagnostics.converged
    assert fitted.diagnostics.max_log_price_residual <= 1e-10
    assert fitted.diagnostics.objective == pytest.approx(
        fitted.diagnostics.kernel_kl + fitted.diagnostics.rho * fitted.diagnostics.prior_kl
    )


def test_balanced_sinkhorn_enforces_both_marginals():
    result = balanced_sinkhorn(_kernel_fixture())

    assert result.diagnostics.converged
    assert np.allclose(result.weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    assert np.allclose(result.weights.mean(axis=0), 0.25, rtol=0.0, atol=1e-10)
    assert result.diagnostics.max_target_marginal_residual <= 1e-10


def test_sinkhorn_outputs_are_deterministic():
    kernel = _kernel_fixture()
    first = relaxed_sinkhorn(kernel, rho=10.0)
    second = relaxed_sinkhorn(kernel, rho=10.0)

    assert np.array_equal(first.weights, second.weights)
    assert np.array_equal(first.log_prices, second.log_prices)
    assert first.diagnostics == second.diagnostics


def test_stronger_relaxed_prior_penalty_moves_mass_toward_q():
    kernel = np.tile(np.array([[0.70, 0.10, 0.10, 0.10]]), (5, 1))
    distances = [
        np.linalg.norm(relaxed_sinkhorn(kernel, rho=rho).weights.mean(axis=0) - DEFAULT_Q)
        for rho in (0.1, 1.0, 10.0)
    ]

    assert distances[0] > distances[1] > distances[2]


@pytest.mark.parametrize("rho", [0.1, 1.0, 10.0])
def test_relaxed_solver_matches_constant_kernel_closed_form(rho):
    row = np.array([0.70, 0.10, 0.10, 0.10])
    kernel = np.tile(row, (4, 1))
    log_expected = (np.log(row) + rho * np.log(DEFAULT_Q)) / (1.0 + rho)
    expected = np.exp(log_expected - np.logaddexp.reduce(log_expected))

    result = relaxed_sinkhorn(kernel, rho=rho)

    assert np.allclose(result.weights, expected[None, :], atol=1e-10)
    assert json.loads(json.dumps(result.diagnostics))["converged"] is True


def test_nonconvergence_raises_with_solver_diagnostics():
    with pytest.raises(OTConvergenceError, match="did not converge") as error:
        relaxed_sinkhorn(_kernel_fixture(), rho=10.0, tolerance=1e-14, max_iterations=1)

    assert not error.value.diagnostics.converged
    assert error.value.diagnostics.iterations == 1
    assert np.isfinite(error.value.diagnostics.max_log_price_residual)


def test_log_domain_global_bias_is_stable_for_extreme_scores():
    kernel = np.array(
        [
            [1e-300, 1e-200, 1e-100, 1.0],
            [1.0, 1e-200, 1e-100, 1e-300],
        ],
        dtype=np.float64,
    )
    kernel /= kernel.sum(axis=1, keepdims=True)
    weights = apply_log_bias(kernel, np.array([-10_000.0, -100.0, 100.0, 10_000.0]))

    assert np.isfinite(weights).all()
    assert np.all(weights >= 0.0)
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-12)


def test_frozen_prices_are_row_independent_and_fit_rows_are_recorded():
    fit_kernel = _kernel_fixture()
    future = np.array(
        [
            [0.60, 0.20, 0.10, 0.10],
            [0.10, 0.20, 0.60, 0.10],
            [0.20, 0.20, 0.20, 0.40],
        ]
    )
    prices = fit_frozen_dual_prices(fit_kernel, rho=1.0)
    batched = prices.apply(future)
    separately = np.vstack([prices.apply(row[None, :])[0] for row in future])

    assert prices.fit_sample_count == len(fit_kernel)
    assert np.array_equal(batched, separately)
    assert np.allclose(prices.apply(future[:1]), batched[:1])
    assert np.allclose(batched.sum(axis=1), 1.0)


def test_prior_only_global_bias_control_multiplies_by_q_and_normalizes_rows():
    kernel = _kernel_fixture()
    controlled = apply_prior_bias(kernel)
    expected = kernel * np.asarray(DEFAULT_Q)[None, :]
    expected /= expected.sum(axis=1, keepdims=True)

    assert np.allclose(controlled, expected)
    assert np.allclose(controlled.sum(axis=1), 1.0)


@pytest.mark.parametrize(
    "kernel",
    [
        np.ones((3, 3)) / 3.0,
        np.ones((3, 5)) / 5.0,
        np.empty((0, 4)),
        np.array([[0.25, 0.25, 0.25, 0.0]]),
        np.array([[0.25, 0.25, np.nan, 0.25]]),
        np.array([[0.20, 0.20, 0.20, 0.20]]),
    ],
)
def test_invalid_kernels_are_rejected(kernel):
    with pytest.raises(RidgeSinkhornOTError):
        relaxed_sinkhorn(kernel, rho=1.0)


@pytest.mark.parametrize(
    "q",
    [
        (0.25, 0.25, 0.25),
        (0.25, 0.25, 0.25, 0.0),
        (0.25, 0.25, np.nan, 0.25),
        (0.20, 0.20, 0.20, 0.20),
    ],
)
def test_invalid_priors_are_rejected(q):
    with pytest.raises(RidgeSinkhornOTError, match="q"):
        relaxed_sinkhorn(_kernel_fixture(), rho=1.0, q=q)


@pytest.mark.parametrize("rho", [-1.0, 0.2, np.inf, np.nan])
def test_rho_is_limited_to_the_frozen_study_grid(rho):
    with pytest.raises(RidgeSinkhornOTError, match="rho"):
        relaxed_sinkhorn(_kernel_fixture(), rho=rho)


def test_invalid_bias_and_iteration_controls_are_rejected():
    with pytest.raises(RidgeSinkhornOTError, match="log_bias"):
        apply_log_bias(_kernel_fixture(), np.zeros(3))
    with pytest.raises(RidgeSinkhornOTError, match="max_iterations"):
        relaxed_sinkhorn(_kernel_fixture(), rho=1.0, max_iterations=0)
    with pytest.raises(RidgeSinkhornOTError, match="tolerance"):
        balanced_sinkhorn(_kernel_fixture(), tolerance=0.0)
