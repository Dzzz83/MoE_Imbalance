"""Numerically stable four-expert optimal-transport weight maps.

The module consumes only row-stochastic expert kernels. It does not access
labels, logits, folds, datasets, or experiment artifacts. Relaxed OT is a
batch-level fit; :class:`FrozenDualPrices` carries its fitted global prices to
future rows, where each prediction is computed independently.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import logsumexp


EXPERT_ORDER = ("CE", "LAL", "BalancedSoftmax", "Mixup")
NUM_EXPERTS = len(EXPERT_ORDER)
SUPPORTED_RHO = (0.0, 0.1, 1.0, 10.0)
DEFAULT_Q = (0.0125, 0.25, 0.4875, 0.25)
UNIFORM_Q = (0.25, 0.25, 0.25, 0.25)
DEFAULT_TOLERANCE = 1e-10
DEFAULT_MAX_ITERATIONS = 10_000


class RidgeSinkhornOTError(ValueError):
    """Raised when an OT input or numerical result violates its contract."""


class OTDiagnostics(dict[str, Any]):
    """JSON-serializable solver report with convenient attribute access."""

    def __init__(
        self,
        *,
        method: str,
        rho: float | None,
        target_prior: tuple[float, ...],
        iterations: int,
        converged: bool,
        objective: float,
        kernel_kl: float,
        prior_kl: float,
        max_row_residual: float,
        max_target_marginal_residual: float,
        max_log_price_residual: float,
    ) -> None:
        super().__init__(
            method=method,
            rho=rho,
            target_prior=[float(value) for value in target_prior],
            iterations=int(iterations),
            converged=bool(converged),
            objective=float(objective),
            kernel_kl=float(kernel_kl),
            prior_kl=float(prior_kl),
            max_row_residual=float(max_row_residual),
            max_target_marginal_residual=float(max_target_marginal_residual),
            max_log_price_residual=float(max_log_price_residual),
        )

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class OTConvergenceError(RidgeSinkhornOTError):
    """Raised when the requested deterministic iteration cap is exhausted."""

    def __init__(self, message: str, diagnostics: OTDiagnostics) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


@dataclass(frozen=True)
class OTResult:
    """Row-stochastic OT weights, fitted log prices, and solver diagnostics."""

    weights: np.ndarray
    log_prices: np.ndarray
    diagnostics: OTDiagnostics
    expert_order: tuple[str, ...] = EXPERT_ORDER

    def __post_init__(self) -> None:
        weights = np.array(self.weights, dtype=np.float64, copy=True)
        prices = np.array(self.log_prices, dtype=np.float64, copy=True)
        if weights.ndim != 2 or weights.shape[1] != NUM_EXPERTS:
            raise RidgeSinkhornOTError("OT result weights must have shape (samples, 4)")
        if prices.shape != (NUM_EXPERTS,) or not np.isfinite(prices).all():
            raise RidgeSinkhornOTError("OT result log prices must be four finite values")
        if not np.isfinite(weights).all() or np.any(weights < 0.0):
            raise RidgeSinkhornOTError("OT result weights must be finite and non-negative")
        if not np.allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
            raise RidgeSinkhornOTError("OT result weights must be row-stochastic")
        if tuple(self.expert_order) != EXPERT_ORDER:
            raise RidgeSinkhornOTError("OT result expert ordering is not the frozen ordering")
        weights.setflags(write=False)
        prices.setflags(write=False)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "log_prices", prices)


@dataclass(frozen=True)
class FrozenDualPrices:
    """Fit-only global expert prices for row-independent future predictions."""

    rho: float
    q: tuple[float, ...]
    log_prices: np.ndarray
    diagnostics: OTDiagnostics
    fit_sample_count: int
    expert_order: tuple[str, ...] = EXPERT_ORDER

    def __post_init__(self) -> None:
        prices = np.array(self.log_prices, dtype=np.float64, copy=True)
        if prices.shape != (NUM_EXPERTS,) or not np.isfinite(prices).all():
            raise RidgeSinkhornOTError("frozen log prices must be four finite values")
        _validate_rho(self.rho)
        q = _validate_prior(self.q)
        if self.fit_sample_count < 1:
            raise RidgeSinkhornOTError("fit_sample_count must be positive")
        if tuple(self.expert_order) != EXPERT_ORDER:
            raise RidgeSinkhornOTError("frozen prices use the wrong expert ordering")
        prices.setflags(write=False)
        object.__setattr__(self, "log_prices", prices)
        object.__setattr__(self, "q", tuple(float(value) for value in q))

    def apply(self, kernel: np.ndarray) -> np.ndarray:
        """Apply these prices to each future row independently."""
        return apply_log_bias(kernel, self.log_prices)


def _as_real_array(value: object, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise RidgeSinkhornOTError(f"{name} must contain real numeric values")
    if not np.isfinite(array).all():
        raise RidgeSinkhornOTError(f"{name} contains non-finite values")
    return array.astype(np.float64, copy=False)


def _validate_kernel(kernel: object) -> np.ndarray:
    array = _as_real_array(kernel, name="kernel")
    if array.ndim != 2 or array.shape[1] != NUM_EXPERTS or array.shape[0] == 0:
        raise RidgeSinkhornOTError(
            f"kernel must have shape (samples, {NUM_EXPERTS}) with at least one row"
        )
    if np.any(array <= 0.0):
        raise RidgeSinkhornOTError("kernel entries must be strictly positive")
    if not np.allclose(array.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise RidgeSinkhornOTError("kernel rows must sum to one")
    return array


def _validate_prior(q: object) -> np.ndarray:
    array = _as_real_array(q, name="q")
    if array.shape != (NUM_EXPERTS,):
        raise RidgeSinkhornOTError(f"q must have shape ({NUM_EXPERTS},)")
    if np.any(array <= 0.0):
        raise RidgeSinkhornOTError("q entries must be strictly positive")
    if not np.isclose(array.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise RidgeSinkhornOTError("q must sum to one")
    return array


def _validate_rho(rho: object) -> float:
    value_array = _as_real_array(rho, name="rho")
    if value_array.ndim != 0:
        raise RidgeSinkhornOTError("rho must be a scalar")
    value = float(value_array)
    if not any(np.isclose(value, allowed, rtol=0.0, atol=1e-12) for allowed in SUPPORTED_RHO):
        raise RidgeSinkhornOTError(f"rho must be one of {SUPPORTED_RHO}")
    return value


def _validate_solver_controls(tolerance: object, max_iterations: object) -> tuple[float, int]:
    tolerance_array = _as_real_array(tolerance, name="tolerance")
    if tolerance_array.ndim != 0 or float(tolerance_array) <= 0.0:
        raise RidgeSinkhornOTError("tolerance must be a positive finite scalar")
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(
        max_iterations, (int, np.integer)
    ) or int(max_iterations) < 1:
        raise RidgeSinkhornOTError("max_iterations must be a positive integer")
    return float(tolerance_array), int(max_iterations)


def _softmax_with_log_probabilities(log_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    log_weights = log_scores - logsumexp(log_scores, axis=1, keepdims=True)
    weights = np.exp(log_weights)
    # Correct the tiny summation error while keeping log probabilities aligned.
    row_sums = weights.sum(axis=1, keepdims=True)
    weights /= row_sums
    log_weights -= np.log(row_sums)
    return weights, log_weights


def _kl_probability(p: np.ndarray, q: np.ndarray) -> float:
    log_p = np.zeros_like(p, dtype=np.float64)
    np.log(p, out=log_p, where=p > 0.0)
    terms = np.zeros_like(p, dtype=np.float64)
    np.multiply(p, log_p - np.log(q), out=terms, where=p > 0.0)
    return float(terms.sum())


def _make_result(
    kernel: np.ndarray,
    log_prices: np.ndarray,
    *,
    q: np.ndarray,
    rho: float | None,
    method: str,
    iterations: int,
    converged: bool,
    max_log_price_residual: float,
    objective_override: float | None = None,
) -> OTResult:
    log_kernel = np.log(kernel)
    weights, log_weights = _softmax_with_log_probabilities(
        log_kernel + log_prices[None, :]
    )
    log_kernel_kl_terms = np.zeros_like(weights)
    np.multiply(
        weights,
        log_weights - log_kernel,
        out=log_kernel_kl_terms,
        where=weights > 0.0,
    )
    kernel_kl = float(log_kernel_kl_terms.sum(axis=1).mean())
    mean_weights = weights.mean(axis=0)
    prior_kl = _kl_probability(mean_weights, q)
    if objective_override is not None:
        objective = float(objective_override)
    elif rho is None:
        objective = kernel_kl
    else:
        objective = kernel_kl + float(rho) * prior_kl
    diagnostics = OTDiagnostics(
        method=method,
        rho=rho,
        target_prior=tuple(float(value) for value in q),
        iterations=iterations,
        converged=converged,
        objective=objective,
        kernel_kl=kernel_kl,
        prior_kl=prior_kl,
        max_row_residual=float(np.max(np.abs(weights.sum(axis=1) - 1.0))),
        max_target_marginal_residual=float(np.max(np.abs(mean_weights - q))),
        max_log_price_residual=float(max_log_price_residual),
    )
    return OTResult(weights, log_prices, diagnostics)


def relaxed_sinkhorn(
    kernel: np.ndarray,
    rho: float,
    q: tuple[float, ...] | np.ndarray = DEFAULT_Q,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> OTResult:
    """Minimize mean ``KL(P_i || K_i) + rho * KL(mean(P) || q)``.

    ``kernel`` must be a strictly positive row-stochastic ``(n, 4)`` array.
    Prices are found by deterministic log-domain updates. For ``rho == 0`` the
    original kernel is returned exactly, with zero iterations.
    """
    values = _validate_kernel(kernel)
    prior = _validate_prior(q)
    strength = _validate_rho(rho)
    tol, iteration_cap = _validate_solver_controls(tolerance, max_iterations)
    if strength == 0.0:
        result = _make_result(
            values,
            np.zeros(NUM_EXPERTS, dtype=np.float64),
            q=prior,
            rho=0.0,
            method="relaxed",
            iterations=0,
            converged=True,
            max_log_price_residual=0.0,
            objective_override=0.0,
        )
        # Preserve bitwise no-OT equivalence after validating the row sums.
        exact_kernel = np.array(values, dtype=np.float64, copy=True)
        exact_kernel.setflags(write=False)
        return OTResult(exact_kernel, np.zeros(NUM_EXPERTS), result.diagnostics)

    log_kernel = np.log(values)
    log_nq = np.log(float(len(values))) + np.log(prior)
    contraction = strength / (strength + 1.0)
    log_prices = np.zeros(NUM_EXPERTS, dtype=np.float64)
    residual = np.inf

    def price_update(prices: np.ndarray) -> np.ndarray:
        row_log_normalizers = logsumexp(log_kernel + prices[None, :], axis=1)
        log_column_base = logsumexp(
            log_kernel - row_log_normalizers[:, None], axis=0
        )
        return contraction * (log_nq - log_column_base)

    for iteration in range(1, iteration_cap + 1):
        log_prices = price_update(log_prices)
        residual = float(np.max(np.abs(price_update(log_prices) - log_prices)))
        if residual <= tol:
            result = _make_result(
                values,
                log_prices,
                q=prior,
                rho=strength,
                method="relaxed",
                iterations=iteration,
                converged=True,
                max_log_price_residual=residual,
            )
            if result.diagnostics.max_row_residual <= 1e-12:
                return result

    result = _make_result(
        values,
        log_prices,
        q=prior,
        rho=strength,
        method="relaxed",
        iterations=iteration_cap,
        converged=False,
        max_log_price_residual=residual,
    )
    raise OTConvergenceError(
        f"relaxed Sinkhorn did not converge in {iteration_cap} iterations "
        f"(log-price residual {residual:.3g})",
        result.diagnostics,
    )


def balanced_sinkhorn(
    kernel: np.ndarray,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> OTResult:
    """Balance rows to one and the four column averages to exactly 0.25."""
    values = _validate_kernel(kernel)
    prior = np.asarray(UNIFORM_Q, dtype=np.float64)
    tol, iteration_cap = _validate_solver_controls(tolerance, max_iterations)
    log_kernel = np.log(values)
    log_nq = np.log(float(len(values))) + np.log(prior)
    log_prices = np.zeros(NUM_EXPERTS, dtype=np.float64)
    residual = np.inf

    for iteration in range(1, iteration_cap + 1):
        weights, _ = _softmax_with_log_probabilities(
            log_kernel + log_prices[None, :]
        )
        column_mass = weights.sum(axis=0)
        column_residual = float(np.max(np.abs(column_mass / len(values) - prior)))
        if column_residual <= tol:
            return _make_result(
                values,
                log_prices,
                q=prior,
                rho=None,
                method="balanced",
                iterations=iteration - 1,
                converged=True,
                max_log_price_residual=column_residual,
            )

        row_log_normalizers = logsumexp(
            log_kernel + log_prices[None, :], axis=1
        )
        log_column_base = logsumexp(
            log_kernel - row_log_normalizers[:, None], axis=0
        )
        next_prices = log_nq - log_column_base
        residual = float(np.max(np.abs(next_prices - log_prices)))
        log_prices = next_prices

    result = _make_result(
        values,
        log_prices,
        q=prior,
        rho=None,
        method="balanced",
        iterations=iteration_cap,
        converged=False,
        max_log_price_residual=residual,
    )
    raise OTConvergenceError(
        f"balanced Sinkhorn did not converge in {iteration_cap} iterations "
        f"(column residual {result.diagnostics.max_target_marginal_residual:.3g})",
        result.diagnostics,
    )


def apply_log_bias(kernel: np.ndarray, log_bias: np.ndarray) -> np.ndarray:
    """Apply one four-expert log bias independently to each kernel row."""
    values = _validate_kernel(kernel)
    bias = _as_real_array(log_bias, name="log_bias")
    if bias.shape != (NUM_EXPERTS,):
        raise RidgeSinkhornOTError(f"log_bias must have shape ({NUM_EXPERTS},)")
    weights, _ = _softmax_with_log_probabilities(np.log(values) + bias[None, :])
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise RidgeSinkhornOTError("global bias produced invalid weights")
    return weights


def apply_prior_bias(
    kernel: np.ndarray,
    q: tuple[float, ...] | np.ndarray = DEFAULT_Q,
) -> np.ndarray:
    """Apply the explicit prior-only global-bias control ``K_i * q``."""
    prior = _validate_prior(q)
    return apply_log_bias(kernel, np.log(prior))


def fit_frozen_dual_prices(
    kernel: np.ndarray,
    rho: float,
    q: tuple[float, ...] | np.ndarray = DEFAULT_Q,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> FrozenDualPrices:
    """Fit relaxed OT prices on fit rows for row-independent future routing."""
    values = _validate_kernel(kernel)
    prior = _validate_prior(q)
    result = relaxed_sinkhorn(
        values,
        rho,
        prior,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    return FrozenDualPrices(
        rho=result.diagnostics.rho,
        q=tuple(float(value) for value in prior),
        log_prices=result.log_prices,
        diagnostics=result.diagnostics,
        fit_sample_count=len(values),
    )
