"""Classification-aligned oracle targets and residual weight mapping.

This module contains array-only pieces for the Ridge/Sinkhorn study. It does
not load datasets or inspect evaluation results. The oracle solves one small
convex multiclass hinge problem per labeled row with SciPy's deterministic
SLSQP implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy
from scipy.optimize import minimize


EXPERT_ORDER = ("CE", "LAL", "BalancedSoftmax", "Mixup")
NUM_EXPERTS = len(EXPERT_ORDER)
FIXED_007_ANCHOR = (0.0, 0.25, 0.5, 0.25)
DEFAULT_SOLVER_FTOL = 1e-10
DEFAULT_SOLVER_MAX_ITERATIONS = 100
FEASIBILITY_TOLERANCE = 1e-7


class RidgeSinkhornOracleError(ValueError):
    """Raised when oracle inputs or constrained solutions violate the contract."""


@dataclass(frozen=True)
class OracleTargetResult:
    """Oracle solutions, residual targets, and solver diagnostics for a batch."""

    oracle_weights: np.ndarray
    residuals: np.ndarray
    diagnostics: dict[str, Any]


def _as_real_finite_array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise RidgeSinkhornOracleError(f"{name} must contain real numeric values")
    if not np.isfinite(array).all():
        raise RidgeSinkhornOracleError(f"{name} must contain only finite values")
    return array.astype(np.float64, copy=False)


def _validate_logits(logits: Any) -> np.ndarray:
    array = _as_real_finite_array(logits, name="logits")
    if array.ndim != 3:
        raise RidgeSinkhornOracleError(
            f"logits must have shape (samples, {NUM_EXPERTS}, classes), got {array.shape}"
        )
    if array.shape[0] == 0:
        raise RidgeSinkhornOracleError("logits must contain at least one sample")
    if array.shape[1] != NUM_EXPERTS:
        raise RidgeSinkhornOracleError(
            f"logits must contain exactly {NUM_EXPERTS} experts in frozen order {EXPERT_ORDER}"
        )
    if array.shape[2] < 2:
        raise RidgeSinkhornOracleError("logits must contain at least two classes")
    return array


def _validate_labels(labels: Any, *, num_samples: int, num_classes: int) -> np.ndarray:
    array = np.asarray(labels)
    if array.ndim != 1 or array.shape[0] != num_samples:
        raise RidgeSinkhornOracleError(
            f"labels must have shape ({num_samples},), got {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise RidgeSinkhornOracleError("labels must contain real integer class indices")
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        raise RidgeSinkhornOracleError(
            "labels must contain finite integer-valued class indices"
        )
    labels_array = array.astype(np.int64, copy=False)
    if np.any(labels_array < 0) or np.any(labels_array >= num_classes):
        raise RidgeSinkhornOracleError(f"labels must be in [0, {num_classes})")
    return labels_array


def _validate_anchor(anchor: Any) -> np.ndarray:
    array = _as_real_finite_array(anchor, name="anchor")
    if array.shape != (NUM_EXPERTS,):
        raise RidgeSinkhornOracleError(f"anchor must have shape ({NUM_EXPERTS},), got {array.shape}")
    if np.any(array < 0.0):
        raise RidgeSinkhornOracleError("anchor weights must be nonnegative")
    if not np.isclose(array.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise RidgeSinkhornOracleError("anchor weights must sum to one")
    return array


def _validate_penalty(penalty: Any) -> float:
    if isinstance(penalty, (bool, np.bool_)):
        raise RidgeSinkhornOracleError("penalty must be finite and positive")
    try:
        value = float(penalty)
    except (TypeError, ValueError) as exc:
        raise RidgeSinkhornOracleError("penalty must be finite and positive") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise RidgeSinkhornOracleError("penalty must be finite and positive")
    return value


def _validate_solver_options(ftol: Any, max_iterations: Any) -> tuple[float, int]:
    if isinstance(ftol, (bool, np.bool_)):
        raise RidgeSinkhornOracleError("ftol must be finite and positive")
    try:
        tolerance = float(ftol)
    except (TypeError, ValueError) as exc:
        raise RidgeSinkhornOracleError("ftol must be finite and positive") from exc
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise RidgeSinkhornOracleError("ftol must be finite and positive")
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(
        max_iterations, (int, np.integer)
    ):
        raise RidgeSinkhornOracleError("max_iterations must be a positive integer")
    iterations = int(max_iterations)
    if iterations < 1:
        raise RidgeSinkhornOracleError("max_iterations must be a positive integer")
    return tolerance, iterations


def _margin_differences(row_logits: np.ndarray, label: int) -> np.ndarray:
    """Return competitor-minus-label logits as (classes - 1, experts)."""
    wrong = np.arange(row_logits.shape[1]) != label
    differences = (row_logits[:, wrong] - row_logits[:, label, None]).T
    if not np.isfinite(differences).all():
        raise RidgeSinkhornOracleError("logit differences overflowed finite float64")
    return differences


def _solve_one_oracle(
    differences: np.ndarray,
    anchor: np.ndarray,
    penalty: float,
    margin: float,
    *,
    ftol: float,
    max_iterations: int,
    row_index: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve one convex epigraph problem and verify its primal solution."""
    score_scale = max(1.0, float(np.max(np.abs(differences))), abs(margin))
    scaled_differences = differences / score_scale
    margin_offset_scaled = margin / score_scale
    anchor_margins_scaled = margin_offset_scaled + scaled_differences @ anchor
    anchor_hinge_scaled = max(0.0, float(np.max(anchor_margins_scaled)))

    # At a zero-hinge anchor, the anchor has objective zero, the global lower
    # bound. Return its bit-identical weights so the residual is exactly zero.
    if anchor_hinge_scaled == 0.0:
        weights = anchor.copy()
        margins_scaled = margin_offset_scaled + scaled_differences @ weights
        hinge_scaled = max(0.0, float(np.max(margins_scaled)))
        objective = penalty * 0.5 * float(np.dot(weights - anchor, weights - anchor))
        return weights, {
            "success": True,
            "message": "exact zero-hinge anchor optimum",
            "iterations": 0,
            "exact_anchor_shortcut": True,
            "score_scale": score_scale,
            "objective": objective,
            "solver_objective": objective,
            "objective_residual": 0.0,
            "hinge_slack": hinge_scaled * score_scale,
            "simplex_residual": abs(float(weights.sum()) - 1.0),
            "lower_bound_violation": float(max(0.0, -float(weights.min()))),
            "upper_bound_violation": float(max(0.0, float(weights.max()) - 1.0)),
            "t_nonnegative_violation": 0.0,
            "max_anchor_margin": float(np.max(anchor_margins_scaled) * score_scale),
            "max_solution_margin": float(np.max(margins_scaled) * score_scale),
            "normalized_slack_violation": 0.0,
        }

    # Scaling the epigraph variable and margins avoids giving SLSQP enormous
    # constraint magnitudes when logits have a large common scale.
    initial = np.concatenate((anchor, np.array([anchor_hinge_scaled])))
    penalty_scaled = penalty / score_scale

    def objective(parameters: np.ndarray) -> float:
        difference = parameters[:NUM_EXPERTS] - anchor
        return float(
            parameters[NUM_EXPERTS]
            + 0.5 * penalty_scaled * np.dot(difference, difference)
        )

    def objective_jacobian(parameters: np.ndarray) -> np.ndarray:
        difference = parameters[:NUM_EXPERTS] - anchor
        return np.concatenate((penalty_scaled * difference, np.array([1.0])))

    def inequalities(parameters: np.ndarray) -> np.ndarray:
        return (
            parameters[NUM_EXPERTS]
            - margin_offset_scaled
            - scaled_differences @ parameters[:NUM_EXPERTS]
        )

    constraint_jacobian = np.column_stack(
        (-scaled_differences, np.ones(scaled_differences.shape[0], dtype=np.float64))
    )
    constraints = (
        {
            "type": "eq",
            "fun": lambda parameters: float(parameters[:NUM_EXPERTS].sum() - 1.0),
            "jac": lambda parameters: np.array([1.0, 1.0, 1.0, 1.0, 0.0]),
        },
        {
            "type": "ineq",
            "fun": inequalities,
            "jac": lambda parameters: constraint_jacobian,
        },
    )
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        jac=objective_jacobian,
        bounds=[(0.0, 1.0)] * NUM_EXPERTS + [(0.0, None)],
        constraints=constraints,
        options={"ftol": ftol, "maxiter": max_iterations, "disp": False},
    )

    parameters = np.asarray(result.x, dtype=np.float64)
    if parameters.shape != (NUM_EXPERTS + 1,) or not np.isfinite(parameters).all():
        raise RidgeSinkhornOracleError(
            f"oracle solver returned invalid parameters for row {row_index}: {result.message}"
        )
    weights = parameters[:NUM_EXPERTS]
    epigraph = float(parameters[NUM_EXPERTS])
    margins_scaled = margin_offset_scaled + scaled_differences @ weights
    hinge_scaled = max(0.0, float(np.max(margins_scaled)))
    normalized_slack_violation = max(
        0.0, float(np.max(margins_scaled - epigraph))
    )
    lower_violation = max(0.0, -float(np.min(weights)))
    upper_violation = max(0.0, float(np.max(weights) - 1.0))
    simplex_residual = abs(float(weights.sum()) - 1.0)
    t_violation = max(0.0, -epigraph)
    objective_at_solution = float(
        hinge_scaled * score_scale
        + 0.5 * penalty * np.dot(weights - anchor, weights - anchor)
    )
    objective_at_solver_parameters = float(
        epigraph * score_scale
        + 0.5 * penalty * np.dot(weights - anchor, weights - anchor)
    )
    solver_objective = float(result.fun) * score_scale
    objective_residual = abs(solver_objective - objective_at_solver_parameters)

    # SLSQP must report convergence and its solution must satisfy every primal
    # condition. The normalized slack threshold remains meaningful across logit
    # scales; simplex, box, and t residuals have unit scale.
    max_primal_residual = max(
        simplex_residual,
        lower_violation,
        upper_violation,
        t_violation,
        normalized_slack_violation,
        abs(epigraph - hinge_scaled),
    )
    if not result.success or max_primal_residual > FEASIBILITY_TOLERANCE:
        raise RidgeSinkhornOracleError(
            "oracle solver failed for row "
            f"{row_index}: success={bool(result.success)}, "
            f"primal_residual={max_primal_residual:.3g}, message={result.message}"
        )
    if not np.isfinite(objective_at_solution) or not np.isfinite(solver_objective):
        raise RidgeSinkhornOracleError(
            f"oracle objective is non-finite for row {row_index}"
        )
    objective_tolerance = max(1e-8, 1e-7 * abs(objective_at_solver_parameters))
    if objective_residual > objective_tolerance:
        raise RidgeSinkhornOracleError(
            f"oracle objective check failed for row {row_index}: "
            f"residual={objective_residual:.3g}"
        )

    return weights, {
        "success": True,
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", 0)),
        "exact_anchor_shortcut": False,
        "score_scale": score_scale,
        "objective": objective_at_solution,
        "solver_objective": solver_objective,
        "objective_residual": objective_residual,
        "hinge_slack": (epigraph - hinge_scaled) * score_scale,
        "simplex_residual": simplex_residual,
        "lower_bound_violation": lower_violation,
        "upper_bound_violation": upper_violation,
        "t_nonnegative_violation": t_violation,
        "max_anchor_margin": float(np.max(anchor_margins_scaled) * score_scale),
        "max_solution_margin": float(np.max(margins_scaled) * score_scale),
        "normalized_slack_violation": normalized_slack_violation,
    }


def compute_oracle_targets(
    logits: Any,
    labels: Any,
    anchor: Any = FIXED_007_ANCHOR,
    penalty: float = 1.0,
    *,
    margin: float = 0.0,
    ftol: float = DEFAULT_SOLVER_FTOL,
    max_iterations: int = DEFAULT_SOLVER_MAX_ITERATIONS,
) -> OracleTargetResult:
    """Compute per-row hinge-oracle weights and residuals from expert logits.

    For each row this solves ``min_w,t t + penalty/2 * ||w-anchor||^2`` over
    the four-expert simplex, subject to ``t >= 0`` and ``t`` exceeding the
    requested margin plus every weighted competitor-minus-label logit margin.
    Labels appear only here, during supervised target construction. The
    primary target uses zero margin; ``margin=1.0`` supports the prespecified
    target-sensitivity diagnostic.
    """
    logits_array = _validate_logits(logits)
    labels_array = _validate_labels(
        labels,
        num_samples=logits_array.shape[0],
        num_classes=logits_array.shape[2],
    )
    anchor_array = _validate_anchor(anchor)
    penalty_value = _validate_penalty(penalty)
    if isinstance(margin, (bool, np.bool_)):
        raise RidgeSinkhornOracleError("margin must be finite and nonnegative")
    try:
        margin_value = float(margin)
    except (TypeError, ValueError) as exc:
        raise RidgeSinkhornOracleError("margin must be finite and nonnegative") from exc
    if not np.isfinite(margin_value) or margin_value < 0.0:
        raise RidgeSinkhornOracleError("margin must be finite and nonnegative")
    tolerance, iteration_limit = _validate_solver_options(ftol, max_iterations)

    oracle_weights = np.empty((logits_array.shape[0], NUM_EXPERTS), dtype=np.float64)
    row_diagnostics: list[dict[str, Any]] = []
    for row_index, (row_logits, label) in enumerate(zip(logits_array, labels_array)):
        differences = _margin_differences(row_logits, int(label))
        weights, row_diagnostic = _solve_one_oracle(
            differences,
            anchor_array,
            penalty_value,
            margin_value,
            ftol=tolerance,
            max_iterations=iteration_limit,
            row_index=row_index,
        )
        oracle_weights[row_index] = weights
        row_diagnostics.append(row_diagnostic)

    residuals = oracle_weights - anchor_array[None, :]
    diagnostic_keys = (
        "iterations",
        "objective",
        "solver_objective",
        "objective_residual",
        "hinge_slack",
        "simplex_residual",
        "lower_bound_violation",
        "upper_bound_violation",
        "t_nonnegative_violation",
        "max_anchor_margin",
        "max_solution_margin",
        "normalized_slack_violation",
    )
    diagnostic_arrays = {
        key: [row[key] for row in row_diagnostics] for key in diagnostic_keys
    }
    diagnostics: dict[str, Any] = {
        "solver": "scipy.optimize.minimize(method='SLSQP')",
        "scipy_version": scipy.__version__,
        "expert_order": EXPERT_ORDER,
        "penalty": penalty_value,
        "margin": margin_value,
        "ftol": tolerance,
        "max_iterations": iteration_limit,
        "row_count": int(logits_array.shape[0]),
        "all_succeeded": all(row["success"] for row in row_diagnostics),
        "exact_anchor_rows": int(
            sum(bool(row["exact_anchor_shortcut"]) for row in row_diagnostics)
        ),
        "max_primal_residual": float(
            max(
                max(row["simplex_residual"], row["lower_bound_violation"],
                    row["upper_bound_violation"], row["t_nonnegative_violation"],
                    row["normalized_slack_violation"])
                for row in row_diagnostics
            )
        ),
        "max_objective_residual": float(
            max(row["objective_residual"] for row in row_diagnostics)
        ),
        "row_diagnostics": row_diagnostics,
        **diagnostic_arrays,
    }
    return OracleTargetResult(
        oracle_weights=oracle_weights,
        residuals=residuals,
        diagnostics=diagnostics,
    )


def _validate_last_dimension(values: Any, *, name: str) -> np.ndarray:
    array = _as_real_finite_array(values, name=name)
    if array.ndim not in (1, 2) or array.shape[-1] != NUM_EXPERTS:
        raise RidgeSinkhornOracleError(
            f"{name} must have shape ({NUM_EXPERTS},) or (samples, {NUM_EXPERTS}), got {array.shape}"
        )
    if array.ndim == 2 and array.shape[0] == 0:
        raise RidgeSinkhornOracleError(f"{name} must contain at least one sample")
    return array


def project_onto_simplex(values: Any) -> np.ndarray:
    """Euclidean-project one or more four-vectors onto the probability simplex."""
    array = _validate_last_dimension(values, name="values")
    was_vector = array.ndim == 1
    rows = array[None, :] if was_vector else array

    sorted_rows = np.sort(rows, axis=1)[:, ::-1]
    cumulative = np.cumsum(sorted_rows, axis=1) - 1.0
    divisors = np.arange(1, NUM_EXPERTS + 1, dtype=np.float64)[None, :]
    admissible = sorted_rows - cumulative / divisors > 0.0
    rho = admissible.sum(axis=1) - 1
    theta = cumulative[np.arange(len(rows)), rho] / (rho + 1.0)
    projected = np.maximum(rows - theta[:, None], 0.0)
    # Correct only the final floating-point sum error on the largest component.
    correction = 1.0 - projected.sum(axis=1)
    largest = projected.argmax(axis=1)
    projected[np.arange(len(projected)), largest] += correction
    if (
        not np.isfinite(projected).all()
        or np.any(projected < -1e-14)
        or not np.allclose(projected.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise RidgeSinkhornOracleError("simplex projection produced invalid weights")
    projected[projected < 0.0] = 0.0
    return projected[0] if was_vector else projected


def residuals_to_weights(
    predicted_residuals: Any,
    anchor: Any = FIXED_007_ANCHOR,
    scale: float = 1.0,
) -> np.ndarray:
    """Map predicted residuals to simplex weights by Euclidean projection."""
    residuals = _validate_last_dimension(predicted_residuals, name="predicted_residuals")
    anchor_array = _validate_anchor(anchor)
    if isinstance(scale, (bool, np.bool_)):
        raise RidgeSinkhornOracleError("scale must be finite and nonnegative")
    try:
        scale_value = float(scale)
    except (TypeError, ValueError) as exc:
        raise RidgeSinkhornOracleError("scale must be finite and nonnegative") from exc
    if not np.isfinite(scale_value) or scale_value < 0.0:
        raise RidgeSinkhornOracleError("scale must be finite and nonnegative")
    shifted = anchor_array + scale_value * residuals
    if not np.isfinite(shifted).all():
        raise RidgeSinkhornOracleError("scaled residual weights overflowed finite float64")
    return project_onto_simplex(shifted)


def _validate_simplex_weights(weights: Any) -> np.ndarray:
    array = _validate_last_dimension(weights, name="weights")
    if np.any(array < 0.0):
        raise RidgeSinkhornOracleError("weights must be nonnegative")
    if not np.allclose(array.sum(axis=-1), 1.0, rtol=0.0, atol=1e-12):
        raise RidgeSinkhornOracleError("weights must sum to one along the expert axis")
    return array


def smooth_positive_kernel(weights: Any, epsilon: float = 1e-6) -> np.ndarray:
    """Mix simplex weights with uniform mass so every kernel entry is positive."""
    array = _validate_simplex_weights(weights)
    if isinstance(epsilon, (bool, np.bool_)):
        raise RidgeSinkhornOracleError("epsilon must be finite and in (0, 1)")
    try:
        epsilon_value = float(epsilon)
    except (TypeError, ValueError) as exc:
        raise RidgeSinkhornOracleError("epsilon must be finite and in (0, 1)") from exc
    if not np.isfinite(epsilon_value) or not 0.0 < epsilon_value < 1.0:
        raise RidgeSinkhornOracleError("epsilon must be finite and in (0, 1)")
    smoothed = (1.0 - epsilon_value) * array + epsilon_value / NUM_EXPERTS
    if (
        not np.isfinite(smoothed).all()
        or np.any(smoothed <= 0.0)
        or not np.allclose(smoothed.sum(axis=-1), 1.0, rtol=0.0, atol=1e-12)
    ):
        raise RidgeSinkhornOracleError("positive-kernel smoothing produced invalid weights")
    return smoothed


__all__ = [
    "DEFAULT_SOLVER_FTOL",
    "DEFAULT_SOLVER_MAX_ITERATIONS",
    "EXPERT_ORDER",
    "FEASIBILITY_TOLERANCE",
    "FIXED_007_ANCHOR",
    "NUM_EXPERTS",
    "OracleTargetResult",
    "RidgeSinkhornOracleError",
    "compute_oracle_targets",
    "project_onto_simplex",
    "residuals_to_weights",
    "smooth_positive_kernel",
]
