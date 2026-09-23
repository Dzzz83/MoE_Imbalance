"""Shared low-level validation for analysis arrays.

These helpers validate representation-level properties only.  They do not
encode a task's population, metric, fold, or provenance rules; callers retain
those checks and choose the exception type that belongs to their public API.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _raise(error_type: type[Exception], message: str) -> None:
    raise error_type(message)


def validate_numeric_array(
    value: Any,
    *,
    name: str = "array",
    ndim: int | None = None,
    shape: tuple[int | None, ...] | None = None,
    finite: bool = True,
    cast_dtype: Any | None = None,
    error_type: type[Exception] = ValueError,
) -> np.ndarray:
    """Return ``value`` as an array after real-numeric validation.

    ``shape`` may contain ``None`` entries for unconstrained dimensions.  No
    values are changed unless ``cast_dtype`` is explicitly supplied.
    """
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        _raise(error_type, f"{name} must be a numeric array")
        raise AssertionError("unreachable") from exc

    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        _raise(error_type, f"{name} must contain real numeric values")

    if ndim is not None and array.ndim != ndim:
        if ndim == 1:
            message = f"{name} must be one-dimensional"
        else:
            message = f"{name} must have {ndim} dimensions, got shape {array.shape}"
        _raise(error_type, message)

    if shape is not None:
        expected_shape = tuple(shape)
        matches = array.ndim == len(expected_shape) and all(
            expected is None or actual == expected
            for actual, expected in zip(array.shape, expected_shape)
        )
        if not matches:
            _raise(
                error_type,
                f"{name} must have shape {expected_shape}, got {array.shape}",
            )

    if finite and not np.isfinite(array).all():
        _raise(error_type, f"{name} contains non-finite values")

    if cast_dtype is not None:
        try:
            return array.astype(cast_dtype, copy=False)
        except (TypeError, ValueError) as exc:
            _raise(error_type, f"{name} cannot be converted to {cast_dtype}")
            raise AssertionError("unreachable") from exc
    return array


def validate_integer_vector(
    value: Any,
    *,
    name: str = "values",
    shape: tuple[int | None, ...] | None = None,
    ndim: int = 1,
    lower_bound: int | None = None,
    upper_bound: int | None = None,
    cast_dtype: Any = np.int64,
    error_type: type[Exception] = ValueError,
) -> np.ndarray:
    """Validate finite integer-valued numeric entries and return an integer array."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        _raise(error_type, f"{name} must contain integer values")
        raise AssertionError("unreachable") from exc

    if shape is not None:
        expected_shape = tuple(shape)
        matches = array.ndim == len(expected_shape) and all(
            expected is None or actual == expected
            for actual, expected in zip(array.shape, expected_shape)
        )
        if not matches:
            _raise(
                error_type,
                f"{name} must have shape {expected_shape}, got {array.shape}",
            )
    elif array.ndim != ndim:
        if ndim == 1:
            _raise(error_type, f"{name} must be one-dimensional")
        _raise(error_type, f"{name} must have {ndim} dimensions, got {array.shape}")

    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        _raise(error_type, f"{name} must contain real numeric integer values")
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        _raise(error_type, f"{name} must contain finite integer-valued entries")

    try:
        result = array.astype(cast_dtype, copy=False)
    except (TypeError, ValueError) as exc:
        _raise(error_type, f"{name} cannot be converted to an integer array")
        raise AssertionError("unreachable") from exc

    if lower_bound is not None and np.any(result < lower_bound):
        _raise(error_type, f"{name} must be greater than or equal to {lower_bound}")
    if upper_bound is not None and np.any(result >= upper_bound):
        _raise(error_type, f"{name} must be in [0, {upper_bound})")
    return result


def validate_expert_weights(
    value: Any,
    *,
    num_experts: int,
    num_samples: int | None = None,
    allow_vector: bool = False,
    sum_atol: float = 1e-12,
    name: str = "expert weights",
    error_type: type[Exception] = ValueError,
) -> np.ndarray:
    """Validate non-negative row-stochastic expert weights.

    A one-dimensional global vector is accepted only when ``allow_vector`` is
    true.  When ``num_samples`` is supplied, that vector is repeated into a
    row-wise matrix so combination functions have one stable representation.
    The returned values are ``float64`` to preserve the existing analysis
    callers' accumulation behavior.
    """
    if num_experts < 1:
        _raise(error_type, "num_experts must be positive")
    if not np.isfinite(float(sum_atol)) or float(sum_atol) < 0.0:
        _raise(error_type, "sum_atol must be finite and non-negative")

    array = validate_numeric_array(
        value,
        name=name,
        finite=True,
        cast_dtype=np.float64,
        error_type=error_type,
    )
    if array.ndim == 1:
        if not allow_vector:
            _raise(error_type, f"{name} must have two dimensions")
        if array.shape != (num_experts,):
            _raise(error_type, f"{name} must have shape ({num_experts},)")
        if num_samples is not None:
            array = np.repeat(array[None, :], num_samples, axis=0)
    elif array.ndim != 2:
        _raise(error_type, f"{name} must have two dimensions")

    if array.ndim == 2:
        if array.shape[1] != num_experts:
            if num_samples is None:
                _raise(error_type, f"{name} must have {num_experts} expert columns")
            _raise(
                error_type,
                f"{name} must have shape ({num_samples}, {num_experts})",
            )
        if num_samples is not None and array.shape[0] != num_samples:
            _raise(
                error_type,
                f"{name} must have shape ({num_samples}, {num_experts})",
            )

    if np.any(array < 0.0):
        _raise(error_type, f"{name} must be non-negative")
    row_sums = array.sum(axis=-1)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=float(sum_atol)):
        _raise(
            error_type,
            f"{name} must sum to one per row within {float(sum_atol):g}",
        )
    return array


__all__ = [
    "validate_expert_weights",
    "validate_integer_vector",
    "validate_numeric_array",
]
