"""Shared validation helpers for array-only inner diagnostics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


class DiagnosticsError(ValueError):
    """Raised when diagnostic inputs or immutable artifacts violate the contract."""


EXPERT_NAMES = ("CE", "LAL", "BalancedSoftmax", "Mixup")


def finite_array(value: Any, *, name: str, ndim: int | None = None) -> np.ndarray:
    """Return a finite real numeric view, optionally enforcing dimensionality."""
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise DiagnosticsError(f"{name} must contain real numeric values")
    if ndim is not None and array.ndim != ndim:
        raise DiagnosticsError(f"{name} must have {ndim} dimensions, got {array.shape}")
    if not np.isfinite(array).all():
        raise DiagnosticsError(f"{name} contains non-finite values")
    return array.astype(np.float64, copy=False)


def aligned_labeled_arrays(
    logits: Any,
    labels: Any,
    *,
    num_experts: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate logits ``(N, E, C)`` and integer labels ``(N,)``."""
    values = finite_array(logits, name="logits", ndim=3)
    targets = np.asarray(labels)
    if values.shape[0] == 0 or values.shape[1] != num_experts or values.shape[2] < 2:
        raise DiagnosticsError(f"logits must have shape (N, {num_experts}, C), N>0, C>=2")
    if targets.shape != (len(values),) or not np.issubdtype(targets.dtype, np.integer):
        raise DiagnosticsError("labels must be an aligned integer vector")
    if np.any(targets < 0) or np.any(targets >= values.shape[2]):
        raise DiagnosticsError("labels fall outside the logit class dimension")
    return values, targets.astype(np.int64, copy=False)


def simplex_array(value: Any, *, name: str, rows: int | None = None) -> np.ndarray:
    """Validate finite row-stochastic arrays of expert weights, shape ``(N, 4)``."""
    weights = finite_array(value, name=name, ndim=2)
    if weights.shape[1] != 4 or (rows is not None and len(weights) != rows):
        raise DiagnosticsError(f"{name} must have shape (N, 4)")
    if len(weights) == 0 or np.any(weights < 0.0):
        raise DiagnosticsError(f"{name} must contain nonnegative weights and at least one row")
    if not np.allclose(weights.sum(axis=1), 1.0, rtol=0.0, atol=1e-10):
        raise DiagnosticsError(f"{name} rows must sum to one")
    return weights


def json_safe(value: Any) -> Any:
    """Convert nested values to JSON-safe Python values; undefined floats become null."""
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)
