"""Numerical comparisons between row-stochastic routing profiles."""

from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import DiagnosticsError, finite_array, simplex_array


def _profile_array(value: Any, *, name: str) -> np.ndarray:
    array = finite_array(value, name=name)
    if array.ndim == 1:
        if array.shape != (4,) or np.any(array < 0.0) or not np.isclose(array.sum(), 1.0, rtol=0.0, atol=1e-10):
            raise DiagnosticsError(f"{name} must be a four-expert probability profile")
        return array[None, :]
    if array.ndim == 2:
        return simplex_array(array, name=name)
    raise DiagnosticsError(f"{name} must have shape (4,) or (N, 4)")


def _entropy_bits(values: np.ndarray) -> float:
    positive = values > 0.0
    return float(-np.sum(values[positive] * np.log2(values[positive])))


def compare_profiles(profile_a: Any, profile_b: Any) -> dict[str, Any]:
    """Compare two allocation vectors or aligned ``(N, 4)`` profile matrices.

    Matrix rows represent the same comparison units, such as classes in
    canonical class order; they need not be the same samples or fold
    population. Distances and cosine similarity are averaged over corresponding
    rows, so a row permutation changes the result. Interpret comparisons across
    different fold input populations descriptively: they combine router and
    population differences. Pearson correlation is computed over paired
    flattened entries, is undefined for constant inputs, and is returned as
    ``None`` in that case. Entropy difference is the signed difference between
    entropies of the two mean allocation vectors, in bits.
    """
    first = _profile_array(profile_a, name="profile_a")
    second = _profile_array(profile_b, name="profile_b")
    if first.shape != second.shape:
        raise DiagnosticsError("profiles must have identical shapes and aligned rows")
    delta = first - second
    first_norm = np.linalg.norm(first, axis=1)
    second_norm = np.linalg.norm(second, axis=1)
    cosine = np.sum(first * second, axis=1) / (first_norm * second_norm)
    flat_a = first.ravel()
    flat_b = second.ravel()
    if np.std(flat_a) == 0.0 or np.std(flat_b) == 0.0:
        correlation: float | None = None
    else:
        correlation = float(np.corrcoef(flat_a, flat_b)[0, 1])
    preferred_agreement = float(
        np.mean(np.argmax(first, axis=1) == np.argmax(second, axis=1))
    )
    mean_a = first.mean(axis=0)
    mean_b = second.mean(axis=0)
    entropy_a = _entropy_bits(mean_a)
    entropy_b = _entropy_bits(mean_b)
    return {
        "n_profiles": int(len(first)),
        "l1_distance": float(np.mean(np.abs(delta).sum(axis=1))),
        "l2_distance": float(np.mean(np.linalg.norm(delta, axis=1))),
        "cosine_similarity": float(np.mean(cosine)),
        "correlation": correlation,
        "preferred_agreement": preferred_agreement,
        "entropy_a_bits": entropy_a,
        "entropy_b_bits": entropy_b,
        "entropy_difference_bits": float(entropy_a - entropy_b),
    }


def selective_gate(
    ridge: Any,
    sinkhorn: Any,
    tau: float,
    anchor: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend Ridge and Sinkhorn rows by L1 distance from a fixed anchor.

    Inputs have shape ``ridge=(N,4)``, ``sinkhorn=(N,4)``, and
    ``anchor=(4,)``; all must be row-stochastic expert allocations. ``tau``
    must be finite and positive. The returned gate vector has shape ``(N,)``
    and follows ``g_i = min(1, ||ridge_i-anchor||_1 / tau)``. The returned
    weights are ``(1-g_i)*ridge_i + g_i*sinkhorn_i``.
    """
    first = simplex_array(ridge, name="ridge")
    second = simplex_array(sinkhorn, name="sinkhorn", rows=len(first))
    anchor_array = finite_array(anchor, name="anchor", ndim=1)
    if anchor_array.shape != (4,) or np.any(anchor_array < 0.0) or not np.isclose(
        anchor_array.sum(), 1.0, rtol=0.0, atol=1e-10
    ):
        raise DiagnosticsError("anchor must be a four-expert probability vector")
    if isinstance(tau, (bool, np.bool_)):
        raise DiagnosticsError("tau must be a positive finite scalar")
    tau_array = finite_array(tau, name="tau")
    if tau_array.ndim != 0 or float(tau_array) <= 0.0:
        raise DiagnosticsError("tau must be a positive finite scalar")
    gates = np.minimum(1.0, np.abs(first - anchor_array[None, :]).sum(axis=1) / float(tau_array))
    blended = (1.0 - gates[:, None]) * first + gates[:, None] * second
    return blended, gates
