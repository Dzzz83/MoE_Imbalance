"""Numerically stable expert-combination primitives.

The functions distinguish weighted-logit and weighted-probability ensembles.
They validate finite numeric inputs and convex weights, but do not know any
experiment's population, labels, metrics, or provenance rules.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts.analysis.validation import (
    validate_expert_weights,
    validate_numeric_array,
)


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax over the final axis using max-shifting.

    This is intentionally a math-only primitive.  Callers with stricter
    finite-value or shape contracts validate before calling it; preserving that
    separation keeps the legacy feature softmax's non-finite propagation.
    """
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def _validate_logit_stack(
    logits: Any,
    *,
    num_experts: int | None,
    min_classes: int,
    cast_dtype: Any | None,
    name: str,
    error_type: type[Exception],
) -> np.ndarray:
    array = validate_numeric_array(
        logits,
        name=name,
        finite=True,
        cast_dtype=cast_dtype,
        error_type=error_type,
    )
    if array.ndim != 3:
        if num_experts is not None:
            expected = f"(samples, {num_experts}, classes)"
        else:
            expected = "(samples, experts, classes)"
        raise error_type(f"{name} must have shape {expected}, got {array.shape}")
    if array.shape[0] == 0:
        raise error_type(f"{name} must contain at least one sample")
    if array.shape[1] == 0 or (
        num_experts is not None and array.shape[1] != num_experts
    ):
        if num_experts is None:
            raise error_type(f"{name} must contain at least one expert")
        raise error_type(
            f"{name} must contain exactly {num_experts} experts in the frozen order"
        )
    if array.shape[2] < min_classes:
        raise error_type(f"{name} must contain at least {min_classes} classes")
    return array


def combine_weighted_logits(
    logits: Any,
    weights: Any,
    *,
    num_experts: int | None = None,
    min_classes: int = 1,
    weight_sum_atol: float = 1e-12,
    logit_cast_dtype: Any | None = None,
    logits_name: str = "logits",
    weights_name: str = "routing weights",
    error_type: type[Exception] = ValueError,
) -> np.ndarray:
    """Return ``sum_e weights[n,e] * logits[n,e,c]`` for each sample."""
    array = _validate_logit_stack(
        logits,
        num_experts=num_experts,
        min_classes=min_classes,
        cast_dtype=logit_cast_dtype,
        name=logits_name,
        error_type=error_type,
    )
    expert_count = array.shape[1]
    validated_weights = validate_expert_weights(
        weights,
        num_experts=expert_count,
        num_samples=array.shape[0],
        allow_vector=True,
        sum_atol=weight_sum_atol,
        name=weights_name,
        error_type=error_type,
    )
    combined = np.einsum("ne,nec->nc", validated_weights, array)
    if not np.isfinite(combined).all():
        raise error_type("combined logits are non-finite")
    return combined


def combine_weighted_probabilities(
    logits: Any,
    weights: Any,
    *,
    num_experts: int | None = None,
    min_classes: int = 1,
    weight_sum_atol: float = 1e-12,
    logit_cast_dtype: Any | None = None,
    logits_name: str = "logits",
    weights_name: str = "routing weights",
    error_type: type[Exception] = ValueError,
) -> np.ndarray:
    """Return ``sum_e weights[n,e] * softmax(logits[n,e])`` per sample."""
    array = _validate_logit_stack(
        logits,
        num_experts=num_experts,
        min_classes=min_classes,
        cast_dtype=logit_cast_dtype,
        name=logits_name,
        error_type=error_type,
    )
    expert_count = array.shape[1]
    validated_weights = validate_expert_weights(
        weights,
        num_experts=expert_count,
        num_samples=array.shape[0],
        allow_vector=True,
        sum_atol=weight_sum_atol,
        name=weights_name,
        error_type=error_type,
    )
    combined = np.einsum(
        "ne,nec->nc", validated_weights, stable_softmax(array)
    )
    if not np.isfinite(combined).all():
        raise error_type("combined probabilities are non-finite")
    return combined


__all__ = [
    "combine_weighted_logits",
    "combine_weighted_probabilities",
    "stable_softmax",
]
