"""Synthetic tests for the Task 3F-E target diagnostic seams."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from scripts.task3f_target_diagnostics import (  # noqa: E402
    Task3FETargetDiagnosticError,
    compute_margin_contributions,
    compute_contribution_targets,
    compute_strongest_incorrect_classes,
    convex_weight_perturbation,
)


def test_margin_contributions_use_uniform_strongest_incorrect_class():
    # The uniform logits are [1.125, 0.375, 0.0], so class 1 is c* for y=0.
    logits = np.array(
        [
            [
                [2.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.5, 0.0],
                [1.5, 0.0, 0.0],
            ]
        ],
        dtype=np.float64,
    )
    labels = np.array([0], dtype=np.int64)

    assert np.array_equal(compute_strongest_incorrect_classes(logits, labels), [1])
    # Expert margins against class 1 are [2, -1, 0.5, 1.5]; the uniform
    # margin is 0.75, giving [1.25, -1.75, -0.25, 0.75].
    assert np.allclose(
        compute_margin_contributions(logits, labels),
        [[1.25, -1.75, -0.25, 0.75]],
    )
    assert np.allclose(compute_margin_contributions(logits, labels).sum(axis=1), 0.0)


def test_supervised_log_probability_target_matches_analytic_direction():
    logits = np.array(
        [
            [
                [2.0, 0.0, -1.0],
                [0.5, 1.0, -0.5],
                [1.0, -0.5, 0.0],
                [1.5, 0.5, -0.25],
            ]
        ],
        dtype=np.float64,
    )
    labels = np.array([0], dtype=np.int64)
    target = compute_contribution_targets(logits, labels)
    ensemble = logits[0].mean(axis=0)
    probabilities = np.exp(ensemble - ensemble.max())
    probabilities /= probabilities.sum()
    delta = logits[0, 0] - ensemble
    expected = delta[0] - float(np.dot(probabilities, delta))

    assert target.shape == (1, 4)
    assert target[0, 0] == pytest.approx(expected)
    assert np.allclose(target.sum(axis=1), 0.0)


def test_convex_weight_perturbation_matches_frozen_formula():
    weights = convex_weight_perturbation(2, expert_index=2, epsilon=0.5)
    assert weights.shape == (2, 4)
    assert np.allclose(weights, [[0.125, 0.125, 0.625, 0.125]] * 2)
    assert np.allclose(weights.sum(axis=1), 1.0)

    pure_expert = convex_weight_perturbation(
        np.full(4, 0.25), expert_index=1, epsilon=1.0
    )
    assert pure_expert.shape == (4,)
    assert np.array_equal(pure_expert, [0.0, 1.0, 0.0, 0.0])


def test_convex_weight_perturbation_rejects_nonuniform_base_or_invalid_epsilon():
    with pytest.raises(Task3FETargetDiagnosticError, match="uniform"):
        convex_weight_perturbation(np.array([0.1, 0.2, 0.3, 0.4]), 0, 0.5)
    with pytest.raises(Task3FETargetDiagnosticError, match="epsilon"):
        convex_weight_perturbation(1, 0, 1.1)
