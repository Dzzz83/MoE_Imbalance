"""Focused synthetic checks for fitted residuals and frozen selection."""

import numpy as np
import pytest

from scripts.ridge_sinkhorn_model import fit_residual_ridge
from scripts.ridge_sinkhorn_selection import select_candidate


def test_residual_fit_predicts_from_logits_without_labels():
    rng = np.random.default_rng(7)
    logits = rng.normal(size=(30, 4, 5))
    targets = rng.normal(size=(30, 4))
    fit = fit_residual_ridge(
        logits, targets, np.ones(30), feature_set="confidence_only", alpha=1, gamma=0
    )
    result = fit.predict_residual(logits[:3])
    assert result.shape == (3, 4)
    assert np.isfinite(result).all()
    with pytest.raises(ValueError):
        fit_residual_ridge(logits, targets, np.ones(29), feature_set="confidence_only", alpha=1, gamma=0)


def test_selection_uses_joint_improvement_and_rejects_reference_ties():
    references = {
        name: {"balanced_accuracy": ba, "tail_accuracy": tail}
        for name, ba, tail in (
            ("uniform_logit", .30, .10), ("uniform_probability", .31, .09),
            ("uniform_without_ce", .31, .10), ("fixed_006", .35, .10),
            ("fixed_007", .32, .14), ("fixed_010", .33, .11),
            ("fixed_011", .31, .12),
        )
    }
    candidates = [
        {"candidate_id": "tie", "metrics": {"balanced_accuracy": .35, "tail_accuracy": .10}},
        {"candidate_id": "new", "metrics": {"balanced_accuracy": .36, "tail_accuracy": .13}},
        {"candidate_id": "no_tail", "metrics": {"balanced_accuracy": .37, "tail_accuracy": .10}},
    ]
    result = select_candidate(candidates, references)
    assert result["selected_candidate_id"] == "new"
    assert result["candidate_decisions"][0]["dominating_references"] == ["fixed_006"]
    assert not result["candidate_decisions"][2]["joint_uniform_improvement"]
