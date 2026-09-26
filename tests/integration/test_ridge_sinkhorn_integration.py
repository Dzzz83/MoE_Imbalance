"""CPU-only array integration without any image or test-set access."""

import numpy as np
import json
import hashlib
import pytest

from scripts.ridge_sinkhorn_model import fit_residual_ridge
from scripts.ridge_sinkhorn_outer import hierarchical_paired_intervals, run_outer
from scripts.ridge_sinkhorn_study import ANCHOR, StudyArrays, allocation_stage, evaluate
from scripts.task3f_ridge import compute_sample_weights


def test_oracle_ridge_weights_original_logit_metric_path():
    from scripts.ridge_sinkhorn_oracle import compute_oracle_targets, residuals_to_weights

    rng = np.random.default_rng(29)
    logits = rng.normal(size=(200, 4, 100))
    labels = np.repeat(np.arange(100), 2)
    logits[np.arange(200), :, labels] += 1
    counts = np.concatenate((np.full(35, 100), np.full(35, 30), np.full(30, 10)))
    data = StudyArrays(logits[::2], labels[::2], np.arange(0, 200, 2), logits[1::2], labels[1::2], np.arange(1, 200, 2), counts)
    targets = compute_oracle_targets(data.fit_logits, data.fit_labels, ANCHOR, 1.0).residuals
    fit = fit_residual_ridge(
        data.fit_logits, targets, compute_sample_weights(data.fit_labels, 0),
        feature_set="confidence_only", alpha=1, gamma=0,
    )
    weights = residuals_to_weights(fit.predict_residual(data.selection_logits), ANCHOR, 0.5)
    assert weights.shape == (100, 4)
    assert np.isfinite(weights).all()
    assert np.allclose(weights.sum(axis=1), 1)
    metrics, prediction = evaluate(data.selection_logits, data.selection_labels, weights, counts)
    assert prediction.shape == (100,)
    assert np.isfinite(metrics["balanced_accuracy"])


def test_paired_bootstrap_is_deterministic_and_zero_for_identical_predictions():
    labels = np.arange(100)
    counts = np.concatenate((np.full(35, 100), np.full(35, 30), np.full(30, 10)))
    predictions = {"uniform_logit": labels.copy(), "fixed_006": labels.copy(), "candidate": labels.copy()}
    first = hierarchical_paired_intervals(labels, predictions, counts, seed=4, replicates=20)
    second = hierarchical_paired_intervals(labels, predictions, counts, seed=4, replicates=20)
    assert first == second
    assert first["candidate"]["balanced_accuracy"] == [0.0, 0.0]
    assert first["candidate"]["tail_accuracy"] == [0.0, 0.0]
    assert first["candidate_minus_fixed_006"]["balanced_accuracy"] == [0.0, 0.0]


def test_allocation_fit_and_prices_ignore_selection_labels():
    rng = np.random.default_rng(8)
    fit_logits = rng.normal(size=(100, 4, 100))
    selection_logits = rng.normal(size=(100, 4, 100))
    counts = np.concatenate((np.full(35, 100), np.full(35, 30), np.full(30, 10)))

    def study(labels):
        data = StudyArrays(
            fit_logits, np.arange(100), np.arange(100),
            selection_logits, labels, np.arange(100, 200), counts,
        )
        return allocation_stage(data)

    first = study(np.arange(100))
    second = study(np.roll(np.arange(100), 1))
    assert np.array_equal(first["highlighted_fit_kernel"], second["highlighted_fit_kernel"])
    for identifier in first["weights"]:
        assert np.array_equal(first["weights"][identifier], second["weights"][identifier])


def test_outer_loader_rejects_unlocked_candidate_before_outer_access(tmp_path, monkeypatch):
    root = tmp_path / "study"
    (root / "selection_v1").mkdir(parents=True)
    (root / "locked_outer_v1").mkdir()
    (root / "allocation_v1").mkdir()
    (root / "selection_v1" / "decision.json").write_text(json.dumps({"selected_candidate_id": "selected"}))
    (root / "allocation_v1" / "config.json").write_text(json.dumps({"source_code_sha256": {}}))
    config_hash = hashlib.sha256((root / "allocation_v1" / "config.json").read_bytes()).hexdigest()
    (root / "locked_outer_v1" / "model.json").write_text(json.dumps({
        "candidate_id": "different",
        "development_config_sha256": config_hash,
    }))
    monkeypatch.setattr(
        "scripts.ridge_sinkhorn_outer._load_outer_predictions",
        lambda *args, **kwargs: pytest.fail("outer predictions were opened before lock validation"),
    )
    with pytest.raises(ValueError, match="selection decision disagree"):
        run_outer(data_root=tmp_path, artifact_root=tmp_path, development_directory=root)
