import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pipeline import run_ablation
from pipeline.ablation_data import prepare_split, split_positions
from pipeline.ablation_weights import class_balanced_weights, transform_weights, weight_diagnostics
from pipeline.data import TrainingData, load_training_data, make_dataset, save_training_data
from pipeline.methods import prototype_weights
from pipeline.runtime import file_sha256, load_checkpoint


@pytest.fixture
def retained_pool():
    labels = np.concatenate((np.tile(np.arange(100), 5), np.zeros(100), np.ones(20))).astype(np.int64)
    return TrainingData(
        np.random.default_rng(17).integers(0, 256, (len(labels), 32, 32, 3), dtype=np.uint8),
        labels, np.bincount(labels, minlength=100), {"imbalance_factor": 0.01, "data_seed": 42},
    )


def test_split_is_disjoint_complete_reproducible_and_preserves_order(tmp_path, retained_pool):
    fitting, development = split_positions(retained_pool.labels)
    assert not np.intersect1d(fitting, development).size
    np.testing.assert_array_equal(np.sort(np.concatenate((fitting, development))), np.arange(len(retained_pool.labels)))
    assert np.all(np.diff(fitting) > 0) and np.all(np.diff(development) > 0)
    np.testing.assert_array_equal(development, split_positions(retained_pool.labels)[1])
    assert not np.array_equal(development, split_positions(retained_pool.labels, 44)[1])
    assert np.bincount(retained_pool.labels[fitting], minlength=100)[99] == 4
    assert np.bincount(retained_pool.labels[development], minlength=100)[99] == 1
    path = tmp_path / "retained.npz"
    save_training_data(path, retained_pool)
    manifest = prepare_split(path, tmp_path / "split")
    assert prepare_split(path, tmp_path / "split") == manifest
    for name, positions in (("fitting", fitting), ("development", development)):
        subset = load_training_data(tmp_path / "split" / f"{name}.npz")
        np.testing.assert_array_equal(subset.images, retained_pool.images[positions])
        np.testing.assert_array_equal(subset.labels, retained_pool.labels[positions])
    with pytest.raises(ValueError, match="split settings"):
        prepare_split(path, tmp_path / "split", 44)
    (tmp_path / "split/fitting.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="has changed"):
        prepare_split(path, tmp_path / "split")


def test_split_rejects_small_missing_classes_and_invalid_seed(retained_pool):
    with pytest.raises(ValueError, match="at least five"):
        split_positions(np.arange(100, dtype=np.int64))
    with pytest.raises(ValueError, match="integer vector"):
        split_positions(np.array([100], dtype=np.int64))
    with pytest.raises(ValueError, match="split_seed"):
        split_positions(retained_pool.labels, True)


def test_weight_controls_preserve_class_masses_and_shuffle_is_isolated():
    labels = torch.arange(100).repeat(12)
    features = torch.cat((torch.tensor([[1., 0.]]).repeat(800, 1),
                          torch.tensor([[0., 1.]]).repeat(400, 1)))
    ot, prototype = prototype_weights.build_weights(
        features, labels, {"seed": 0, "method_options": prototype_weights.prepare_options({"weighting": "sinkhorn"})},
        torch.device("cpu"),
    )
    # Use explicitly unequal weights; each class has the same three-value pattern.
    ot = torch.tensor([0.5, 1., 1.5]).repeat_interleave(400) / 1200
    rng_state = torch.random.get_rng_state().clone()
    shuffled = transform_weights(ot, labels, "shuffled_ot", 7)
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert not torch.equal(shuffled, ot)
    torch.testing.assert_close(shuffled, transform_weights(ot, labels, "shuffled_ot", 7))
    assert not torch.equal(shuffled, transform_weights(ot, labels, "shuffled_ot", 8))
    balanced = class_balanced_weights(labels)
    half = transform_weights(ot, labels, "half_ot")
    strong = transform_weights(ot, labels, "strong_ot")
    shuffled_strong = transform_weights(ot, labels, "shuffled_strong_ot", 7)
    torch.testing.assert_close(half, (ot + balanced) / 2)
    torch.testing.assert_close(transform_weights(balanced, labels, "half_ot"), balanced)
    torch.testing.assert_close(transform_weights(balanced, labels, "strong_ot"), balanced)
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    for c in range(100):
        members = labels == c
        torch.testing.assert_close(shuffled[members].sort().values, ot[members].sort().values)
        torch.testing.assert_close(shuffled_strong[members].sort().values, strong[members].sort().values)
        torch.testing.assert_close(strong[members][::4] / strong[members][0], torch.tensor([1., 8., 27.]))
        for weights in (balanced, ot, shuffled, half, strong, shuffled_strong):
            assert weights[members].sum().item() == pytest.approx(0.01)
    diagnostics = weight_diagnostics(balanced, labels, [12] * 100, prototype)
    assert all(r["effective_sample_fraction"] == pytest.approx(1) for r in diagnostics["classes"])
    with pytest.raises(ValueError, match="shuffle seed"):
        transform_weights(ot, labels, "shuffled_ot")
    with pytest.raises(ValueError, match="finite positive"):
        transform_weights(torch.zeros_like(ot), labels, "ot")
    with pytest.raises(ValueError, match="mass 1/100"):
        transform_weights(ot * 2, labels, "ot")


def test_prototype_diagnostics_distinguish_active_and_fallback():
    labels = torch.cat((torch.zeros(20, dtype=torch.long), torch.arange(1, 100)))
    features = torch.tensor([[1., 0.]]).repeat(len(labels), 1)
    features[10:20] = torch.tensor([0., 1.])
    weights, diagnostics = prototype_weights.build_weights(
        features, labels, {"seed": 0, "method_options": prototype_weights.prepare_options({"weighting": "sinkhorn"})},
        torch.device("cpu"),
    )
    assert diagnostics[0]["used_ot"]
    assert diagnostics[0]["fallback_reason"] is None
    assert not diagnostics[1]["used_ot"]
    assert diagnostics[1]["fallback_reason"] == "too_few_examples"
    report = weight_diagnostics(weights, labels, [100] + [5] * 99, diagnostics)
    assert report["prototype_ot_coverage"]["head"]["ot_active"] == 1
    assert report["prototype_ot_coverage"]["tail"]["fallback"] == 99
    probe = {}
    assert prototype_weights._cluster_class(torch.ones(12, 2), 5, torch.Generator(), probe) is None
    assert probe["fallback_reason"] == "indistinguishable_features"


def test_feature_evaluation_uses_original_groups_and_preserves_state():
    model = torch.nn.Module()
    model.fc = torch.nn.Linear(100, 100, bias=False)
    with torch.no_grad():
        model.fc.weight.copy_(torch.eye(100))
    model.train()
    labels = torch.cat((torch.arange(100), torch.zeros(5, dtype=torch.long)))
    features = torch.eye(100)[labels]
    features[-5:] = torch.eye(100)[1]
    before = copy.deepcopy(model.state_dict())
    metrics = run_ablation.evaluate_features(model, features, labels, [105, 25] + [5] * 98, "cpu", 32, [0])
    assert metrics["balanced_accuracy"] == pytest.approx((99 + 1 / 6) / 100)
    assert metrics["accuracy"] == pytest.approx(100 / 105)
    assert metrics["head_recall"] == pytest.approx(1 / 6)
    assert metrics["medium_recall"] == metrics["tail_recall"] == 1
    assert metrics["ot_active_recall"] == pytest.approx(1 / 6)
    assert metrics["fallback_recall"] == 1
    assert metrics["development_support"][-1] == 1
    assert "test_support" not in metrics
    assert model.training and model.fc.training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_summary_averages_shuffles_within_seed_before_paired_differences():
    rows = []
    for seed, reference, shuffled in ((0, 0.2, [0.1, 0.3, 0.5]), (1, 0.4, [0.4, 0.4, 0.4])):
        for condition, values in (("class_balanced", [reference]), ("shuffled_ot", shuffled)):
            for value in values:
                rows.append({"seed": seed, "condition": condition, "metrics": {
                    name: value for name in ("balanced_accuracy", "accuracy", "head_recall", "medium_recall", "tail_recall")
                }})
    result = run_ablation.summarize_results(rows)["shuffled_ot"]["balanced_accuracy"]
    assert result["mean"] == pytest.approx(0.35)
    assert result["paired_difference_mean"] == pytest.approx(0.05)
    assert result["paired_difference_sd"] == pytest.approx(np.std([0.1, 0], ddof=1))
    assert result["training_seed_count"] == 2
    assert result["per_seed"][0]["shuffle_sd"] == pytest.approx(0.2)


@pytest.mark.parametrize("overrides", [
    {"seeds": [0, 0]}, {"seeds": [[0]]}, {"seeds": [1]}, {"shuffle_seeds": []},
    {"learning_rates": [float("nan")]}, {"learning_rates": [True]},
    {"classifier": {"learning_rate": 0.1}}, {"split_seed": -1}, {"unknown": 1},
])
def test_study_rejects_invalid_configuration(overrides):
    with pytest.raises(ValueError):
        run_ablation.resolve_study_config(overrides)


def test_cpu_ablation_all_stages_and_completed_run_reuse(tmp_path, retained_pool, monkeypatch):
    source = tmp_path / "retained.npz"
    save_training_data(source, retained_pool)
    config = {
        "train_data": str(source), "seeds": [0], "shuffle_seeds": [0, 1],
        "learning_rates": [0.00002, 0.002], "batch_size": 128, "num_workers": 0,
        "baseline": {"epochs": 1, "warmup_epochs": 0}, "classifier": {"epochs": 1},
    }
    config_before = copy.deepcopy(config)
    config, root, split = run_ablation.prepare_study(config, tmp_path / "study")
    assert config_before["classifier"] == {"epochs": 1}
    device = torch.device("cpu")

    def forbidden_test_access(*args, **kwargs):
        raise AssertionError("Development stages must never load official test data")

    monkeypatch.setattr("pipeline.evaluate.datasets.CIFAR100", forbidden_test_access)
    with pytest.raises(FileNotFoundError):
        run_ablation.run_final(config, root, split, device, tmp_path / "raw")
    progress = []
    selected = run_ablation.run_pilot(config, root, split, device, on_progress=progress.append)
    assert "save fitting baseline seed 0" in progress
    assert f"save pilot rate {selected['learning_rate']}" in progress
    pilot_results = json.loads((root / "pilot_results.json").read_text())
    expected = max(pilot_results, key=lambda r: (r["metrics"]["balanced_accuracy"], -r["learning_rate"]))
    assert selected["learning_rate"] == expected["learning_rate"]
    pilot_path = Path(selected["output_dir"]) / "final.pt"
    pilot_hash = file_sha256(pilot_path)
    rows = run_ablation.run_study(config, root, split, device, on_progress=progress.append)
    assert "save development seed 0 ot" in progress
    assert len(rows) == 7  # Baseline and six continuations with two shuffle replicates.
    assert file_sha256(pilot_path) == pilot_hash
    assert run_ablation.run_study(config, root, split, device) == rows
    baseline_path = root / "baselines/seed0/final.pt"
    baseline_hash = file_sha256(baseline_path)
    baseline_model, baseline_checkpoint = load_checkpoint(baseline_path, "cpu")
    fitting = load_training_data(root / "split/fitting.npz")
    assert baseline_checkpoint["class_counts"].tolist() == fitting.class_counts.tolist()
    assert baseline_checkpoint["class_counts"][0].item() == 84  # Original count was 105.
    for row in rows:
        if row["condition"] == "baseline":
            continue
        path = Path(row["output_dir"])
        model, _ = load_checkpoint(path / "final.pt", "cpu")
        for name, value in baseline_model.backbone.state_dict().items():
            torch.testing.assert_close(model.backbone.state_dict()[name], value, rtol=0, atol=0)
        weights = torch.load(path / "weights.pt", weights_only=True)
        np.testing.assert_array_equal(weights["labels"].numpy(), fitting.labels)
        history = json.loads((path / "history.json").read_text())
        assert history[0]["classifier_parameter_delta_l2"] > 0
        assert history[0]["development"]["development_support"] == split["development_class_counts"]
        assert len(json.loads((path / "weight_diagnostics.json").read_text())["classes"]) == 100
    original_report_hash = file_sha256(root / "study_results.json")

    def forbidden_feature_extraction(*args, **kwargs):
        raise AssertionError("Stronger OT must reuse the original feature caches")

    with monkeypatch.context() as followup:
        followup.setattr(run_ablation, "extract_features", forbidden_feature_extraction)
        followup.setattr(run_ablation, "run_training", forbidden_feature_extraction)
        strength = run_ablation.run_strength_study(root, device)
        assert run_ablation.run_strength_study(root, device) == strength
    assert file_sha256(root / "study_results.json") == original_report_hash
    old_controls = [row for row in rows if row["condition"] in ("class_balanced", "ot")]
    assert strength[:2] == old_controls
    strong, *shuffles = strength[2:]
    assert strong["condition"] == "strong_ot"
    assert [row["shuffle_seed"] for row in shuffles] == [0, 1]
    summary = json.loads((root / "strength/study_summary.json").read_text())
    strong_delta = strong["metrics"]["balanced_accuracy"] - np.mean([
        row["metrics"]["balanced_accuracy"] for row in shuffles
    ])
    assert summary["strong_ot_vs_shuffled_strong_ot"]["balanced_accuracy"]["paired_difference_mean"] == strong_delta
    frozen = run_ablation.freeze_recipe(config, root)
    assert len(frozen["final_conditions"]) == 6
    with pytest.raises(ValueError, match="frozen"):
        run_ablation.run_study(config, root, split, device)

    test_labels = np.arange(100, dtype=np.int64)
    test_data = TrainingData(retained_pool.images[:100], test_labels, np.ones(100, dtype=np.int64), retained_pool.data_settings)
    calls = []

    def artificial_test_data(*, root, train, transform, download):
        assert train is False and download is False
        calls.append(root)
        return make_dataset(test_data, transform)

    monkeypatch.setattr("pipeline.evaluate.datasets.CIFAR100", artificial_test_data)
    final = run_ablation.run_final(config, root, split, device, tmp_path / "raw", on_progress=progress.append)
    assert "save test seed 0 baseline" in progress
    assert len(calls) == len(final) == 7
    assert all(row["metrics"]["accuracy"] == pytest.approx(row["metrics"]["balanced_accuracy"]) for row in final)
    assert run_ablation.run_final(config, root, split, device, tmp_path / "raw") == final
    assert len(calls) == 7  # Completed test evaluations are reused.
    assert file_sha256(baseline_path) == baseline_hash
    (root / "selected_schedule.json").write_text("{}")
    with pytest.raises(ValueError, match="Frozen recipe inputs"):
        run_ablation.run_final(config, root, split, device, tmp_path / "raw")


def test_incomplete_and_changed_runs_are_not_silently_reused(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    with pytest.raises(FileExistsError, match="Incomplete"):
        run_ablation._completed_run(output, {})
    (output / "result.txt").write_text("original")
    run_ablation._finish_run(output, {"seed": 0})
    assert run_ablation._completed_run(output, {"seed": 0})
    with pytest.raises(ValueError, match="inputs changed"):
        run_ablation._completed_run(output, {"seed": 1})
    (output / "result.txt").write_text("changed")
    with pytest.raises(ValueError, match="file changed"):
        run_ablation._completed_run(output, {"seed": 0})


def test_pilot_excludes_nonfinite_runs_reuses_failures_and_stops_on_other_errors(tmp_path, monkeypatch):
    config = run_ablation.resolve_study_config({"learning_rates": [0.00002, 0.02]})
    monkeypatch.setattr(run_ablation, "_baseline", lambda *args: tmp_path / "baseline.pt")
    monkeypatch.setattr(run_ablation, "_seed_inputs", lambda *args: {"cache_sha256": "unchanged"})
    calls = []

    def numerical_failure(config, fitting, source, bundle, counts, seed, condition, rate, output, device):
        calls.append(rate)
        if rate == 0.02:
            raise RuntimeError("Nonfinite weighted classifier loss")
        return {"balanced_accuracy": 0.4}

    monkeypatch.setattr(run_ablation, "_continuation", numerical_failure)
    selected = run_ablation.run_pilot(config, tmp_path, {"original_class_counts": [5] * 100}, "cpu")
    assert selected["learning_rate"] == 0.00002
    assert json.loads((tmp_path / "pilot_results.json").read_text())[1]["status"] == "failed"
    run_ablation.run_pilot(config, tmp_path, {"original_class_counts": [5] * 100}, "cpu")
    assert calls.count(0.02) == 1

    def infrastructure_failure(*args):
        raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(run_ablation, "_continuation", infrastructure_failure)
    with pytest.raises(RuntimeError, match="out of memory"):
        run_ablation.run_pilot(config, tmp_path, {"original_class_counts": [5] * 100}, "cpu")


def test_pilot_keeps_close_learning_rates_in_separate_folders(tmp_path, monkeypatch):
    rates = [0.00002000002, 0.00002000001]
    config = run_ablation.resolve_study_config({"learning_rates": rates})
    monkeypatch.setattr(run_ablation, "_baseline", lambda *args: tmp_path / "baseline.pt")
    monkeypatch.setattr(run_ablation, "_seed_inputs", lambda *args: {"cache_sha256": "unchanged"})

    def equal_scores(config, fitting, source, bundle, counts, seed, condition, rate, output, device):
        output.mkdir(parents=True)  # A collision must fail rather than merge two runs.
        return {"balanced_accuracy": 0.4}

    monkeypatch.setattr(run_ablation, "_continuation", equal_scores)
    selected = run_ablation.run_pilot(config, tmp_path, {"original_class_counts": [5] * 100}, "cpu")
    results = json.loads((tmp_path / "pilot_results.json").read_text())
    assert len({row["output_dir"] for row in results}) == 2
    assert selected["learning_rate"] == min(rates)
