import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from pipeline import evaluate
from pipeline import train_classifier as trainer
from pipeline.methods.synthesis import resolve_options
from pipeline.model import create_model
from pipeline.runtime import load_checkpoint


@pytest.fixture
def inputs(tmp_path):
    torch.manual_seed(4)
    model = create_model()
    labels = torch.arange(100, dtype=torch.int64)
    counts = torch.ones(100, dtype=torch.int64)
    settings = {"imbalance_factor": 0.01, "data_seed": 42}
    checkpoint = tmp_path / "pretrained.pt"
    torch.save({
        "model_state_dict": model.state_dict(), "class_counts": counts,
        "data_settings": settings, "config": {"seed": 0},
    }, checkpoint)
    features = torch.randn(100, 64)
    cache = {
        "features": features, "labels": labels, "class_counts": counts,
        "data_settings": settings,
        "feature_scale": features.double().square().sum(1).mean().sqrt().item(),
        "checkpoint_sha256": trainer.file_sha256(checkpoint),
        "train_data_sha256": "artificial-test-data",
    }
    cache_path = tmp_path / "real.pt"
    torch.save(cache, cache_path)
    synthetic = {
        "features": features + 0.01, "labels": labels.clone(),
        "real_class_counts": counts.clone(), "synthetic_class_counts": counts.clone(),
        "data_settings": settings, "feature_scale": cache["feature_scale"],
        "checkpoint_sha256": cache["checkpoint_sha256"],
        "generation_config": {
            "feature_cache_sha256": trainer.file_sha256(cache_path),
            "seed": 0, "method_options": resolve_options({"target_count": 2}),
        },
    }
    synthetic_path = tmp_path / "synthetic.pt"
    torch.save(synthetic, synthetic_path)
    config = {
        "pretrained_checkpoint": str(checkpoint), "feature_cache": str(cache_path),
        "synthetic_features": str(synthetic_path), "output_dir": str(tmp_path / "trained"),
        "mode": "synthetic", "seed": 0, "target_count": 2,
        "epochs": 2, "batch_size": 50, "num_workers": 0,
    }
    return config, cache, synthetic


def test_raw_coordinates_and_duplicate_counts(inputs):
    config, cache, synthetic = inputs
    resolved = trainer.resolve_config(config)
    actual, labels = trainer.prepare_training_features(
        cache, resolved, trainer.file_sha256(config["feature_cache"]), synthetic
    )
    # Neither real nor synthetic features should be rescaled during Stage 4.
    torch.testing.assert_close(actual[:100], cache["features"], rtol=0, atol=0)
    torch.testing.assert_close(actual[100:], synthetic["features"], rtol=0, atol=0)
    assert torch.equal(torch.bincount(labels, minlength=100), torch.full((100,), 2))
    duplicate = resolved | {"mode": "duplicate", "synthetic_features": None}
    first, first_labels = trainer.prepare_training_features(cache, duplicate, "unused")
    second, second_labels = trainer.prepare_training_features(cache, duplicate, "unused")
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert torch.equal(first_labels, second_labels)
    assert torch.equal(torch.bincount(first_labels, minlength=100), torch.full((100,), 2))
    for class_id in range(100):
        expected = cache["features"][class_id].expand(2, -1)
        torch.testing.assert_close(first[first_labels == class_id], expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode,expected", [("real", 100), ("duplicate", 200), ("synthetic", 200)])
def test_training_changes_only_head_and_preserves_real_counts(inputs, mode, expected):
    config, cache, _ = inputs
    config = config | {"mode": mode}
    if mode != "synthetic":
        config["synthetic_features"] = None
    before = torch.load(config["pretrained_checkpoint"], weights_only=True)
    real_bytes = Path(config["feature_cache"]).read_bytes()
    checkpoint_path = trainer.run_training(config)
    model, after = load_checkpoint(checkpoint_path, "cpu")
    assert torch.equal(after["class_counts"], cache["class_counts"])
    assert after["training_summary"]["examples_per_epoch"] == expected
    assert after["training_summary"]["total_examples_processed"] == expected * 2
    assert after["training_summary"]["backbone_unchanged"]
    assert Path(config["feature_cache"]).read_bytes() == real_bytes
    assert not torch.equal(after["model_state_dict"]["fc.weight"], before["model_state_dict"]["fc.weight"])
    for name, value in before["model_state_dict"].items():
        if name.startswith("backbone."):
            torch.testing.assert_close(after["model_state_dict"][name], value, rtol=0, atol=0)
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 100)


def test_resume_matches_uninterrupted_training(inputs, monkeypatch, tmp_path):
    config, _, _ = inputs
    complete = trainer.run_training(config | {"output_dir": str(tmp_path / "complete")})
    split_config = config | {"output_dir": str(tmp_path / "interrupted")}
    original_save = trainer.atomic_torch_save

    def interrupt_after_checkpoint(value, path):
        original_save(value, path)
        if Path(path).name == "last.pt" and value["epoch"] == 1:
            raise RuntimeError("simulated interruption")

    with monkeypatch.context() as patch:
        patch.setattr(trainer, "atomic_torch_save", interrupt_after_checkpoint)
        with pytest.raises(RuntimeError, match="simulated"):
            trainer.run_training(split_config)
    resumed = trainer.run_training(split_config, resume=True)
    left = torch.load(complete, weights_only=True)
    right = torch.load(resumed, weights_only=True)
    for name in left["model_state_dict"]:
        torch.testing.assert_close(left["model_state_dict"][name], right["model_state_dict"][name],
                                   rtol=0, atol=0)
    first_history = json.loads((complete.parent / "history.json").read_text())
    second_history = json.loads((resumed.parent / "history.json").read_text())
    assert first_history == second_history


def test_shared_evaluator_needs_no_feature_or_synthesis_files(inputs, monkeypatch, tmp_path):
    config, _, _ = inputs
    checkpoint = trainer.run_training(config)
    Path(config["synthetic_features"]).unlink()
    Path(config["feature_cache"]).unlink()

    def fake_test_data(**kwargs):
        assert kwargs["train"] is False
        return TensorDataset(torch.randn(100, 3, 32, 32), torch.arange(100))

    monkeypatch.setattr(evaluate.datasets, "CIFAR100", fake_test_data)
    metrics = evaluate.evaluate_checkpoint(
        checkpoint, tmp_path, tmp_path / "metrics.json", batch_size=20, num_workers=0
    )
    assert 0 <= metrics["accuracy"] <= 1
    assert len(metrics["per_class_recall"]) == 100


def test_wrong_checkpoint_cache_and_incomplete_synthesis_are_rejected(inputs, tmp_path):
    config, cache, synthetic = inputs
    bad_path = tmp_path / "bad.pt"
    torch.save(cache | {"checkpoint_sha256": "wrong"}, bad_path)
    with pytest.raises(ValueError, match="different pretrained checkpoint"):
        trainer.run_training(config | {"feature_cache": str(bad_path)})
    incomplete = synthetic | {"features": synthetic["features"][:5], "labels": synthetic["labels"][:5]}
    with pytest.raises(ValueError, match="all classes"):
        trainer.prepare_training_features(
            cache, trainer.resolve_config(config), trainer.file_sha256(config["feature_cache"]), incomplete
        )


def test_output_and_resume_protect_inputs(inputs):
    config, _, _ = inputs
    trainer.run_training(config)
    with pytest.raises(FileExistsError):
        trainer.run_training(config)
    with pytest.raises(ValueError, match="identical"):
        trainer.run_training(config | {"learning_rate": 0.02}, resume=True)


@pytest.mark.parametrize("change", [
    {"epochs": 0}, {"batch_size": 0}, {"mode": "unknown"},
    {"seed": True}, {"momentum": 1}, {"learning_rate": float("nan")},
])
def test_invalid_settings(inputs, change):
    config, _, _ = inputs
    with pytest.raises(ValueError):
        trainer.resolve_config(config | change)
