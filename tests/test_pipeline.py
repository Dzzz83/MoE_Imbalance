import json
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from pipeline import evaluate, train
from pipeline.data import evaluation_transform, make_dataset, save_training_data
from pipeline.model import create_model
from pipeline.runtime import load_checkpoint, resolve_device, seed_everything


def test_baseline_cli_checkpoint_and_evaluation(tmp_path, training_data, monkeypatch):
    data_path = tmp_path / "train.npz"
    save_training_data(data_path, training_data)
    config = {
        "method": "baseline", "train_data": str(data_path), "batch_size": 4,
        "num_workers": 0, "method_options": {"epochs": 1, "learning_rate": 0.01},
    }
    config_path = tmp_path / "baseline.json"
    config_path.write_text(json.dumps(config))
    output = tmp_path / "baseline"
    seed_everything(0)
    initial_weights = create_model().fc.weight.detach().clone()
    trained_models = []

    def capture_model():
        model = create_model()
        trained_models.append(model)
        return model

    def forbid_raw_data(**kwargs):
        pytest.fail("Training must not construct a raw CIFAR dataset")

    monkeypatch.setattr(train, "create_model", capture_model)
    monkeypatch.setattr(evaluate.datasets, "CIFAR100", forbid_raw_data)
    monkeypatch.setattr(sys, "argv", [
        "train", "--config", str(config_path), "--output-dir", str(output), "--device", "cpu",
    ])
    train.main()
    checkpoint_path = output / "final.pt"
    checkpoint_bytes = checkpoint_path.read_bytes()
    restored, checkpoint = load_checkpoint(checkpoint_path, "cpu")
    assert checkpoint["config"]["method_options"]["scheduler"] == "cosine"
    assert checkpoint["config"]["seed"] == 0
    assert json.loads((output / "config.json").read_text()) == checkpoint["config"]
    history = json.loads((output / "history.json").read_text())
    assert len(history) == 1 and np.isfinite(history[0]["loss"])
    assert not torch.equal(initial_weights, restored.fc.weight)
    assert checkpoint["data_settings"] == training_data.data_settings
    for name, value in restored.state_dict().items():
        torch.testing.assert_close(value, trained_models[0].state_dict()[name], rtol=0, atol=0)

    loader = DataLoader(make_dataset(training_data, evaluation_transform()), batch_size=4)
    expected_predictions, expected_labels = evaluate.predict(trained_models[0], loader, "cpu")
    predictions, labels = evaluate.predict(restored, loader, "cpu")
    np.testing.assert_array_equal(predictions, expected_predictions)
    np.testing.assert_array_equal(labels, expected_labels)

    def test_split(**kwargs):
        assert kwargs["train"] is False and kwargs["download"] is False
        return make_dataset(training_data, kwargs["transform"])

    monkeypatch.setattr(evaluate.datasets, "CIFAR100", test_split)
    metrics_path = output / "metrics.json"
    monkeypatch.setattr(sys, "argv", [
        "evaluate", "--checkpoint", str(checkpoint_path), "--data-root", str(tmp_path),
        "--output", str(metrics_path), "--device", "cpu", "--num-workers", "0", "--batch-size", "4",
    ])
    evaluate.main()
    assert json.loads(metrics_path.read_text()) == evaluate.compute_metrics(
        predictions, labels, training_data.class_counts
    )
    assert checkpoint_path.read_bytes() == checkpoint_bytes


def test_equal_training_seeds_reproduce_the_baseline(tmp_path, training_data):
    path = tmp_path / "train.npz"
    save_training_data(path, training_data)
    config = {
        "method": "baseline", "train_data": str(path), "seed": 5,
        "batch_size": 4, "num_workers": 0, "method_options": {"epochs": 1},
    }
    first = train.run_training(config, tmp_path / "first")
    second = train.run_training(config, tmp_path / "second")
    first_model, _ = load_checkpoint(first, "cpu")
    second_model, _ = load_checkpoint(second, "cpu")
    for name, value in first_model.state_dict().items():
        torch.testing.assert_close(value, second_model.state_dict()[name], rtol=0, atol=0)


@pytest.mark.parametrize("change", [
    {"method": "missing"}, {"batch_size": 0}, {"seed": True}, {"num_workers": -1},
    {"typo": 1}, {"method_options": {"epoch": 1}}, {"method_options": {"epochs": 0}},
    {"method_options": {"scheduler": "unknown"}},
])
def test_invalid_configuration(change):
    config = {"method": "baseline", "train_data": "unused.npz"} | change
    with pytest.raises(ValueError):
        train.resolve_config(config)


def test_existing_output_is_not_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        train.run_training({"method": "baseline", "train_data": "unused.npz"}, tmp_path)


def test_unavailable_cuda_is_reported(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="use --device cpu"):
        resolve_device("cuda")


@pytest.mark.parametrize("module", ["pipeline.data", "pipeline.train", "pipeline.evaluate"])
def test_commands_are_runnable(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"], capture_output=True, text=True, check=True,
    )
    assert "usage:" in result.stdout
