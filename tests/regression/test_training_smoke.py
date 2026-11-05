"""Keep dry-run artifacts isolated from reportable expert runs."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from scripts import train as train_mod
from scripts.base_trainer import CheckpointValidationError, validate_checkpoint_metadata
from scripts.config import TrainingConfig
from scripts.trainers import ConfigDrivenTrainer


@pytest.fixture
def synthetic_training(monkeypatch):
    """Exercise real training and artifact writes without reading CIFAR data."""
    class SyntheticDataModule:
        def __init__(self, config, max_batches=None):
            generator = torch.Generator().manual_seed(12)
            images = torch.randn(4, 4, generator=generator)
            targets = torch.tensor([0, 1, 2, 3])
            size = len(images) if max_batches is None else min(4, max_batches * 2)
            self.dataset = TensorDataset(images[:size], targets[:size])

        def class_counts(self):
            return np.ones(100, dtype=np.int64)

        def describe(self):
            return "synthetic training rows"

        def train_loader(self):
            return DataLoader(self.dataset, batch_size=2)

    monkeypatch.setattr(train_mod, "build_datamodule", SyntheticDataModule)
    monkeypatch.setattr(
        ConfigDrivenTrainer, "build_model", lambda self, config: torch.nn.Linear(4, 100)
    )


def _config(directory):
    return TrainingConfig(expert="ce", seed=78, device="cpu").replace(
        epochs=1, checkpoint_dir=str(directory)
    )


def test_smoke_run_preserves_existing_full_run_artifacts(tmp_path, synthetic_training):
    names = ("CE_seed78_final.pt", "CE_seed78_history.json", "ce_seed78_config.yaml")
    originals = {name: f"existing full run: {name}".encode() for name in names}
    for name, content in originals.items():
        (tmp_path / name).write_bytes(content)

    train_mod.run_training(_config(tmp_path), max_batches=1)

    assert {name: (tmp_path / name).read_bytes() for name in names} == originals


def test_repeated_smoke_runs_keep_separate_complete_outputs(tmp_path, synthetic_training):
    config = _config(tmp_path)
    for _ in range(2):
        history = train_mod.run_training(config, max_batches=1)
        assert [row["epoch"] for row in history] == [1]
        assert np.isfinite(history[0]["train_loss"])
        assert np.isfinite(history[0]["grad_norm"])
        assert history[0]["grad_norm"] > 0

    directories = list((tmp_path / "smoke-runs").glob("ce_seed78_*"))
    assert len(directories) == 2
    assert list(tmp_path.iterdir()) == [tmp_path / "smoke-runs"]
    assert config.checkpoint.dir == str(tmp_path)
    for directory in directories:
        state = torch.load(directory / "CE_seed78_final.pt", weights_only=False)
        assert all(torch.isfinite(value).all() for value in state["model_state_dict"].values())
        history = json.loads((directory / "CE_seed78_history.json").read_text())
        assert [row["epoch"] for row in history] == [1]
        recorded = yaml.safe_load((directory / "ce_seed78_config.yaml").read_text())
        assert Path(recorded["checkpoint"]["dir"]) == directory


def test_smoke_checkpoint_is_not_a_reportable_final_model(tmp_path, synthetic_training):
    train_mod.run_training(_config(tmp_path), max_batches=1)

    [path] = tmp_path.rglob("CE_seed78_final.pt")
    state = torch.load(path, weights_only=False)
    assert state["is_final"] is False
    assert state["is_dry_run"] is True
    assert state["max_batches"] == 1
    with pytest.raises(CheckpointValidationError, match="is_final|dry.run|max_batches"):
        validate_checkpoint_metadata(state, path=path, expected_epoch=1, require_final=True)


def test_full_run_keeps_existing_output_paths(tmp_path, synthetic_training):
    train_mod.run_training(_config(tmp_path))

    state = torch.load(tmp_path / "CE_seed78_final.pt", weights_only=False)
    assert state["is_final"] is True
    assert state.get("is_dry_run", False) is False
    assert state.get("max_batches") is None
    assert (tmp_path / "CE_seed78_history.json").is_file()
    assert (tmp_path / "ce_seed78_config.yaml").is_file()
    assert not (tmp_path / "smoke-runs").exists()


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "1"])
def test_invalid_batch_limit_fails_before_data_or_output_access(tmp_path, monkeypatch, limit):
    def unexpected_data_access(*args, **kwargs):
        pytest.fail("invalid batch limit reached the data module")

    monkeypatch.setattr(train_mod, "build_datamodule", unexpected_data_access)
    with pytest.raises(ValueError, match="max_batches.*positive integer"):
        train_mod.run_training(_config(tmp_path / "checkpoints"), max_batches=limit)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_cli_rejects_invalid_batch_limit_before_loading_config(tmp_path, capsys, limit):
    with pytest.raises(SystemExit) as error:
        train_mod.main([
            "--config", str(tmp_path / "missing.yaml"),
            "--max-batches", limit,
            "--checkpoint-dir", str(tmp_path / "checkpoints"),
        ])
    assert error.value.code == 2
    assert "--max-batches must be a positive integer" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []
