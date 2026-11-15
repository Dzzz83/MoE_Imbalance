import json
from pathlib import Path

import pytest
import torch

from pipeline.methods import baseline
from pipeline import train


@pytest.mark.parametrize("epoch, expected", [
    (1, 0.02), (2, 0.04), (3, 0.06), (4, 0.08), (5, 0.1),
    (6, 0.1), (160, 0.1), (161, 0.001), (180, 0.001),
    (181, 0.00001), (200, 0.00001),
])
def test_ldam_learning_rate_boundaries(epoch, expected):
    options = baseline.prepare_method_options({})
    assert baseline._learning_rate_for_epoch(epoch, options) == pytest.approx(expected)


def test_schedule_can_disable_warmup_and_decay():
    options = baseline.prepare_method_options({"warmup_epochs": 0, "decay_epochs": []})
    assert baseline._learning_rate_for_epoch(1, options) == 0.1
    assert baseline._learning_rate_for_epoch(200, options) == 0.1


@pytest.mark.parametrize("options", [
    {"warmup_epochs": -1}, {"warmup_epochs": True}, {"warmup_epochs": 1.5},
    {"decay_epochs": [180, 160]}, {"decay_epochs": [160, 160]},
    {"decay_epochs": [0]}, {"decay_epochs": [True]}, {"decay_epochs": [1.5]},
    {"decay_epochs": "160,180"}, {"decay_epochs": [4]},
    {"decay_factor": 0}, {"decay_factor": 1}, {"decay_factor": True},
    {"decay_factor": float("nan")}, {"decay_factor": float("inf")},
    {"scheduler": []},
])
def test_invalid_schedule_options(options):
    with pytest.raises(ValueError):
        baseline.prepare_method_options(options)


def test_baseline_config_matches_defaults_and_milestones_are_not_shared():
    config_path = Path(__file__).resolve().parents[1] / "configs" / "baseline.json"
    config = json.loads(config_path.read_text())
    assert config["batch_size"] == 128
    assert config["method_options"] == baseline.prepare_method_options({})
    options = baseline.prepare_method_options(config["method_options"])
    options["decay_epochs"].append(190)
    assert config["method_options"]["decay_epochs"] == [160, 180]
    assert baseline.prepare_method_options({})["decay_epochs"] == [160, 180]


def test_cosine_baseline_recipe_and_checkpoint_can_feed_reweighting(tmp_path, training_data):
    from pipeline.data import save_training_data
    from pipeline.methods.finetuning import _load_pretrained_baseline
    from pipeline.model import create_model

    path = tmp_path / "train.npz"
    save_training_data(path, training_data)
    config = {
        "method": "baseline", "train_data": str(path), "seed": 0,
        "num_workers": 0, "batch_size": 4,
        "method_options": {"scheduler": "cosine", "epochs": 3, "weight_decay": 0.0005},
    }
    checkpoint = train.run_training(config, tmp_path / "cosine")
    history = json.loads((checkpoint.parent / "history.json").read_text())
    assert [row["learning_rate"] for row in history] == pytest.approx([0.1, 0.075, 0.025])
    model = create_model()
    assert _load_pretrained_baseline(model, training_data, checkpoint) == 3
    loaded = torch.load(checkpoint, weights_only=True)
    assert "warmup_epochs" not in loaded["config"]["method_options"]
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, loaded["model_state_dict"][name], rtol=0, atol=0)


def test_cosine_does_not_silently_ignore_multistep_settings():
    with pytest.raises(ValueError, match="Cosine scheduling"):
        baseline.prepare_method_options({"scheduler": "cosine", "warmup_epochs": 5})
