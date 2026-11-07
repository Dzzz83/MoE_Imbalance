import json
from pathlib import Path

import pytest

from pipeline.methods import baseline


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
