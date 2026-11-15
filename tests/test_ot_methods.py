"""Exercise both OT paths from the same shared baseline and retained pool."""

import json
from pathlib import Path

import pytest
import torch

from pipeline import train, train_classifier
from pipeline.data import save_training_data
from pipeline.extract_features import run_extraction
from pipeline.methods.synthesis import run_generation


@pytest.mark.parametrize("scheduler", ["multistep", "cosine"])
def test_reweighting_and_synthesis_share_baseline_and_raw_coordinates(
    tmp_path, all_class_training_data, scheduler,
):
    prepared = tmp_path / "train.npz"
    save_training_data(prepared, all_class_training_data)
    shared = {"train_data": str(prepared), "seed": 2, "batch_size": 32, "num_workers": 0}
    source = train.run_training(shared | {
        "method": "baseline", "method_options": {"epochs": 1, "scheduler": scheduler},
    }, tmp_path / "baseline")
    before = torch.load(source, weights_only=True)
    source_bytes = source.read_bytes()
    cache_path = tmp_path / "features.pt"
    cache = run_extraction(source, prepared, cache_path, batch_size=32)
    generation_dir = tmp_path / "generated"
    generation_config = {
        "feature_cache": str(cache_path), "output_dir": str(generation_dir), "seed": 2,
        "method_options": {"target_count": 2, "guide_steps": 2, "max_iterations": 2},
    }
    synthetic = run_generation(generation_config)
    classifier_config = {
        "pretrained_checkpoint": str(source), "feature_cache": str(cache_path),
        "synthetic_features": str(generation_dir / "synthetic_features.pt"),
        "output_dir": str(tmp_path / "synthetic_classifier"), "seed": 2,
        "target_count": 2, "epochs": 1, "batch_size": 32,
    }
    synthesis_checkpoint = train_classifier.run_training(classifier_config)
    # Kaggle reruns generation before the classifier's completed-run check.
    run_generation(generation_config, resume=True)
    assert train_classifier.run_training(classifier_config, resume=True) == synthesis_checkpoint
    weighted_checkpoint = train.run_training(shared | {
        "method": "prototype_finetuning", "method_options": {
            "pretrained_checkpoint": str(source), "weighting": "class_balanced",
            "classifier": {"epochs": 1},
        },
    }, tmp_path / "weighted_classifier")
    # With one positive example, a barycenter is that exact raw-coordinate row.
    single_class = 1
    expected = cache["features"][cache["labels"] == single_class]
    actual = synthetic["features"][synthetic["labels"] == single_class]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for path in (synthesis_checkpoint, weighted_checkpoint):
        after = torch.load(path, weights_only=True)
        assert torch.equal(after["class_counts"], before["class_counts"])
        assert after["data_settings"] == before["data_settings"]
        assert not torch.equal(after["model_state_dict"]["fc.weight"], before["model_state_dict"]["fc.weight"])
        for name, value in before["model_state_dict"].items():
            if name.startswith("backbone."):
                torch.testing.assert_close(after["model_state_dict"][name], value, rtol=0, atol=0)
    assert source.read_bytes() == source_bytes


def test_matched_seed_configs_share_checkpoints_and_generation_paths():
    root = Path(__file__).resolve().parents[1] / "configs"
    for seed in (0, 1, 2):
        baseline = json.loads((root / f"baseline_matched_seed{seed}.json").read_text())
        generation = json.loads((root / f"synthesis_matched_seed{seed}.json").read_text())
        classifier = train_classifier.resolve_config(json.loads(
            (root / f"classifier_synthesis_matched_seed{seed}.json").read_text()
        ))
        assert baseline["seed"] == generation["seed"] == classifier["seed"] == seed
        assert train.resolve_config(baseline)["method_options"]["scheduler"] == "multistep"
        assert classifier["pretrained_checkpoint"] == f"outputs/baseline_seed{seed}/final.pt"
        assert generation["feature_cache"] == classifier["feature_cache"]
        assert classifier["synthetic_features"] == f"{generation['output_dir']}/synthetic_features.pt"
        assert generation["method_options"]["target_count"] == classifier["target_count"]
