from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from pipeline import evaluate, train
from pipeline.data import make_dataset, save_training_data
from pipeline.methods import baseline, finetuning, reweighting
from pipeline.model import create_model
from pipeline.runtime import load_checkpoint, save_checkpoint


def test_comparison_configs_share_checkpoint_classifier_and_training_settings():
    root = Path(__file__).resolve().parents[1]
    configs = [
        train.resolve_config(json.loads((root / "configs" / f"{name}.json").read_text()))
        for name in ("uniform_finetuning", "ot_finetuning")
    ]
    uniform, ot = configs
    for name in ("seed", "batch_size", "num_workers", "train_data"):
        assert uniform[name] == ot[name]
    for name in ("pretrained_checkpoint", "classifier"):
        assert uniform["method_options"][name] == ot["method_options"][name]
    assert "pretraining" not in ot["method_options"]
    assert ot["method_options"]["score_learning_rate"] == 0.001
    original = deepcopy(ot)
    resolved = train.resolve_config(ot)
    resolved["method_options"]["classifier"]["epochs"] = 1
    assert ot == original


@pytest.mark.parametrize("method", ["uniform_finetuning", "ot_finetuning"])
@pytest.mark.parametrize("change", [
    {"pretrained_checkpoint": None}, {"pretrained_checkpoint": ""},
    {"pretrained_checkpoint": "   "}, {"pretrained_checkpoint": True},
    {"pretraining": {"epochs": 1}}, {"classifier": {"epochs": 0}},
])
def test_finetuning_rejects_invalid_options(method, change):
    config = {
        "method": method, "train_data": "unused.npz",
        "method_options": {"pretrained_checkpoint": "baseline.pt"} | change,
    }
    with pytest.raises(ValueError):
        train.resolve_config(config)


def test_uniform_finetuning_rejects_ot_settings():
    with pytest.raises(ValueError, match="Unknown fine-tuning options: score_updates"):
        finetuning.prepare_uniform_options({"pretrained_checkpoint": "baseline.pt", "score_updates": 1})


@pytest.mark.parametrize("invalid_field", ["config", "class_counts", "data_settings", "model_state_dict"])
def test_incompatible_pretraining_checkpoint_is_rejected(tmp_path, training_data, invalid_field):
    path = tmp_path / "baseline.pt"
    model = create_model()
    config = train.resolve_config({"method": "baseline", "train_data": "unused.npz"})
    save_checkpoint(path, model, config, training_data)
    checkpoint = torch.load(path, weights_only=True)
    if invalid_field == "config":
        checkpoint["config"]["method"] = "reweighting"
    elif invalid_field == "class_counts":
        checkpoint["class_counts"][0] += 1
    elif invalid_field == "data_settings":
        checkpoint["data_settings"]["data_seed"] += 1
    else:
        del checkpoint[invalid_field]
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        finetuning._load_pretrained_baseline(model, training_data, path)


@pytest.mark.parametrize("force_uniform_ot", [False, True])
def test_matched_finetuning_from_baseline_checkpoint(
    tmp_path, all_class_training_data, monkeypatch, force_uniform_ot,
):
    data_path = tmp_path / "train.npz"
    save_training_data(data_path, all_class_training_data)
    baseline_config = {
        "method": "baseline", "train_data": str(data_path), "seed": 7,
        "batch_size": 32, "num_workers": 0, "method_options": {"epochs": 1},
    }
    source_path = train.run_training(baseline_config, tmp_path / "baseline")
    source_bytes = source_path.read_bytes()
    pretrained, _ = load_checkpoint(source_path, "cpu")
    initial_states, epoch_batches, epoch_weights = [], [], []
    score_update_count = 0
    cache_features = reweighting._cache_training_features
    classifier_epoch = reweighting._train_classifier_epoch
    update_scores = reweighting._update_example_scores

    def forbid_pretraining(*args):
        pytest.fail("Fine-tuning must load the baseline without repeating pretraining")

    def capture_initial_state(model, *args):
        initial_states.append(deepcopy(model.state_dict()))
        return cache_features(model, *args)

    def capture_classifier_epoch(model, loader, optimizer, weights, device):
        batches = list(loader)
        epoch_batches.append(torch.cat([positions for _, _, positions in batches]))
        epoch_weights.append(weights.clone())
        return classifier_epoch(model, batches, optimizer, weights, device)

    def capture_score_update(*args, **kwargs):
        nonlocal score_update_count
        score_update_count += 1
        weights, diagnostics = update_scores(*args, **kwargs)
        if force_uniform_ot:
            weights = torch.full_like(weights, 1 / len(weights))
        return weights, diagnostics

    monkeypatch.setattr(baseline, "train", forbid_pretraining)
    monkeypatch.setattr(reweighting, "_pretrain", forbid_pretraining)
    monkeypatch.setattr(reweighting, "_cache_training_features", capture_initial_state)
    monkeypatch.setattr(reweighting, "_train_classifier_epoch", capture_classifier_epoch)
    monkeypatch.setattr(reweighting, "_update_example_scores", capture_score_update)
    paths = []
    for method in ("uniform_finetuning", "ot_finetuning"):
        config = baseline_config | {
            "method": method,
            "method_options": {
                "pretrained_checkpoint": str(source_path),
                "classifier": {"epochs": 2, "learning_rate": 0.02},
            },
        }
        paths.append(train.run_training(config, tmp_path / method))
        if method == "uniform_finetuning":
            assert score_update_count == 0
        history = json.loads((tmp_path / method / "history.json").read_text())
        assert [record["epoch"] for record in history] == [2, 3]
        assert [record["stage_epoch"] for record in history] == [1, 2]
        assert [record["stage"] for record in history] == [
            "reweighting" if method == "ot_finetuning" else method
        ] * 2
        if method == "uniform_finetuning":
            assert all("ot_cost_before_update" not in record for record in history)
            assert not (tmp_path / method / "reweighting_diagnostics.json").exists()
            diagnostics = json.loads((tmp_path / method / "uniform_finetuning_diagnostics.json").read_text())
            torch.testing.assert_close(
                torch.tensor(diagnostics[-1]["class_weight_masses"]),
                torch.tensor(all_class_training_data.class_counts / len(all_class_training_data.labels)),
                check_dtype=False,
            )
    assert score_update_count == 2
    assert source_path.read_bytes() == source_bytes
    for state in initial_states:
        for name, value in pretrained.state_dict().items():
            torch.testing.assert_close(state[name], value, rtol=0, atol=0)
    for first, second in zip(epoch_batches[:2], epoch_batches[2:], strict=True):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        assert sorted(first.tolist()) == list(range(len(all_class_training_data.labels)))
    for weights in epoch_weights[:2]:
        torch.testing.assert_close(weights, torch.full_like(weights, 1 / len(weights)), rtol=0, atol=0)
    if not force_uniform_ot:
        assert not torch.equal(epoch_weights[0], epoch_weights[2])
    models = [load_checkpoint(path, "cpu")[0] for path in paths]
    for model in models:
        for name, value in pretrained.backbone.state_dict().items():
            torch.testing.assert_close(model.backbone.state_dict()[name], value, rtol=0, atol=0)
        assert not torch.equal(model.fc.weight, pretrained.fc.weight)
    if force_uniform_ot:
        for name, value in models[0].state_dict().items():
            torch.testing.assert_close(models[1].state_dict()[name], value, rtol=0, atol=0)

    def artificial_test_split(**kwargs):
        assert kwargs["train"] is False and kwargs["download"] is False
        return make_dataset(all_class_training_data, kwargs["transform"])

    monkeypatch.setattr(evaluate.datasets, "CIFAR100", artificial_test_split)
    data_path.unlink()
    for path in paths:
        metrics = evaluate.evaluate_checkpoint(
            path, tmp_path, path.parent / "metrics.json", num_workers=0, batch_size=32,
        )
        assert sum(metrics["test_support"]) == len(all_class_training_data.labels)
