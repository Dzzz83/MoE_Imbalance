import json
from pathlib import Path

import pytest
import torch

from pipeline import train
from pipeline.data import save_training_data
from pipeline.methods import prototype_weights
from pipeline.model import create_model
from pipeline.runtime import load_checkpoint, save_checkpoint


def test_prototype_weights_balance_views_and_fall_back_for_small_classes():
    # Interleave the other classes so weight positions cannot be confused with cluster order.
    labels = torch.cat((
        torch.zeros(80, dtype=torch.long), torch.arange(1, 100), torch.zeros(20, dtype=torch.long),
    ))
    features = torch.tensor([[1., 0.]]).repeat(len(labels), 1)
    features[-20:] = torch.tensor([0., 1.])
    expected_view_masses = {
        "class_balanced": [0.008, 0.002],
        "cluster_balanced": [0.005, 0.005],
        "sinkhorn": [0.005, 0.005],
    }
    for mode, expected in expected_view_masses.items():
        options = prototype_weights.prepare_options({"weighting": mode})
        weights, diagnostics = prototype_weights.build_weights(
            features, labels, {"method_options": options, "seed": 7}, torch.device("cpu")
        )
        torch.testing.assert_close(weights[80:-20], torch.full((99,), 0.01))
        view_masses = [weights[:80].sum().item(), weights[-20:].sum().item()]
        assert sum(view_masses) == pytest.approx(0.01)
        # Soft transport fixes the prototype quotas and allows some cross-view mass.
        tolerance = 0.005 if mode == "sinkhorn" else 1e-6
        assert view_masses == pytest.approx(expected, rel=tolerance)
        assert diagnostics[1]["cluster_sizes"] is None
        if mode != "class_balanced":
            assert sorted(diagnostics[0]["cluster_sizes"]) == [20, 80]


def test_clustering_rejects_a_view_with_too_few_examples():
    features = torch.tensor([[1., 0.]] * 11 + [[0., 1.]] * 4)
    assert prototype_weights._cluster_class(features, 5, torch.Generator().manual_seed(0)) is None


def test_sinkhorn_changes_row_weights_and_rejects_unconverged_plan():
    cost = torch.tensor([[0., 1.], [1., 0.], [1., 1.]])
    options = {"epsilon": 0.1, "row_kl_strength": 0.2, "sinkhorn_tolerance": 0.001}
    plan, diagnostics = prototype_weights._relaxed_sinkhorn(cost, sinkhorn_iterations=200, **options)
    torch.testing.assert_close(plan.sum(dim=0), torch.full((2,), 0.5, dtype=plan.dtype))
    assert plan[2].sum() < plan[0].sum()
    assert diagnostics["sinkhorn_residual"] <= options["sinkhorn_tolerance"]
    with pytest.raises(RuntimeError, match="did not converge"):
        prototype_weights._relaxed_sinkhorn(cost, sinkhorn_iterations=1, **options)


@pytest.mark.parametrize("mode", ["class_balanced", "cluster_balanced", "sinkhorn", "sava"])
def test_prototype_finetuning_config_and_checkpoint(tmp_path, all_class_training_data, mode):
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    uniform = train.resolve_config(json.loads((config_dir / "uniform_finetuning.json").read_text()))
    config = train.resolve_config(json.loads((config_dir / f"{mode}_finetuning.json").read_text()))
    for key in ("seed", "batch_size", "num_workers", "train_data"):
        assert config[key] == uniform[key]
    options = config["method_options"]
    for key in ("classifier", "pretrained_checkpoint"):
        assert options[key] == uniform["method_options"][key]
    if mode != "sava":
        assert options["weighting"] == mode
    else:
        options["sava"]["reference_per_class"] = 1
        options["sava"]["scoring_batch_size"] = 64

    data_path = tmp_path / "train.npz"
    save_training_data(data_path, all_class_training_data)
    model = create_model()
    baseline = train.resolve_config({"method": "baseline", "train_data": str(data_path)})
    source = tmp_path / "baseline.pt"
    save_checkpoint(source, model, baseline, all_class_training_data)
    options["pretrained_checkpoint"] = str(source)
    options["classifier"]["epochs"] = 1
    config.update(train_data=str(data_path), num_workers=0, batch_size=32)
    output_dir = tmp_path / mode
    final_path = train.run_training(config, output_dir)

    final_model, _ = load_checkpoint(final_path, "cpu")
    for name, value in model.backbone.state_dict().items():
        torch.testing.assert_close(value, final_model.backbone.state_dict()[name], rtol=0, atol=0)
    history = json.loads((output_dir / "history.json").read_text())
    assert history[0]["classifier_parameter_delta_l2"] > 0
    diagnostic_file = "sava_finetuning_diagnostics.json" if mode == "sava" else "prototype_finetuning_diagnostics.json"
    diagnostics = json.loads((output_dir / diagnostic_file).read_text())
    assert diagnostics[0]["class_weight_masses"] == pytest.approx([0.01] * 100)
