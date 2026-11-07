from copy import deepcopy
import json
from pathlib import Path

import pytest
import torch

from pipeline.data import evaluation_transform, make_dataset
from pipeline.methods import baseline, reweighting
from pipeline.model import create_model
from pipeline.runtime import seed_everything


def test_pretraining_matches_baseline_without_changing_config(tmp_path, training_data):
    config = {
        "method": "reweighting",
        "seed": 7,
        "batch_size": 4,
        "num_workers": 0,
        "method_options": {"pretraining": {"epochs": 1}},
    }
    original_config = deepcopy(config)
    seed_everything(config["seed"])
    model = create_model()
    initial_backbone_weights = model.backbone.conv1.weight.detach().clone()
    initial_classifier_weights = model.fc.weight.detach().clone()
    history = reweighting._pretrain(
        model, training_data, config, tmp_path, torch.device("cpu")
    )

    seed_everything(config["seed"])
    baseline_model = create_model()
    baseline_config = config | {
        "method_options": baseline.prepare_method_options({"epochs": 1})
    }
    baseline_history = baseline.train(
        baseline_model, training_data, baseline_config, tmp_path, torch.device("cpu")
    )

    assert config == original_config
    assert history == [record | {"stage": "pretraining"} for record in baseline_history]
    assert "stage" not in baseline_history[0]
    assert history[0]["epoch"] == 1
    assert not torch.equal(model.backbone.conv1.weight, initial_backbone_weights)
    assert not torch.equal(model.fc.weight, initial_classifier_weights)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, baseline_model.state_dict()[name], rtol=0, atol=0)


def test_frozen_backbone_preserves_parameters_and_buffers_during_classifier_updates():
    seed_everything(3)
    model = create_model()
    images = torch.randn(3, 3, 32, 32)
    labels = torch.tensor([0, 1, 99])
    # Pretraining leaves gradients and updated batch-norm buffers on the model.
    torch.nn.functional.cross_entropy(model(images), labels).backward()
    backbone_state = deepcopy(model.backbone.state_dict())
    classifier_weights = model.fc.weight.detach().clone()

    reweighting._freeze_backbone(model)
    assert all(not module.training for module in model.backbone.modules())
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.fc.parameters())

    optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.01)
    model.fc.train()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()

    for name, value in model.backbone.state_dict().items():
        torch.testing.assert_close(value, backbone_state[name], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.backbone.parameters())
    assert not torch.equal(model.fc.weight, classifier_weights)


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_feature_cache_preserves_retained_order_and_backbone(training_data, batch_size):
    seed_everything(4)
    model = create_model()
    backbone_state = deepcopy(model.backbone.state_dict())
    config = {"batch_size": batch_size, "num_workers": 0, "seed": 4}
    features, labels = reweighting._cache_training_features(
        model, training_data, config, torch.device("cpu")
    )
    assert features.shape == (len(training_data.labels), 64)
    assert features.device.type == labels.device.type == "cpu"
    assert not features.requires_grad
    assert not features.is_inference()
    assert labels.dtype == torch.int64
    torch.testing.assert_close(labels, torch.from_numpy(training_data.labels), rtol=0, atol=0)

    dataset = make_dataset(training_data, evaluation_transform())
    images = torch.stack([dataset[index][0] for index in range(len(dataset))])
    with torch.no_grad():
        expected_features = model.backbone(images)
    torch.testing.assert_close(features, expected_features, rtol=1e-5, atol=1e-6)

    repeated_features, repeated_labels = reweighting._cache_training_features(
        model, training_data, config, torch.device("cpu")
    )
    torch.testing.assert_close(features, repeated_features, rtol=0, atol=0)
    torch.testing.assert_close(labels, repeated_labels, rtol=0, atol=0)
    for name, value in model.backbone.state_dict().items():
        torch.testing.assert_close(value, backbone_state[name], rtol=0, atol=0)
    assert all(not module.training for module in model.backbone.modules())
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())

    # Cached raw features must support backward through the existing classifier.
    torch.nn.functional.cross_entropy(model.fc(features), labels).backward()
    assert model.fc.weight.grad is not None
    assert torch.isfinite(model.fc.weight.grad).all()
    assert all(parameter.grad is None for parameter in model.backbone.parameters())


def test_feature_cache_rejects_nonfinite_features(training_data, monkeypatch):
    model = create_model()

    def nonfinite_features(images):
        return images.new_full((len(images), 64), float("nan"))

    monkeypatch.setattr(model.backbone, "forward", nonfinite_features)
    config = {"batch_size": 4, "num_workers": 0, "seed": 0}
    with pytest.raises(RuntimeError, match="Nonfinite backbone features"):
        reweighting._cache_training_features(model, training_data, config, torch.device("cpu"))


def test_ot_references_average_raw_features_in_class_order_without_mutating_cache():
    # Class 0 has unequal-length examples; averaging normalized examples instead
    # would incorrectly give a representative along the diagonal.
    labels = torch.cat([torch.tensor([0]), torch.arange(100)])
    raw_features = torch.tensor([[0.0, 2.0]] + [[6.0, 0.0]] * 100, dtype=torch.float64)
    raw_features[2] = torch.tensor([0.0, 3.0])  # Class 1, distinct from class 0.
    permutation = torch.arange(len(labels) - 1, -1, -1)
    labels = labels[permutation]
    raw_features = raw_features[permutation]
    original_features = raw_features.clone()
    original_labels = labels.clone()

    features, references, masses = reweighting._build_ot_references(raw_features, labels)

    assert references.shape == (100, 2)
    expected_class_zero = torch.tensor([3.0, 1.0], dtype=torch.float64) / (10 ** 0.5)
    torch.testing.assert_close(references[0], expected_class_zero)
    torch.testing.assert_close(references[1], torch.tensor([0.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(references[2:], torch.tensor([[1.0, 0.0]] * 98, dtype=torch.float64))
    torch.testing.assert_close(features[labels == 1], torch.tensor([[0.0, 1.0]], dtype=torch.float64))
    torch.testing.assert_close(features.norm(dim=1), torch.ones(len(labels), dtype=torch.float64))
    torch.testing.assert_close(references.norm(dim=1), torch.ones(100, dtype=torch.float64))
    assert torch.all(masses > 0)
    torch.testing.assert_close(masses, torch.full((100,), 0.01, dtype=torch.float64))
    assert masses.sum().item() == pytest.approx(1.0)
    torch.testing.assert_close(raw_features, original_features, rtol=0, atol=0)
    torch.testing.assert_close(labels, original_labels, rtol=0, atol=0)
    assert features.data_ptr() != raw_features.data_ptr()


def test_ot_references_handle_zero_examples_and_zero_class_means():
    labels = torch.cat([torch.arange(100), torch.tensor([1])])
    raw_features = torch.ones(101, 2)
    raw_features[0] = 0
    raw_features[1] = torch.tensor([1.0, -1.0])
    raw_features[-1] = torch.tensor([-1.0, 1.0])
    raw_features.requires_grad_(True)

    features, references, masses = reweighting._build_ot_references(raw_features, labels)

    assert torch.isfinite(features).all()
    assert torch.isfinite(references).all()
    torch.testing.assert_close(features[0], torch.zeros(2))
    torch.testing.assert_close(references[:2], torch.zeros(2, 2))
    assert not features.requires_grad and not references.requires_grad
    assert not masses.requires_grad


def test_ot_references_reject_missing_classes():
    with pytest.raises(ValueError, match=r"missing classes: \[99\]"):
        reweighting._build_ot_references(torch.ones(99, 2), torch.arange(99))


def test_ot_references_reject_nonfinite_class_means():
    raw_features = torch.ones(100, 2)
    raw_features[0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="Nonfinite class means"):
        reweighting._build_ot_references(raw_features, torch.arange(100))


def test_ot_cost_values_alignment_and_zero_vectors():
    features = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0]], dtype=torch.float64)
    references = torch.tensor([[1.0, 0.0]] * 100, dtype=torch.float64)
    references[99] = torch.tensor([0.0, 1.0])
    labels = torch.tensor([99, 0, 1])
    original_features = features.clone()
    original_references = references.clone()
    features.requires_grad_(True)
    references.requires_grad_(True)

    cost = reweighting._build_ot_cost(features, references, labels)

    assert cost.shape == (3, 100)
    assert cost.dtype == torch.float64
    assert not cost.requires_grad
    assert torch.all(cost >= 0)
    assert cost[0, 0].item() == pytest.approx(0.01)  # Same features, wrong class.
    assert cost[0, 99].item() == pytest.approx(0.5)  # Orthogonal, correct class.
    assert cost[1, 0].item() == pytest.approx(1.0)  # Opposite, correct class.
    assert cost[1, 1].item() == pytest.approx(1.01)  # Opposite, wrong class.
    assert cost[2, 1].item() == pytest.approx(0.5)  # Zero vector, correct class.
    assert cost[2, 0].item() == pytest.approx(0.51)
    torch.testing.assert_close(features, original_features, rtol=0, atol=0)
    torch.testing.assert_close(references, original_references, rtol=0, atol=0)

    # Independently express the release's feature and one-hot-label distances.
    reference_labels = torch.arange(100)
    feature_distance = 1 - torch.nn.functional.cosine_similarity(
        features[:, None, :], references[None, :, :], dim=-1
    )
    example_onehot = torch.nn.functional.one_hot(labels, 100).to(torch.float64)
    reference_onehot = torch.nn.functional.one_hot(reference_labels, 100).to(torch.float64)
    label_distance = (example_onehot[:, None, :] - reference_onehot[None, :, :]).square().mean(-1)
    torch.testing.assert_close(cost, 0.5 * feature_distance + 0.5 * label_distance)


@pytest.mark.parametrize("feature_scale, label_cost", [(0.0, 0.7), (0.3, 0.0), (0.3, 0.7)])
def test_ot_cost_coefficients_are_configurable(feature_scale, label_cost):
    features = torch.tensor([[1.0, 0.0]])
    references = torch.tensor([[0.0, 1.0]] * 100)
    cost = reweighting._build_ot_cost(
        features, references, torch.tensor([99]),
        feature_cost_scale=feature_scale, label_mismatch_cost=label_cost,
    )
    assert cost[0, 99].item() == pytest.approx(feature_scale)
    torch.testing.assert_close(cost[0, :99], torch.full((99,), feature_scale + label_cost))


@pytest.mark.parametrize("coefficients", [
    {"feature_cost_scale": -1}, {"label_mismatch_cost": -1},
    {"feature_cost_scale": True}, {"label_mismatch_cost": "0.01"},
    {"feature_cost_scale": float("nan")}, {"label_mismatch_cost": float("inf")},
    {"feature_cost_scale": 0, "label_mismatch_cost": 0},
])
def test_ot_cost_rejects_invalid_coefficients(coefficients):
    with pytest.raises(ValueError):
        reweighting._build_ot_cost(torch.ones(1, 2), torch.ones(100, 2), torch.tensor([0]), **coefficients)


def test_ot_cost_rejects_nonfinite_features():
    with pytest.raises(RuntimeError, match="Nonfinite OT costs"):
        reweighting._build_ot_cost(
            torch.full((1, 2), float("nan")), torch.ones(100, 2), torch.tensor([0])
        )


def test_ot_cost_clamps_cosine_roundoff():
    # A tiny overshoot beyond a unit norm can otherwise create negative costs.
    features = torch.tensor([[1.0 + 1e-7, 0.0]], dtype=torch.float64)
    references = torch.tensor([[1.0, 0.0]] * 100, dtype=torch.float64)
    cost = reweighting._build_ot_cost(features, references, torch.tensor([0]))
    assert cost[0, 0].item() == 0
    assert torch.all(cost >= 0)


@pytest.mark.parametrize("example_count", [1, 3, 10_847])
def test_example_scores_start_with_uniform_global_weights(example_count):
    scores = reweighting._initialize_example_scores(example_count, torch.device("cpu"))
    weights = reweighting._global_example_weights(scores)
    assert isinstance(scores, torch.nn.Parameter)
    assert scores.is_leaf and scores.requires_grad
    assert scores.shape == (example_count,)
    torch.testing.assert_close(scores, torch.zeros(example_count), rtol=0, atol=0)
    assert weights.requires_grad
    assert torch.all(weights > 0)
    torch.testing.assert_close(weights, torch.full((example_count,), 1 / example_count))
    assert weights.sum().item() == pytest.approx(1.0)


def test_global_weights_preserve_score_gradients():
    scores = reweighting._initialize_example_scores(3, torch.device("cpu"), dtype=torch.float64)
    costs = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)
    weights = reweighting._global_example_weights(scores)
    (weights * costs).sum().backward()
    torch.testing.assert_close(scores.grad, torch.tensor([-4 / 9, -1 / 9, 5 / 9], dtype=torch.float64))


def test_global_weights_match_known_distribution_without_changing_scores():
    scores = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64).log().requires_grad_()
    original_scores = scores.detach().clone()
    weights = reweighting._global_example_weights(scores)
    torch.testing.assert_close(weights, torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64))
    torch.testing.assert_close(scores, original_scores, rtol=0, atol=0)


@pytest.mark.parametrize("example_count", [0, -1, True, 1.5])
def test_example_scores_reject_invalid_counts(example_count):
    with pytest.raises(ValueError, match="example_count"):
        reweighting._initialize_example_scores(example_count, torch.device("cpu"))


@pytest.mark.parametrize("scores", [torch.empty(0), torch.zeros(2, 2)])
def test_global_weights_reject_invalid_score_shapes(scores):
    with pytest.raises(ValueError, match="nonempty one-dimensional"):
        reweighting._global_example_weights(scores)


@pytest.mark.parametrize("invalid_score", [float("nan"), float("inf"), -float("inf")])
def test_global_weights_reject_nonfinite_scores(invalid_score):
    with pytest.raises(RuntimeError, match="Nonfinite global example scores"):
        reweighting._global_example_weights(torch.tensor([invalid_score, 0.0]))


def test_global_weights_reject_softmax_underflow():
    with pytest.raises(RuntimeError, match="underflowed to zero"):
        reweighting._global_example_weights(torch.tensor([-1000.0, 0.0]))


def test_sinkhorn_matches_analytic_symmetric_solution():
    cost = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    masses = torch.full((2,), 0.5, dtype=torch.float64)
    transport, diagnostics = reweighting._sinkhorn_transport(cost, masses, masses)
    off_diagonal_ratio = torch.exp(torch.tensor(-10.0, dtype=torch.float64))
    diagonal = 0.5 / (1 + off_diagonal_ratio)
    expected = torch.stack([
        torch.stack([diagonal, diagonal * off_diagonal_ratio]),
        torch.stack([diagonal * off_diagonal_ratio, diagonal]),
    ])
    torch.testing.assert_close(transport, expected, rtol=1e-10, atol=1e-12)
    assert diagnostics["sinkhorn_iterations"] == 200
    assert diagnostics["row_marginal_error"] < 1e-12
    assert diagnostics["column_marginal_error"] < 1e-12


def test_sinkhorn_nonuniform_marginals_and_score_gradients_match_finite_differences():
    scores = torch.tensor([0.2, -0.4, 0.7], dtype=torch.float64, requires_grad=True)
    cost = torch.tensor([[0.1, 0.5], [0.3, 0.2], [0.8, 0.1]], dtype=torch.float64, requires_grad=True)
    references = torch.full((2,), 0.5, dtype=torch.float64)
    weights = reweighting._global_example_weights(scores)
    transport, diagnostics = reweighting._sinkhorn_transport(cost, weights, references)
    torch.testing.assert_close(transport.sum(1), weights, rtol=1e-10, atol=1e-12)
    torch.testing.assert_close(transport.sum(0), references, rtol=1e-10, atol=1e-12)
    assert diagnostics["row_marginal_error"] < 1e-10
    assert diagnostics["column_marginal_error"] < 1e-10

    # Entropy is used inside the solve, but not added to this outer objective.
    objective = (transport * cost.detach()).sum()
    objective.backward()
    assert cost.grad is None
    assert torch.isfinite(scores.grad).all()
    numerical_gradient = torch.zeros_like(scores)
    step = 1e-6
    for index in range(len(scores)):
        objectives = []
        for direction in (1, -1):
            perturbed_scores = scores.detach().clone()
            perturbed_scores[index] += direction * step
            perturbed_weights = reweighting._global_example_weights(perturbed_scores)
            plan, _ = reweighting._sinkhorn_transport(cost.detach(), perturbed_weights, references)
            objectives.append((plan * cost.detach()).sum())
        numerical_gradient[index] = (objectives[0] - objectives[1]) / (2 * step)
    torch.testing.assert_close(scores.grad, numerical_gradient, rtol=1e-5, atol=1e-8)
    changed_weights = reweighting._global_example_weights(scores.detach() - 0.1 * scores.grad)
    assert (changed_weights - weights.detach()).abs().max().item() > 1e-4


def test_sinkhorn_reports_marginal_error_when_iteration_budget_is_insufficient():
    cost = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    weights = torch.tensor([0.8, 0.2], dtype=torch.float64)
    references = torch.full((2,), 0.5, dtype=torch.float64)
    transport, diagnostics = reweighting._sinkhorn_transport(cost, weights, references, max_iterations=1)
    assert diagnostics["sinkhorn_iterations"] == 1
    assert diagnostics["row_marginal_error"] == pytest.approx((transport.sum(1) - weights).abs().max().item())
    assert diagnostics["row_marginal_error"] > 0.1
    assert diagnostics["column_marginal_error"] < 1e-12


def test_sinkhorn_is_stable_under_large_common_cost_offset():
    cost = torch.tensor([[0.0, 0.2], [0.2, 0.0]], dtype=torch.float64)
    masses = torch.full((2,), 0.5, dtype=torch.float64)
    expected, _ = reweighting._sinkhorn_transport(cost, masses, masses)
    transport, diagnostics = reweighting._sinkhorn_transport(cost + 1000, masses, masses)
    torch.testing.assert_close(transport, expected, rtol=1e-9, atol=1e-12)
    assert diagnostics["row_marginal_error"] < 1e-10
    assert diagnostics["column_marginal_error"] < 1e-10


def test_sinkhorn_single_example_has_reference_marginal_as_its_plan():
    cost = torch.tensor([[0.0, 0.5, 0.2]], dtype=torch.float64)
    references = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    transport, _ = reweighting._sinkhorn_transport(cost, torch.ones(1, dtype=torch.float64), references)
    torch.testing.assert_close(transport[0], references)


@pytest.mark.parametrize("settings", [
    {"epsilon": 0}, {"epsilon": -1}, {"epsilon": True}, {"epsilon": float("nan")},
    {"epsilon": float("inf")}, {"max_iterations": 0}, {"max_iterations": True},
    {"max_iterations": 1.5},
])
def test_sinkhorn_rejects_invalid_settings(settings):
    with pytest.raises(ValueError):
        reweighting._sinkhorn_transport(torch.zeros(2, 2), torch.full((2,), 0.5), torch.full((2,), 0.5), **settings)


@pytest.mark.parametrize("weights", [
    torch.tensor([0.0, 1.0]), torch.tensor([-0.1, 1.1]), torch.tensor([0.4, 0.4]),
    torch.tensor([float("nan"), 0.5]), torch.tensor([float("inf"), 0.5]),
    torch.ones(1), torch.full((2, 1), 0.5), torch.full((2,), 0.5, dtype=torch.float64),
])
@pytest.mark.parametrize("invalid_side", ["examples", "references"])
def test_sinkhorn_rejects_invalid_masses(weights, invalid_side):
    valid = torch.full((2,), 0.5)
    rows, columns = (weights, valid) if invalid_side == "examples" else (valid, weights)
    with pytest.raises(ValueError):
        reweighting._sinkhorn_transport(torch.zeros(2, 2), rows, columns)


@pytest.mark.parametrize("cost", [
    torch.empty(0, 2), torch.zeros(2), torch.zeros(2, 2, dtype=torch.int64),
    torch.tensor([[0.0, -1.0], [1.0, 0.0]]), torch.full((2, 2), float("nan")),
])
def test_sinkhorn_rejects_invalid_costs(cost):
    with pytest.raises(ValueError, match="cost"):
        reweighting._sinkhorn_transport(cost, torch.full((2,), 0.5), torch.full((2,), 0.5))


def test_score_update_matches_ot_gradient_and_returns_updated_detached_weights():
    scores = reweighting._initialize_example_scores(3, torch.device("cpu"), dtype=torch.float64)
    cost = torch.tensor([[0.1, 0.5], [0.3, 0.2], [0.8, 0.1]], dtype=torch.float64, requires_grad=True)
    references = torch.full((2,), 0.5, dtype=torch.float64, requires_grad=True)
    original_cost = cost.detach().clone()
    original_references = references.detach().clone()
    optimizer = torch.optim.SGD([scores], lr=0.1)
    initial_weights = reweighting._global_example_weights(scores)
    plan, expected_diagnostics = reweighting._sinkhorn_transport(cost, initial_weights, references)
    objective = (plan * cost.detach()).sum()
    expected_gradient, = torch.autograd.grad(objective, scores)

    weights, diagnostics = reweighting._update_example_scores(scores, optimizer, cost, references)

    torch.testing.assert_close(scores, -0.1 * expected_gradient)
    torch.testing.assert_close(weights, reweighting._global_example_weights(scores).detach())
    assert (weights - initial_weights.detach()).abs().max().item() > 1e-4
    assert weights.grad_fn is None and not weights.requires_grad
    assert weights.sum().item() == pytest.approx(1.0)
    assert torch.all(weights > 0)
    assert diagnostics == expected_diagnostics | {
        "score_updates": 1, "ot_cost_before_update": objective.item(),
    }
    assert cost.grad is None and references.grad is None
    torch.testing.assert_close(cost, original_cost, rtol=0, atol=0)
    torch.testing.assert_close(references, original_references, rtol=0, atol=0)


def test_configurable_score_update_count_matches_separate_calls_with_persistent_optimizer():
    combined_scores = reweighting._initialize_example_scores(3, torch.device("cpu"), dtype=torch.float64)
    separate_scores = reweighting._initialize_example_scores(3, torch.device("cpu"), dtype=torch.float64)
    cost = torch.tensor([[0.1, 0.5], [0.3, 0.2], [0.8, 0.1]], dtype=torch.float64)
    references = torch.full((2,), 0.5, dtype=torch.float64)
    # Nonzero momentum makes this check sensitive to accidentally resetting the
    # optimizer state between updates. These are test settings, not method defaults.
    combined_optimizer = torch.optim.SGD([combined_scores], lr=0.1, momentum=0.9)
    separate_optimizer = torch.optim.SGD([separate_scores], lr=0.1, momentum=0.9)
    combined_weights, combined_diagnostics = reweighting._update_example_scores(
        combined_scores, combined_optimizer, cost, references, score_updates=2
    )
    for _ in range(2):
        separate_weights, separate_diagnostics = reweighting._update_example_scores(
            separate_scores, separate_optimizer, cost, references
        )
    torch.testing.assert_close(combined_scores, separate_scores, rtol=0, atol=0)
    torch.testing.assert_close(combined_weights, separate_weights, rtol=0, atol=0)
    assert combined_diagnostics == separate_diagnostics | {"score_updates": 2}


@pytest.mark.parametrize("score_updates", [0, -1, True, 1.5])
def test_score_update_rejects_invalid_update_counts(score_updates):
    scores = reweighting._initialize_example_scores(2, torch.device("cpu"))
    optimizer = torch.optim.SGD([scores], lr=0.1)
    with pytest.raises(ValueError, match="score_updates"):
        reweighting._update_example_scores(
            scores, optimizer, torch.zeros(2, 2), torch.full((2,), 0.5),
            score_updates=score_updates,
        )
    torch.testing.assert_close(scores, torch.zeros(2), rtol=0, atol=0)


@pytest.mark.parametrize("learning_rate", [0.001, 0.005])
def test_adam_score_optimizer_uses_agreed_defaults_and_preserves_state(learning_rate):
    scores = reweighting._initialize_example_scores(3, torch.device("cpu"), dtype=torch.float64)
    if learning_rate == 0.001:
        optimizer = reweighting._make_score_optimizer(scores)
    else:
        optimizer = reweighting._make_score_optimizer(scores, learning_rate=learning_rate)
    assert isinstance(optimizer, torch.optim.Adam)
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert len(group["params"]) == 1 and group["params"][0] is scores
    assert group["lr"] == learning_rate
    assert group["betas"] == (0.9, 0.999)
    assert group["eps"] == 1e-8
    assert group["weight_decay"] == 0
    cost = torch.tensor([[0.1, 0.5], [0.3, 0.2], [0.8, 0.1]], dtype=torch.float64)
    references = torch.full((2,), 0.5, dtype=torch.float64)

    weights, _ = reweighting._update_example_scores(scores, optimizer, cost, references)
    # Adam's first bias-corrected update has this closed form at zero scores.
    expected_scores = -learning_rate * scores.grad / (scores.grad.abs() + 1e-8)
    torch.testing.assert_close(scores, expected_scores, rtol=1e-10, atol=1e-12)
    assert (weights - 1 / 3).abs().max().item() > 1e-5
    assert optimizer.state[scores]["step"].item() == 1
    first_moment = optimizer.state[scores]["exp_avg"].clone()

    reweighting._update_example_scores(scores, optimizer, cost, references)
    assert optimizer.state[scores]["step"].item() == 2
    torch.testing.assert_close(optimizer.state[scores]["exp_avg"], 0.9 * first_moment + 0.1 * scores.grad)
    assert optimizer.param_groups[0]["lr"] == learning_rate


@pytest.mark.parametrize("learning_rate", [0, -0.001, True, "0.001", float("nan"), float("inf")])
def test_score_optimizer_rejects_invalid_learning_rates(learning_rate):
    scores = reweighting._initialize_example_scores(3, torch.device("cpu"))
    with pytest.raises(ValueError, match="score learning_rate"):
        reweighting._make_score_optimizer(scores, learning_rate=learning_rate)


def test_uniform_weighted_loss_matches_cross_entropy_and_its_gradients():
    seed_everything(11)
    logits = torch.randn(5, 100, dtype=torch.float64, requires_grad=True)
    labels = torch.tensor([0, 99, 1, 5, 2])
    weights = torch.full((5,), 0.2, dtype=torch.float64, requires_grad=True)
    ordinary_logits = logits.detach().clone().requires_grad_()
    loss = reweighting._weighted_cross_entropy(logits, labels, weights, 5)
    ordinary_loss = torch.nn.functional.cross_entropy(ordinary_logits, labels)
    loss.backward()
    ordinary_loss.backward()
    torch.testing.assert_close(loss, ordinary_loss)
    torch.testing.assert_close(logits.grad, ordinary_logits.grad)
    assert weights.grad is None


def test_classifier_epoch_matches_manual_shuffled_weighting_and_preserves_backbone():
    seed_everything(12)
    model = create_model()
    reweighting._freeze_backbone(model)
    backbone_state = deepcopy(model.backbone.state_dict())
    expected_classifier = deepcopy(model.fc)
    features = torch.randn(7, 64)
    labels = torch.tensor([0, 1, 99, 5, 1, 0, 2])
    weights = (torch.arange(1, 8, dtype=torch.float32) / 28).requires_grad_()
    positions = torch.tensor([3, 0, 6, 2, 1, 5, 4])
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features[positions], labels[positions], positions), batch_size=3,
    )
    optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.02, momentum=0.9)
    expected_optimizer = torch.optim.SGD(expected_classifier.parameters(), lr=0.02, momentum=0.9)
    mean_loss = reweighting._train_classifier_epoch(model, loader, optimizer, weights, torch.device("cpu"))
    expected_loss_sum = 0.0
    for batch_positions in positions.split(3):
        expected_optimizer.zero_grad(set_to_none=True)
        losses = torch.nn.functional.cross_entropy(
            expected_classifier(features[batch_positions]), labels[batch_positions], reduction="none"
        )
        loss = (losses * (7 * weights.detach()[batch_positions])).mean()
        loss.backward()
        expected_optimizer.step()
        expected_loss_sum += loss.item() * len(batch_positions)
    assert mean_loss == pytest.approx(expected_loss_sum / 7)
    for name, value in model.fc.state_dict().items():
        torch.testing.assert_close(value, expected_classifier.state_dict()[name], rtol=0, atol=0)
    for name, value in model.backbone.state_dict().items():
        torch.testing.assert_close(value, backbone_state[name], rtol=0, atol=0)
    assert all(not module.training for module in model.backbone.modules())
    assert weights.grad is None


def test_score_update_refuses_unconverged_transport_before_changing_scores():
    scores = torch.tensor([0.8, 0.2], dtype=torch.float64).log().requires_grad_()
    original_scores = scores.detach().clone()
    optimizer = reweighting._make_score_optimizer(scores)
    cost = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float64)
    references = torch.full((2,), 0.5, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="Sinkhorn marginal relative errors"):
        reweighting._update_example_scores(scores, optimizer, cost, references, max_iterations=1)
    torch.testing.assert_close(scores, original_scores, rtol=0, atol=0)
    assert not optimizer.state


def test_reweighting_configuration_matches_defaults_and_does_not_share_nested_settings():
    path = Path(__file__).resolve().parents[1] / "configs" / "reweighting.json"
    config = json.loads(path.read_text())
    original_config = deepcopy(config)
    options = reweighting.prepare_method_options(config["method_options"])
    assert options == reweighting.prepare_method_options({})
    options["pretraining"]["decay_epochs"].append(190)
    options["classifier"]["epochs"] = 1
    assert config == original_config
    assert reweighting.prepare_method_options({})["classifier"]["epochs"] == 40


@pytest.mark.parametrize("options", [
    {"unknown": 1}, {"pretraining": None}, {"pretraining": {"epochs": 0}},
    {"classifier": []}, {"classifier": {"unknown": 1}}, {"classifier": {"epochs": True}},
    {"classifier": {"learning_rate": 0}}, {"classifier": {"momentum": 1}},
    {"classifier": {"weight_decay": -1}}, {"classifier": {"scheduler": "cosine"}},
    {"classifier": {"nesterov": 1}}, {"classifier": {"nesterov": True, "momentum": 0}},
    {"score_learning_rate": 0}, {"score_learning_rate": float("nan")}, {"score_updates": 0},
    {"epsilon": 0}, {"sinkhorn_iterations": True}, {"marginal_tolerance": 0},
    {"feature_cost_scale": -1}, {"label_mismatch_cost": True},
    {"feature_cost_scale": 0, "label_mismatch_cost": 0},
])
def test_reweighting_rejects_invalid_method_options(options):
    with pytest.raises(ValueError):
        reweighting.prepare_method_options(options)


def test_missing_class_fails_before_pretraining(tmp_path, training_data, monkeypatch):
    def forbid_pretraining(*args):
        pytest.fail("Missing classes must be rejected before image training")

    monkeypatch.setattr(reweighting, "_pretrain", forbid_pretraining)
    config = {"method_options": reweighting.prepare_method_options({})}
    with pytest.raises(ValueError, match="every class"):
        reweighting.train(create_model(), training_data, config, tmp_path, torch.device("cpu"))


def test_effective_sample_size_handles_probability_rounding_without_changing_weights():
    uniform_weights = torch.full((10_847,), 0.99999 / 10_847)
    original_weights = uniform_weights.clone()
    assert reweighting._effective_sample_size(uniform_weights) == pytest.approx(10_847)
    torch.testing.assert_close(uniform_weights, original_weights, rtol=0, atol=0)
    assert reweighting._effective_sample_size(torch.tensor([0.2, 0.3, 0.5])) == pytest.approx(1 / 0.38)
