import json

import pytest
import torch

from pipeline.methods.synthesis import (
    barycentric_features, fit_guide, geometry_cost, guide_loss,
    guidance_penalty, kl_project, marginal_errors, negative_score_moments,
    optimize_transport, resolve_options, run_generation, sample_prior,
    synthesize_class, synthetic_count, target_masses, transport_objective,
)


@pytest.fixture
def toy():
    positive = torch.tensor([[1., 0.], [1.5, .3], [.7, -.2]], dtype=torch.float64)
    negative = -positive
    weight = torch.tensor([.8, -.2], dtype=torch.float64)
    options = resolve_options({
        "target_count": 6, "guide_steps": 30, "max_iterations": 8,
        "projection_iterations": 1000, "marginal_tolerance": 1e-10,
        "objective_tolerance": 1e-12,
    })
    return positive, negative, weight, options


@pytest.mark.parametrize("real,target,expected", [(5, 500, 495), (500, 500, 0), (501, 500, 0)])
def test_synthetic_counts(real, target, expected):
    assert synthetic_count(real, target) == expected


@pytest.mark.parametrize("bad", [
    {"unknown": 1}, {"temperature": 0}, {"max_iterations": 0},
    {"step_size": -1}, {"target_count": True}, {"guidance_weight": float("nan")},
])
def test_bad_options(bad):
    with pytest.raises(ValueError):
        resolve_options(bad)


def test_guide_separates_small_binary_problem(toy):
    positive, negative, _, options = toy
    options = options | {"guide_steps": 300}
    weight, diagnostics = fit_guide(positive, negative, options)
    assert not weight.requires_grad
    assert (positive @ weight > 0).all()
    assert (negative @ weight < 0).all()
    assert diagnostics["loss"] < 0.1
    zeros = torch.zeros_like(weight)
    assert guide_loss(zeros, positive, negative, .001).item() == pytest.approx(1.0)


def test_target_masses_and_hard_examples(toy):
    positive, _, weight, _ = toy
    masses = target_masses(positive, weight)
    assert (masses > 0).all()
    assert masses.sum().item() == pytest.approx(1.0)
    losses = torch.relu(1 - positive @ weight)
    assert masses.argmax() == losses.argmax()
    uniform = target_masses(positive, weight, uniform=True)
    torch.testing.assert_close(uniform, torch.full_like(uniform, 1 / len(positive)))


def test_kl_projection_satisfies_marginals_and_preserves_cross_ratios():
    log_q = torch.tensor([[2., -1.], [.3, .8], [-2., 1.]], dtype=torch.float64)
    a = torch.tensor([.2, .3, .5], dtype=torch.float64)
    b = torch.tensor([.4, .6], dtype=torch.float64)
    transport, _ = kl_project(log_q, a, b, max_iterations=2000, tolerance=1e-11)
    assert max(marginal_errors(transport, a, b)) <= 1e-11
    # Diagonal matrix scaling preserves this ratio; replacing Q would not.
    expected = (log_q[0, 0] + log_q[1, 1] - log_q[0, 1] - log_q[1, 0]).exp()
    actual = transport[0, 0] * transport[1, 1] / (transport[0, 1] * transport[1, 0])
    torch.testing.assert_close(actual, expected)


def test_projection_reports_failure():
    log_q = torch.tensor([[8., -8.], [-8., 8.]], dtype=torch.float64)
    a = torch.tensor([.9, .1], dtype=torch.float64)
    b = torch.tensor([.1, .9], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="marginal"):
        kl_project(log_q, a, b, max_iterations=1, tolerance=1e-14)


def test_barycentric_features_are_convex_combinations(toy):
    positive, _, _, _ = toy
    transport = torch.tensor([[.1, .2, .3], [.2, .1, .1]], dtype=torch.float64)
    coefficients = transport / transport.sum(1, keepdim=True)
    assert (coefficients >= 0).all()
    torch.testing.assert_close(coefficients.sum(1), torch.ones(2, dtype=torch.float64))
    synthetic = barycentric_features(transport, positive)
    torch.testing.assert_close(synthetic, coefficients @ positive)
    assert (synthetic >= positive.min(0).values).all()
    assert (synthetic <= positive.max(0).values).all()


def test_guidance_moments_match_pairwise_value_and_gradient(toy):
    positive, negative, weight, _ = toy
    synthetic = positive.clone().requires_grad_(True)
    fast = guidance_penalty(synthetic, weight, *negative_score_moments(negative, weight))
    explicit = .5 * (
        (synthetic @ weight)[:, None] - (negative @ weight)[None, :] - 1
    ).square().mean()
    torch.testing.assert_close(fast, explicit)
    torch.testing.assert_close(
        torch.autograd.grad(fast, synthetic)[0],
        torch.autograd.grad(explicit, synthetic)[0],
    )


def test_full_objective_gradient_matches_finite_differences(toy):
    positive, negative, weight, options = toy
    prior = sample_prior(positive, 2, 9)
    cost = geometry_cost(prior, positive)
    moments = negative_score_moments(negative, weight)
    transport = torch.full((2, 3), 1 / 6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda t: transport_objective(t, cost, positive, weight, moments, options),
        (transport,), eps=1e-6, atol=1e-5, rtol=1e-4,
    )


def test_disabling_guidance_removes_its_gradient(toy):
    positive, negative, weight, options = toy
    options = options | {"guidance_weight": 0.0}
    cost = geometry_cost(sample_prior(positive, 2, 0), positive)
    moments = negative_score_moments(negative, weight)
    transport = torch.full((2, 3), 1 / 6, dtype=torch.float64, requires_grad=True)
    loss = transport_objective(transport, cost, positive, weight, moments, options)
    gradient, = torch.autograd.grad(loss, transport)
    expected = options["transport_weight"] * cost + options["entropy_coefficient"] * transport.log()
    torch.testing.assert_close(gradient, expected)


def test_projected_optimization_is_feasible_and_descends(toy):
    positive, negative, weight, options = toy
    prior = sample_prior(positive, 3, 0)
    b = target_masses(positive, weight)
    transport, diagnostics = optimize_transport(prior, positive, negative, weight, b, options)
    a = torch.full((3,), 1 / 3, dtype=torch.float64)
    assert max(marginal_errors(transport, a, b)) <= options["marginal_tolerance"]
    values = diagnostics["objective_history"]
    assert all(after <= before + 1e-10 for before, after in zip(values, values[1:]))
    assert not transport.requires_grad


def test_one_feature_class_and_at_target(toy):
    positive, negative, _, options = toy
    single = positive[:1]
    synthetic, _ = synthesize_class(single, negative, options, seed=0)
    torch.testing.assert_close(synthetic, single.expand(5, -1))
    empty, diagnostics = synthesize_class(positive, negative, options | {"target_count": 3}, 0)
    assert empty.shape == (0, 2)
    assert diagnostics["status"] == "at_target"


def test_same_seed_reproduces_prior_and_generation(toy):
    positive, negative, _, options = toy
    torch.testing.assert_close(sample_prior(positive, 3, 7), sample_prior(positive, 3, 7),
                               rtol=0, atol=0)
    first, _ = synthesize_class(positive, negative, options, seed=7)
    second, _ = synthesize_class(positive, negative, options, seed=7)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_cache_generation_scaling_class_isolation_and_resume(tmp_path):
    # One real feature per selected class must be reproduced in RAW coordinates.
    raw = torch.stack([torch.ones(64) * 4, torch.ones(64) * -6])
    labels = torch.tensor([0, 1], dtype=torch.int64)
    cache = {
        "features": raw, "labels": labels,
        "class_counts": torch.bincount(labels, minlength=100),
        "feature_scale": 5.0, "data_settings": {"data_seed": 42},
    }
    source = tmp_path / "train_features.pt"
    torch.save(cache, source)
    output = tmp_path / "generation"
    config = {
        "feature_cache": str(source), "output_dir": str(output), "seed": 0,
        "method_options": {"target_count": 3, "guide_steps": 5, "max_iterations": 3},
    }
    source_bytes = source.read_bytes()
    result = run_generation(config, classes=[0, 1])
    for class_id in (0, 1):
        actual = result["features"][result["labels"] == class_id]
        torch.testing.assert_close(actual, raw[class_id].expand(2, -1))
    assert result["features"].shape == (4, 64)
    assert source.read_bytes() == source_bytes
    assert torch.equal(result["real_class_counts"], cache["class_counts"])
    saved = torch.load(output / "synthetic_features.pt", weights_only=True)
    torch.testing.assert_close(saved["features"], result["features"])
    assert not json.loads((output / "generation_summary.json").read_text())["all_classes_processed"]
    resumed = run_generation(config, classes=[0, 1], resume=True)
    torch.testing.assert_close(resumed["features"], result["features"], rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        run_generation(config, classes=[0, 1])
    changed = config | {"seed": 1}
    with pytest.raises(ValueError, match="Resume"):
        run_generation(changed, classes=[0, 1], resume=True)
