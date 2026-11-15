import pytest
import torch

from pipeline.methods import sava_weights as sava


def test_smallest_scoring_batches_do_not_leave_a_singleton():
    positions = torch.arange(101)
    batches = sava._scoring_batches(positions, 2)
    assert all(len(batch) >= 2 for batch in batches)
    torch.testing.assert_close(torch.cat(batches), positions)


def test_dual_direction_and_marginals():
    options = sava.prepare_options({})
    cost = torch.tensor([[0., 0.], [2., 2.]], dtype=torch.float64)
    mass = torch.tensor([0.5, 0.5], dtype=torch.float64)
    plan, scores, _, error = sava._transport(cost, mass, mass, options)
    torch.testing.assert_close(plan.sum(1), mass)
    torch.testing.assert_close(plan.sum(0), mass)
    assert scores[0] < scores[1]
    assert error <= options["sinkhorn_tolerance"]
    with pytest.raises(RuntimeError, match="did not converge"):
        sava._transport(torch.tensor([[0., 0.], [0., 10.]], dtype=torch.float64),
                        mass, mass, options | {"sinkhorn_iterations": 1})


def test_class_balance_equal_scores_and_score_direction():
    labels = torch.cat((torch.arange(100), torch.tensor([0])))
    scores = torch.zeros(101, dtype=torch.float64)
    weights = sava._scores_to_weights(scores, labels, 1.)
    assert weights[0] == pytest.approx(0.005)
    assert weights[-1] == pytest.approx(0.005)
    scores[-1] = 1
    weights = sava._scores_to_weights(scores, labels, 1.)
    assert weights[0] > weights[-1]
    masses = torch.zeros(100, dtype=weights.dtype).scatter_add_(0, labels, weights)
    torch.testing.assert_close(masses, torch.full_like(masses, 0.01))


def test_reproducibility_positions_partial_batches_and_missing_class():
    labels = torch.arange(100).repeat(2)
    features = torch.randn(200, 8, generator=torch.Generator().manual_seed(3))
    options = sava.prepare_options({"reference_per_class": 1, "scoring_batch_size": 67})
    config = {"seed": 2, "method_options": {"sava": options}}
    first, diagnostics = sava.build_weights(features, labels, config, "cpu")
    second, repeated = sava.build_weights(features, labels, config, "cpu")
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert diagnostics == repeated
    assert len(set(diagnostics["reference_positions"])) == 100
    assert torch.isfinite(first).all()
    assert first.sum() == pytest.approx(1.)
    with pytest.raises(ValueError, match="Every class"):
        sava.build_weights(features[labels != 99], labels[labels != 99], config, "cpu")


@pytest.mark.parametrize("options", [{"temperature": 0}, {"epsilon": float("nan")},
                                     {"scoring_batch_size": 1}, {"unknown": 1}, [],
                                     {"reference_seed": 2**32}])
def test_invalid_options(options):
    with pytest.raises(ValueError):
        sava.prepare_options(options)
