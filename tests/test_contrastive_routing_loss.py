"""
Tests for losses/contrastive_routing_loss.py.

Expected behavior:
  - Loss = 0 when batch_size < 2 (trivial)
  - Loss > 0 when embeddings are random (no structure)
  - Loss decreases when embeddings are pushed to cluster by label
  - Gradient flows through embeddings
  - Different temperatures scale the loss magnitude
"""

import sys
import os
import torch

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from losses.contrastive_routing_loss import ContrastiveRoutingLoss

# Two assertions below compare random embeddings against clustered ones, so the
# fixture must be reproducible: without a seed the file's verdict changes from
# run to run.
torch.manual_seed(0)


def test_single_sample_no_loss():
    """batch_size=1 should return 0 loss (no pairs to contrast)."""
    loss_fn = ContrastiveRoutingLoss(temperature=0.5)
    emb = torch.randn(1, 32)
    labels = torch.tensor([0])
    loss, aux = loss_fn(emb, labels)
    assert loss.item() == 0.0, f"Single sample should give 0 loss, got {loss.item()}"


def test_single_sample_keeps_the_normal_path_contract():
    """The B<2 shortcut must return the same *types* as the main path.

    It used to return a plain float in ``aux`` and a loss with no graph, so a
    caller doing ``aux['contrastive_loss'].item()`` (as the DACE trainers do)
    crashed, and ``loss.backward()`` raised on a size-1 batch.
    """
    loss_fn = ContrastiveRoutingLoss(temperature=0.5)
    emb = torch.randn(1, 32, requires_grad=True)
    loss, aux = loss_fn(emb, torch.tensor([0]))

    assert isinstance(aux['contrastive_loss'], torch.Tensor), (
        f"aux['contrastive_loss'] is {type(aux['contrastive_loss']).__name__}, "
        f"but the main path returns a tensor"
    )
    assert loss.requires_grad, "the zero loss must still carry a graph"
    loss.backward()
    assert emb.grad is not None, "gradient should flow on the B<2 path too"
    print("  ✅ B<2 path returns a zero tensor with a graph")


def test_four_samples_loss_finite():
    """4 samples with mixed labels should produce a finite, non-negative loss."""
    loss_fn = ContrastiveRoutingLoss(temperature=0.5)
    emb = torch.randn(4, 32)
    labels = torch.tensor([0, 0, 1, 1])
    loss, aux = loss_fn(emb, labels)
    assert torch.isfinite(loss), "Loss should be finite"
    assert loss.item() >= 0, f"Loss should be >= 0, got {loss.item()}"


def test_gradient_flow():
    """Gradients should flow through embeddings."""
    loss_fn = ContrastiveRoutingLoss(temperature=0.5)
    emb = torch.randn(4, 32, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1])
    loss, aux = loss_fn(emb, labels)
    loss.backward()
    assert emb.grad is not None, "Gradient should flow to embeddings"
    assert emb.grad.abs().sum().item() > 0, "Gradient should be non-zero"


def test_loss_decreases_with_clustering():
    """Loss should decrease when same-label embeddings are pushed closer."""
    loss_fn = ContrastiveRoutingLoss(temperature=0.5)

    # Two clusters: 4 samples with label 0, 4 with label 1
    n_per = 4
    emb_random = torch.randn(2 * n_per, 32, requires_grad=True)
    labels = torch.tensor([0] * n_per + [1] * n_per)

    loss_random, _ = loss_fn(emb_random, labels)

    # Now create well-separated clusters
    emb_clustered = torch.zeros(2 * n_per, 32)
    emb_clustered[:n_per, 0] = 1.0  # label 0: first dim = 1
    emb_clustered[n_per:, 1] = 1.0  # label 1: second dim = 1

    loss_clustered, _ = loss_fn(emb_clustered, labels)

    assert loss_clustered.item() < loss_random.item(), (
        f"Clustered loss {loss_clustered.item():.4f} should be "
        f"less than random loss {loss_random.item():.4f}"
    )


def test_temperature_scaling():
    """Higher temperature should increase the loss magnitude (due to T/base_T scaling factor)."""
    loss_fn_hot = ContrastiveRoutingLoss(temperature=1.0)
    loss_fn_cold = ContrastiveRoutingLoss(temperature=0.1)

    emb = torch.randn(8, 32)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])

    loss_hot, _ = loss_fn_hot(emb, labels)
    loss_cold, _ = loss_fn_cold(emb, labels)

    # For random embeddings with uniform similarities ≈ 0, the loss
    # is approximately (T/base_T) * log(N-1).  Since T/base_T is larger
    # for T=1.0 (factor 2.0) than for T=0.1 (factor 0.2), hot loss > cold loss.
    assert loss_hot.item() > loss_cold.item(), (
        f"Hot loss {loss_hot.item():.4f} should be > "
        f"cold loss {loss_cold.item():.4f}"
    )


def test_aux_dict_contains_loss():
    """Aux dict should contain 'contrastive_loss' key."""
    loss_fn = ContrastiveRoutingLoss(temperature=0.5)
    emb = torch.randn(4, 32)
    labels = torch.tensor([0, 0, 1, 1])
    loss, aux = loss_fn(emb, labels)
    assert 'contrastive_loss' in aux
    assert aux['contrastive_loss'].item() == loss.item()


if __name__ == "__main__":
    tests = [
        ("Single sample no loss", test_single_sample_no_loss),
        ("Single sample keeps contract", test_single_sample_keeps_the_normal_path_contract),
        ("Four samples finite loss", test_four_samples_loss_finite),
        ("Gradient flow", test_gradient_flow),
        ("Loss decreases with clustering", test_loss_decreases_with_clustering),
        ("Temperature scaling", test_temperature_scaling),
        ("Aux dict contains loss", test_aux_dict_contains_loss),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✅ {name}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"  {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    else:
        print("  All tests passed!")
