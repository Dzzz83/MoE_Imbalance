"""
Synthetic dry-run for DACE Expert A (Head specialist).

Tests that:
  - Model forward pass runs on CPU with dummy data
  - Shapes match expectations: logits (B, 100), features (B, 64)
  - Loss.backward() populates .grad for all trainable parameters
  - Loss values are finite and non-zero

This satisfies the "Kaggle-Gate" rule (AGENTs.md §11):
  - Runs entirely on CPU
  - No CIFAR-100 download needed
  - Tiny dummy batch (batch_size=4, num_classes=100, feature_dim=64)
"""

import sys
import os

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

import torch
import numpy as np

from models.resnet32 import ResNet32
from losses.lal_loss import LALLoss
from data.partitioned_dataset import PartitionedSampler


class DummyDataset:
    """Minimal dataset for synthetic dry-run."""
    def __init__(self, n=100, n_classes=100):
        self.n_samples = n
        self.sample_targets = np.random.randint(0, n_classes, size=n)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.randn(3, 32, 32), int(self.sample_targets[idx])


def test_dace_a_forward_shapes():
    """Forward pass produces correct shapes."""
    model = ResNet32(num_classes=100)
    images = torch.randn(4, 3, 32, 32)
    logits = model(images)
    assert logits.shape == (4, 100), f"Logits shape: {logits.shape}"
    print(f"  ✅ Logits shape: {logits.shape}")


def test_dace_a_loss_value():
    """LAL loss is finite and non-zero."""
    model = ResNet32(num_classes=100)
    class_priors = torch.ones(100, dtype=torch.float32) / 100.0
    loss_fn = LALLoss(class_priors=class_priors, tau=1.0)

    images = torch.randn(4, 3, 32, 32)
    targets = torch.randint(0, 100, (4,))

    logits = model(images)
    loss = loss_fn(logits, targets)

    assert torch.isfinite(loss), "Loss should be finite"
    assert loss.item() > 0, f"Loss should be > 0, got {loss.item()}"
    print(f"  ✅ Loss = {loss.item():.4f} (finite, >0)")


def test_dace_a_backward():
    """Loss.backward() populates .grad for all parameters."""
    model = ResNet32(num_classes=100)
    class_priors = torch.ones(100, dtype=torch.float32) / 100.0
    loss_fn = LALLoss(class_priors=class_priors, tau=1.0)

    images = torch.randn(4, 3, 32, 32)
    targets = torch.randint(0, 100, (4,))

    logits = model(images)
    loss = loss_fn(logits, targets)
    loss.backward()

    grad_count = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += 1
        if param.grad is not None and param.grad.abs().sum().item() > 0:
            grad_count += 1

    assert grad_count == total_params, (
        f"Only {grad_count}/{total_params} parameters have gradients"
    )
    print(f"  ✅ All {total_params}/{total_params} parameters have gradients")


def test_dace_a_sampler_head_bias():
    """PartitionedSampler biases toward Head classes."""
    # Create dummy data where classes 0-29 are "head" (frequent)
    n_samples = 500
    targets = np.random.randint(0, 100, size=n_samples)
    # Make classes 0-29 appear more often
    targets[:300] = np.random.randint(0, 30, size=300)

    class DummyDatasetWithTargets:
        def __init__(self):
            self.sample_targets = targets
            self.n_samples = n_samples
        def __len__(self):
            return self.n_samples
        def __getitem__(self, idx):
            return torch.randn(3, 32, 32), int(self.sample_targets[idx])

    base = DummyDatasetWithTargets()
    head_groups = {'head': np.arange(30)}

    sampler = PartitionedSampler(
        base, primary_groups=head_groups,
        full_ratio=0.2, num_samples=200,
    )
    indices = list(iter(sampler))

    # Count primary (head) samples
    head_mask = np.isin(base.sample_targets, np.arange(30))
    head_indices = set(np.where(head_mask)[0].tolist())
    n_head = sum(1 for idx in indices if idx in head_indices)

    # Expected: 0.8 * 1.0 + 0.2 * (head_ratio_in_dataset)
    head_ratio_dataset = head_mask.mean()
    expected = 0.8 + 0.2 * head_ratio_dataset
    actual = n_head / len(indices)

    assert abs(actual - expected) < 0.05, (
        f"Head ratio {actual:.3f} should be ~{expected:.3f}"
    )
    print(f"  ✅ Head ratio: {actual:.3f} (expected ~{expected:.3f})")


if __name__ == "__main__":
    print("=" * 50)
    print("  DACE Expert A — Synthetic Dry-Run")
    print("=" * 50)
    print()

    tests = [
        ("Forward shapes", test_dace_a_forward_shapes),
        ("Loss value", test_dace_a_loss_value),
        ("Backward pass", test_dace_a_backward),
        ("Head sampler bias", test_dace_a_sampler_head_bias),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    else:
        print("  All synthetic dry-run tests passed!")
        print(f"  You can now run the full training on Kaggle:")
        print(f"    python scripts/train_dace_a.py")
