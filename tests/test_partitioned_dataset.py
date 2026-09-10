"""
Tests for data/partitioned_dataset.py — PartitionedDataset and PartitionedSampler.

Expected behavior:
  - PartitionedDataset delegates to base dataset
  - PartitionedSampler generates indices with correct class ratio
  - PartitionedSampler maintains ~full_ratio fraction of all-class samples
  - PartitionedSampler maintains ~(1-full_ratio) fraction of primary-class samples
  - Primary class set correctly computed from provided groups
  - Works with a synthetic dataset (no CIFAR-100 download needed)
"""

import sys
import os
import numpy as np
import torch

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from data.partitioned_dataset import PartitionedDataset, PartitionedSampler


class SyntheticDataset:
    """Minimal dataset with sample_targets for testing."""
    def __init__(self, n_samples: int, n_classes: int):
        self.n_samples = n_samples
        self.n_classes = n_classes
        # Create imbalanced targets: first 30 classes have most samples
        self.sample_targets = np.random.randint(0, n_classes, size=n_samples)
        self.data = np.random.randn(n_samples, 32, 32, 3)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), int(self.sample_targets[idx])


def test_partitioned_dataset_len():
    """PartitionedDataset.__len__ matches base dataset."""
    base = SyntheticDataset(100, 10)
    ds = PartitionedDataset(base)
    assert len(ds) == 100, f"Expected 100, got {len(ds)}"


def test_partitioned_dataset_getitem():
    """PartitionedDataset.__getitem__ delegates to base dataset."""
    base = SyntheticDataset(100, 10)
    ds = PartitionedDataset(base)
    img, label = ds[5]
    assert isinstance(img, torch.Tensor)
    assert isinstance(label, (int, np.integer))


def test_sampler_length():
    """PartitionedSampler.__len__ returns num_samples."""
    base = SyntheticDataset(100, 10)
    primary_groups = {'group1': np.array([0, 1, 2])}
    sampler = PartitionedSampler(
        base, primary_groups=primary_groups, full_ratio=0.2,
        num_samples=50, primary_name='Test',
    )
    assert len(sampler) == 50, f"Expected 50, got {len(sampler)}"


def test_sampler_class_ratio():
    """Sampler yields expected class distribution.

    Expected primary ratio = (1 - full_ratio) * 1.0 + full_ratio * (n_primary / n_total)
    i.e., all primary-pool samples are primary + some full-pool samples are primary.
    """
    base = SyntheticDataset(1000, 10)
    # Classes 0-4 are "primary"
    primary_groups = {'group1': np.array([0, 1, 2, 3, 4])}
    sampler = PartitionedSampler(
        base, primary_groups=primary_groups, full_ratio=0.2,
        num_samples=2000,  # large sample for stable ratio
    )

    indices = list(iter(sampler))
    assert len(indices) == 2000

    # Count how many indices point to primary-class samples
    primary_mask = np.isin(base.sample_targets, [0, 1, 2, 3, 4])
    primary_indices_set = set(np.where(primary_mask)[0].tolist())

    n_primary_actual = sum(1 for idx in indices if idx in primary_indices_set)
    primary_ratio = n_primary_actual / len(indices)

    # Expected: 0.8 * 1.0 + 0.2 * (fraction of dataset that is primary)
    n_primary_total = int(primary_mask.sum())
    expected_ratio = 0.8 + 0.2 * (n_primary_total / 1000)

    assert abs(primary_ratio - expected_ratio) < 0.05, (
        f"Primary ratio {primary_ratio:.3f} should be ~{expected_ratio:.3f}"
    )


def test_sampler_full_ratio_zero():
    """With full_ratio=0, all samples come from primary classes."""
    base = SyntheticDataset(1000, 10)
    primary_groups = {'group1': np.array([0, 1, 2])}
    sampler = PartitionedSampler(
        base, primary_groups=primary_groups, full_ratio=0.0,
        num_samples=500,
    )

    indices = list(iter(sampler))
    primary_mask = np.isin(base.sample_targets, [0, 1, 2])
    primary_indices_set = set(np.where(primary_mask)[0].tolist())

    n_primary = sum(1 for idx in indices if idx in primary_indices_set)
    assert n_primary == 500, f"With full_ratio=0, all should be primary, got {n_primary}/500"


def test_sampler_full_ratio_one():
    """With full_ratio=1.0, all samples come from all classes."""
    base = SyntheticDataset(1000, 10)
    primary_groups = {'group1': np.array([0, 1, 2])}
    sampler = PartitionedSampler(
        base, primary_groups=primary_groups, full_ratio=1.0,
        num_samples=500,
    )

    indices = list(iter(sampler))
    # All indices should be valid (within range)
    assert all(0 <= idx < 1000 for idx in indices)


def test_sampler_multiple_primary_groups():
    """Multiple primary groups are merged correctly."""
    base = SyntheticDataset(1000, 10)
    primary_groups = {
        'head': np.array([0, 1, 2]),
        'tail': np.array([8, 9]),
    }
    sampler = PartitionedSampler(
        base, primary_groups=primary_groups, full_ratio=0.2,
        num_samples=1000,
    )

    indices = list(iter(sampler))
    primary_mask = np.isin(base.sample_targets, [0, 1, 2, 8, 9])
    primary_indices_set = set(np.where(primary_mask)[0].tolist())

    n_primary = sum(1 for idx in indices if idx in primary_indices_set)
    primary_ratio = n_primary / len(indices)

    # Expected: 0.8 + 0.2 * (n_primary_total / 1000)
    n_primary_total = int(primary_mask.sum())
    expected_ratio = 0.8 + 0.2 * (n_primary_total / 1000)

    assert abs(primary_ratio - expected_ratio) < 0.05, (
        f"Primary ratio {primary_ratio:.3f} should be ~{expected_ratio:.3f}"
    )


def test_sampler_reproducibility():
    """Same generator seed produces same indices."""
    base = SyntheticDataset(100, 10)
    primary_groups = {'g1': np.array([0, 1, 2])}
    gen1 = torch.Generator()
    gen1.manual_seed(42)
    gen2 = torch.Generator()
    gen2.manual_seed(42)

    sampler1 = PartitionedSampler(base, primary_groups=primary_groups, generator=gen1)
    sampler2 = PartitionedSampler(base, primary_groups=primary_groups, generator=gen2)

    indices1 = list(iter(sampler1))
    indices2 = list(iter(sampler2))
    assert indices1 == indices2, "Same seed should produce same indices"


if __name__ == "__main__":
    tests = [
        ("PartitionedDataset len", test_partitioned_dataset_len),
        ("PartitionedDataset getitem", test_partitioned_dataset_getitem),
        ("Sampler length", test_sampler_length),
        ("Sampler class ratio", test_sampler_class_ratio),
        ("Sampler full_ratio=0", test_sampler_full_ratio_zero),
        ("Sampler full_ratio=1", test_sampler_full_ratio_one),
        ("Sampler multiple groups", test_sampler_multiple_primary_groups),
        ("Sampler reproducibility", test_sampler_reproducibility),
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
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*40}")
    print(f"  {passed} passed, {failed} failed")
    if failed > 0:
        sys.exit(1)
    else:
        print("  All tests passed!")
