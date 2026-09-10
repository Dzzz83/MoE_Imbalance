"""
PartitionedDataset — class-group-biased sampling for DACE cascade training.

Each DACE expert specializes in a class group (Head, Med, or Tail) while
retaining general feature quality through a "full mix" ratio.

Sampling strategy (via PartitionedSampler):
  - With probability `full_ratio` (default 0.2): sample uniformly from ALL classes
  - With probability (1 - `full_ratio`): sample uniformly from the PRIMARY class group

Usage:
    train_set = LongTailCIFAR100(...)   # base training set
    partitioned = PartitionedDataset(base_dataset=train_set)
    sampler = PartitionedSampler(
        base_dataset=train_set,
        primary_groups={'head': head_indices},
        full_ratio=0.2,
    )
    loader = DataLoader(partitioned, batch_size=128, sampler=sampler)

No data leakage: only the training set indices are used.
"""

from __future__ import annotations

import math
import numpy as np
import torch
from torch.utils.data import Dataset


class PartitionedDataset(Dataset):
    """
    Wraps a base CIFAR-100-LT dataset.  Simply delegates to the base dataset;
    class-group-biased sampling is handled by PartitionedSampler.

    Args:
        base_dataset: The underlying LongTailCIFAR100 training set.
    """

    def __init__(self, base_dataset: Dataset):
        super().__init__()
        self.base_dataset = base_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.base_dataset[idx]


class PartitionedSampler(torch.utils.data.Sampler):
    """
    Sampler that draws with replacement from primary and all-class pools.

    For each epoch, creates:
      - floor(full_ratio * N) samples from ALL classes
      - N - floor(full_ratio * N) samples from PRIMARY classes

    This ensures the exact ratio is maintained per epoch.

    Args:
        base_dataset: Dataset with 'sample_targets' or 'targets' attribute.
        primary_groups: Dict of class indices to bias toward, e.g.
                        {'head': np.array([0, 1, ...])}.
                        All groups listed are merged into the primary set.
        full_ratio: Fraction of each batch that samples from ALL classes
                    (default 0.2).
        num_samples: Number of samples per epoch (default: len(base_dataset)).
        generator: Optional torch.Generator for reproducibility.
        primary_name: Optional name for logging (e.g. 'Head', 'Med', 'Tail').
    """

    def __init__(
        self,
        base_dataset: Dataset,
        primary_groups: dict[str, np.ndarray],
        full_ratio: float = 0.2,
        num_samples: int | None = None,
        generator: torch.Generator | None = None,
        primary_name: str = "Primary",
    ):
        super().__init__()
        # Extract targets
        if hasattr(base_dataset, 'sample_targets'):
            all_targets = base_dataset.sample_targets
        elif hasattr(base_dataset, 'targets'):
            all_targets = base_dataset.targets
        else:
            raise AttributeError(
                "base_dataset must have 'sample_targets' or 'targets' attribute"
            )

        # Build primary class set
        primary_classes = np.unique(np.concatenate(list(primary_groups.values())))
        primary_mask = np.isin(all_targets, primary_classes)
        self.primary_indices = np.where(primary_mask)[0]
        self.all_indices = np.arange(len(base_dataset))
        self.full_ratio = full_ratio
        self.num_samples = num_samples or len(base_dataset)
        self.generator = generator

        n_primary = len(self.primary_indices)
        n_total = len(self.all_indices)
        print(f"[PartitionedSampler] {primary_name}: "
              f"{n_primary}/{n_total} samples "
              f"({100.0 * n_primary / n_total:.1f}%) in primary classes, "
              f"full_ratio={full_ratio}")

    def __iter__(self):
        """Yield indices for one epoch with the desired class ratio."""
        n_full = math.floor(self.full_ratio * self.num_samples)
        n_primary = self.num_samples - n_full

        gen = self.generator
        if gen is None:
            gen = torch.default_generator

        # Sample with replacement from primary indices
        primary_t = torch.from_numpy(self.primary_indices).long()
        if n_primary > 0:
            primary_choices = primary_t[
                torch.randint(0, len(primary_t), (n_primary,), generator=gen)
            ].tolist()
        else:
            primary_choices = []

        # Sample with replacement from all indices
        all_t = torch.from_numpy(self.all_indices).long()
        if n_full > 0:
            full_choices = all_t[
                torch.randint(0, len(all_t), (n_full,), generator=gen)
            ].tolist()
        else:
            full_choices = []

        # Combine and shuffle
        indices = primary_choices + full_choices
        shuffle_order = torch.randperm(len(indices), generator=gen).tolist()
        indices = [indices[i] for i in shuffle_order]

        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples
