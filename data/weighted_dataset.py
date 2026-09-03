"""
Weighted dataset wrapper for boosting-style training.

Wraps a base PyTorch Dataset and adds per-sample loss weights.
The wrapped dataset returns (image, target, weight) tuples,
where weight is a float32 scalar used to scale the loss for each sample.

Usage:
    base = LongTailCIFAR100(...)
    weights = np.ones(len(base))
    weights[error_mask] = 2.0  # upweight errors
    weighted = WeightedDataset(base, weights)
    loader = DataLoader(weighted, batch_size=128, shuffle=True)
    # loader now yields (images, targets, weights)
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class WeightedDataset(Dataset):
    """
    Wraps a base dataset and adds per-sample loss weights.

    The wrapped dataset returns (image, target, weight) tuples.
    The weight is a float32 scalar that the trainer uses to scale
    the loss for each sample.

    Args:
        base_dataset: Dataset returning (image, target) tuples.
        weights: 1D numpy array of shape (N,) with per-sample weights.
                 Must be non-negative. length must equal len(base_dataset).
    """

    def __init__(self, base_dataset: Dataset, weights: np.ndarray):
        assert len(base_dataset) == len(weights), (
            f"Dataset has {len(base_dataset)} samples but weights has {len(weights)}"
        )
        assert np.all(weights >= 0), "Weights must be non-negative"
        self.base_dataset = base_dataset
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, target = self.base_dataset[idx]
        return image, target, self.weights[idx]


class ConcatDataset(Dataset):
    """
    Concatenates multiple datasets. If any dataset returns 3 items
    (image, target, weight), the output includes weights.
    For datasets returning 2 items, weight defaults to 1.0.

    Useful for combining weighted validation set with unweighted training set.
    """

    def __init__(self, datasets: list[Dataset]):
        self.datasets = datasets
        self.cumulative_lengths = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative_lengths.append(total)

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        # Find which dataset this index belongs to
        for i, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                base_idx = idx - (self.cumulative_lengths[i - 1] if i > 0 else 0)
                result = self.datasets[i][base_idx]
                # Ensure 3-item tuple (image, target, weight)
                if len(result) == 2:
                    img, tgt = result
                    return img, tgt, torch.tensor(1.0, dtype=torch.float32)
                return result
        raise IndexError(f"Index {idx} out of range")
