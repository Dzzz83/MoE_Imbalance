"""
Data module for CIFAR-100-LT expert training.

Owns the whole "where do training samples come from" question so trainers never
touch index files. There is no validation split: the loader serves the full
canonical long-tailed training set, and the balanced 10K test set is loaded
separately by the evaluation code.

Deliberately independent of the config schema (dependency inversion): the config
layer adapts its values to these plain constructor arguments, not the reverse.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from data.cifar_lt import LongTailCIFAR100
from data.protocol_splits import load_lt_train_indices


class LongTailDataModule:
    """Builds the long-tailed training loader.

    Args:
        root: directory holding ``cifar-100-python`` and ``processed/``.
        imbalance_ratio: IR of the canonical training set (100 for this project).
        batch_size: training batch size.
        num_workers: DataLoader workers.
        pin_memory: pin host memory (useful on GPU).
        max_batches: if set, expose only the first N batches. This exists for
            CPU dry-runs and the AGENTs.md lightweight-verification gate; it must
            never be used for a run whose numbers are reported.
    """

    def __init__(
        self,
        root: str = './data',
        imbalance_ratio: float = 100.0,
        batch_size: int = 128,
        num_workers: int = 2,
        pin_memory: bool = True,
        max_batches: int | None = None,
        download: bool = False,
    ) -> None:
        self.root = root
        self.imbalance_ratio = imbalance_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_batches = max_batches
        self.download = download
        self._dataset: LongTailCIFAR100 | None = None
        self._class_counts: np.ndarray | None = None

    # ── dataset ───────────────────────────────────────────────────────

    @property
    def indices(self) -> np.ndarray:
        """The canonical long-tailed training indices."""
        return load_lt_train_indices(self.root)

    def dataset(self) -> LongTailCIFAR100:
        """The training dataset (all 10,847 long-tailed samples)."""
        if self._dataset is None:
            self._dataset = LongTailCIFAR100(
                root=self.root,
                base_train_indices=self.indices,
                imbalance_ratio=self.imbalance_ratio,
                train=True,
                download=self.download,
                already_subsampled=True,
            )
        return self._dataset

    def class_counts(self) -> np.ndarray:
        """Per-class sample counts of the training set (for head/med/tail splits)."""
        if self._class_counts is None:
            self._class_counts = self.dataset().get_class_counts()
        return self._class_counts

    # ── loader ────────────────────────────────────────────────────────

    def train_loader(self) -> DataLoader:
        """Training loader over the full long-tailed training set."""
        dataset = self.dataset()
        if self.max_batches is not None:
            n = min(len(dataset), self.max_batches * self.batch_size)
            if n < len(dataset):
                print(f"[LongTailDataModule] DRY RUN: exposing only {n} samples "
                      f"({self.max_batches} batches) — not a reportable run")
            dataset = Subset(dataset, range(n))
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )

    def describe(self) -> str:
        counts = self.class_counts()
        return (
            f"{len(self.indices)} samples, head={int(counts.max())}, "
            f"tail={int(counts.min())}, classes={len(counts)}"
        )
