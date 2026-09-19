"""Fold-aware datasets and loaders for nested OOF runs.

This module is deliberately separate from :mod:`data.lt_datamodule`.  The
existing data module serves the canonical full training population for the
baseline protocol; this one can only construct a dataset from a resolved
``OOFRunContext`` and validates the membership before creating a loader.

No test dataset path is exposed here.  The default dataset factory is told
explicitly to use the CIFAR-100 training split and to treat the supplied
indices as already subsampled canonical LT data.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from data.cifar_lt import LongTailCIFAR100
from data.nested_oof import NestedOOFFoldManager, OOFProtocolError
from data.protocol_splits import assert_no_test_leakage


class OOFDataError(OOFProtocolError):
    """Raised when a fold-aware dataset cannot satisfy its membership contract."""


class _IndexedPredictionDataset(Dataset):
    """Add original CIFAR training IDs to an inference-only dataset."""

    def __init__(self, base_dataset: Dataset, sample_indices: np.ndarray) -> None:
        self.base_dataset = base_dataset
        self.sample_indices = np.asarray(sample_indices, dtype=np.int64).copy()

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, position: int):
        image, label = self.base_dataset[position]
        return image, label, int(self.sample_indices[position])


class FoldAwareDataModule:
    """Construct loaders for one immutable nested-OOF run context.

    ``dataset_factory`` is injectable for synthetic tests.  The production
    default is ``LongTailCIFAR100`` and receives only canonical training
    indices from the fold manager.
    """

    def __init__(
        self,
        *,
        manager: NestedOOFFoldManager,
        root: str = "./data",
        imbalance_ratio: float = 100.0,
        batch_size: int = 128,
        num_workers: int = 0,
        pin_memory: bool = True,
        seed: int = 0,
        dataset_factory: Callable[..., Dataset] = LongTailCIFAR100,
        download: bool = False,
        max_batches: int | None = None,
    ) -> None:
        if imbalance_ratio != 100.0:
            raise OOFDataError(
                "OOF training is fixed to the canonical CIFAR-100-LT IR=100 "
                f"protocol, got imbalance_ratio={imbalance_ratio}"
            )
        if batch_size < 1:
            raise OOFDataError(f"batch_size must be positive, got {batch_size}")
        if num_workers < 0:
            raise OOFDataError(f"num_workers must be non-negative, got {num_workers}")
        if max_batches is not None and max_batches < 1:
            raise OOFDataError(f"max_batches must be positive, got {max_batches}")

        self.manager = manager
        self.root = str(root)
        self.imbalance_ratio = float(imbalance_ratio)
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.seed = int(seed)
        self.dataset_factory = dataset_factory
        self.download = bool(download)
        self.max_batches = max_batches

        self._labels_by_index = dict(
            zip(manager.canonical_indices, manager.training_labels)
        )

    def validate_context(self, context: Any) -> None:
        """Validate context memberships against the frozen manager.

        This method performs no dataset construction.  Callers use it before
        constructing a trainer or a DataLoader.
        """
        if context.manager is not self.manager:
            raise OOFDataError("run context belongs to a different fold manager")

        try:
            outer = self.manager.outer_fold(context.outer_fold_id)
        except OOFProtocolError as exc:
            raise OOFDataError(str(exc)) from exc

        if context.inner_fold_id is None:
            expected_training = outer.expert_training_indices
            expected_prediction = outer.evaluation_indices
            expected_counts = outer.expert_training_class_counts
            expected_hash = outer.expert_training_membership_hash
        else:
            try:
                inner = self.manager.inner_fold(
                    context.outer_fold_id, context.inner_fold_id
                )
            except OOFProtocolError as exc:
                raise OOFDataError(str(exc)) from exc
            expected_training = inner.expert_training_indices
            expected_prediction = inner.prediction_indices
            expected_counts = inner.expert_training_class_counts
            expected_hash = inner.expert_training_membership_hash

        if tuple(context.training_indices) != tuple(expected_training):
            raise OOFDataError("run training membership disagrees with the fold manifest")
        if tuple(context.prediction_indices) != tuple(expected_prediction):
            raise OOFDataError(
                "run prediction membership disagrees with the fold manifest"
            )
        if tuple(context.training_class_counts) != tuple(expected_counts):
            raise OOFDataError("run class counts disagree with the fold manifest")
        if context.training_membership_hash != expected_hash:
            raise OOFDataError(
                "run training membership hash disagrees with the fold manifest"
            )

        training = set(context.training_indices)
        prediction = set(context.prediction_indices)
        if training & prediction:
            raise OOFDataError("expert training and prediction populations overlap")
        if not training or not prediction:
            raise OOFDataError("expert training and prediction populations must be non-empty")
        if any(count < 1 for count in context.training_class_counts):
            raise OOFDataError("fold-specific expert training drops a class")

    def class_counts(self, context: Any) -> np.ndarray:
        """Return loss statistics from this run's training population only."""
        self.validate_context(context)
        counts = np.asarray(context.training_class_counts, dtype=np.int64)
        if int(counts.sum()) != len(context.training_indices):
            raise OOFDataError("fold-specific class counts do not sum to training size")
        return counts.copy()

    def _build_dataset(
        self,
        indices: tuple[int, ...],
        *,
        train: bool,
    ) -> Dataset:
        sample_indices = np.asarray(indices, dtype=np.int64)
        assert_no_test_leakage(sample_indices, "OOF dataset")
        if len(np.unique(sample_indices)) != len(sample_indices):
            raise OOFDataError("OOF dataset membership contains duplicate sample IDs")
        if not set(sample_indices.tolist()) <= set(self._labels_by_index):
            raise OOFDataError("OOF dataset membership escapes the canonical population")

        # ``use_test_set=False`` is explicit even though it is the dataset
        # default: a future refactor must not turn an OOF loader into a test
        # loader accidentally.
        dataset = self.dataset_factory(
            root=self.root,
            base_train_indices=sample_indices.copy(),
            imbalance_ratio=self.imbalance_ratio,
            train=train,
            download=self.download,
            already_subsampled=True,
            use_test_set=False,
        )
        actual_indices = np.asarray(getattr(dataset, "sample_indices", ()), dtype=np.int64)
        if not np.array_equal(actual_indices, sample_indices):
            raise OOFDataError(
                "dataset factory changed the declared OOF membership; refusing to train"
            )
        actual_targets = np.asarray(
            getattr(dataset, "sample_targets", ()), dtype=np.int64
        )
        expected_targets = np.asarray(
            [self._labels_by_index[int(index)] for index in sample_indices],
            dtype=np.int64,
        )
        if not np.array_equal(actual_targets, expected_targets):
            raise OOFDataError(
                "dataset labels are not aligned with canonical training labels"
            )
        if bool(getattr(dataset, "train", train)) is not train:
            raise OOFDataError(
                f"dataset factory returned train={getattr(dataset, 'train', None)!r}; "
                f"expected {train!r}"
            )
        return dataset

    def training_dataset(self, context: Any) -> Dataset:
        """Return an augmented dataset over exactly the declared train IDs."""
        self.validate_context(context)
        return self._build_dataset(context.training_indices, train=True)

    def prediction_dataset(self, context: Any) -> Dataset:
        """Return a deterministic, non-augmented indexed prediction dataset."""
        self.validate_context(context)
        base = self._build_dataset(context.prediction_indices, train=False)
        return _IndexedPredictionDataset(
            base, np.asarray(context.prediction_indices, dtype=np.int64)
        )

    def training_loader(self, context: Any) -> DataLoader:
        """Build the training loader after membership validation."""
        dataset = self.training_dataset(context)
        if self.max_batches is not None:
            limit = min(len(dataset), self.max_batches * self.batch_size)
            if limit < len(dataset):
                print(
                    "[FoldAwareDataModule] DRY RUN: exposing "
                    f"{limit} of {len(dataset)} declared training samples — "
                    "not a reportable run"
                )
                dataset = Subset(dataset, range(limit))

        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            generator=generator,
        )

    # A descriptive alias keeps call sites explicit while matching the
    # existing data-module vocabulary used by the baseline trainer.
    train_loader = training_loader

    def prediction_loader(self, context: Any) -> DataLoader:
        """Build an ordered deterministic loader over the held-out population."""
        dataset = self.prediction_dataset(context)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
