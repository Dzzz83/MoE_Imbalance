"""
Long-tailed CIFAR-100 PyTorch Dataset.

Applies an exponential decay to the per-class sample count to simulate
a long-tailed distribution with a desired Imbalance Ratio (IR).

Usage:
    dataset = LongTailCIFAR100(
        root="./data",
        base_train_indices=base_indices,   # from split_cifar100.py
        imbalance_ratio=100,
        train=True,                        # applies augmentations
    )
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
"""

import math
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


class LongTailCIFAR100(Dataset):
    """
    Produces a long-tailed subset of CIFAR-100 from a pool of base-training
    indices. The per-class count follows:

        n_i = n_max * (IR) ^ (-i / (C - 1))

    where i = 0 is the head (most frequent class) and i = 99 is the tail.
    """

    def __init__(
        self,
        root: str = "./data",
        base_train_indices: np.ndarray | None = None,
        imbalance_ratio: float = 100.0,
        train: bool = True,
        download: bool = True,
        seed: int = 42,
        skip_longtail: bool = False,
        two_view: bool = False,
        already_subsampled: bool = False,
        use_test_set: bool = False,
    ):
        """
        Args:
            skip_longtail: If True, keep ALL samples from base_train_indices
                           without applying any long-tail subsampling.
                           Use this for the balanced validation set (old protocol).
            two_view: If True, return two independently augmented views per
                      sample (used by PaCo / contrastive training).
            already_subsampled: If True, base_train_indices are already
                                LT-subsampled. Skip internal subsampling.
                                Use this for the standard CIFAR-100-LT protocol.
            use_test_set: If True, load the original CIFAR-100 test set (10K).
                          Overrides base_train_indices and already_subsampled.
        """
        super().__init__()
        self.imbalance_ratio = imbalance_ratio
        self.train = train
        self.two_view = two_view

        # ── load data ──
        if use_test_set:
            # Load the original CIFAR-100 test set (10K balanced)
            test = datasets.CIFAR100(root=root, train=False, download=download)
            self.images = test.data
            self.targets = np.array(test.targets)
            self.base_indices = np.arange(len(self.images))
            self.classes = test.classes
            self.n_classes = len(test.classes)
            # Keep all test samples (no subsampling)
            self.sample_images = self.images
            self.sample_targets = self.targets
            self.sample_indices = self.base_indices.copy()
        else:
            # ── load the full CIFAR-100 training set ──
            full = datasets.CIFAR100(
                root=root, train=True, download=download
            )
            all_images = full.data          # ndarray (50000, 32, 32, 3)
            all_targets = np.array(full.targets)  # (50000,)
            self.classes = full.classes
            self.n_classes = len(full.classes)    # 100

            # ── restrict to the base-training pool ──
            if base_train_indices is not None:
                self.images = all_images[base_train_indices]
                self.targets = all_targets[base_train_indices]
                self.base_indices = base_train_indices.copy()
            else:
                # fallback: use EVERYTHING (no held-out val set)
                self.images = all_images
                self.targets = all_targets
                self.base_indices = np.arange(len(all_images))

            if skip_longtail:
                # ── keep ALL samples — no long-tail subsampling ──
                self.sample_images = self.images
                self.sample_targets = self.targets
                self.sample_indices = self.base_indices.copy()
            elif already_subsampled:
                # ── indices are already LT-subsampled — use directly ──
                self.sample_images = self.images
                self.sample_targets = self.targets
                self.sample_indices = self.base_indices.copy()
            else:
                # ── compute per-class long-tailed counts ──
                rng = np.random.default_rng(seed)
                n_available_per_class = self._count_per_class(self.targets)
                n_max = int(n_available_per_class[0].item())

                target_counts = self._exponential_counts(
                    n_max, self.n_classes, self.imbalance_ratio
                )
                target_counts = np.minimum(target_counts, n_available_per_class)

                # ── sample without replacement for each class ──
                chosen_indices: list[int] = []
                chosen_targets: list[int] = []

                for cls in range(self.n_classes):
                    cls_positions = np.where(self.targets == cls)[0]
                    n_keep = int(target_counts[cls].item())
                    assert n_keep <= len(cls_positions), (
                        f"Class {cls}: want {n_keep} but only {len(cls_positions)} available"
                    )
                    sampled = rng.choice(cls_positions, size=n_keep, replace=False)
                    chosen_indices.extend(sampled.tolist())
                    chosen_targets.extend([cls] * n_keep)

                order = np.argsort(chosen_indices)
                self.sample_indices = np.array(chosen_indices, dtype=np.int64)[order]
                self.sample_images = self.images[self.sample_indices]
                self.sample_targets = np.array(chosen_targets, dtype=np.int64)[order]

        # ── transforms ──
        if self.train:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
            ])

        # ── logging ──
        final_counts = self._count_per_class(self.sample_targets)
        head_count = int(final_counts[0].item())
        tail_count = int(final_counts[-1].item())
        actual_ir = head_count / max(tail_count, 1)

        print(f"[LongTailCIFAR100] {'Train' if train else 'Val'} set created:")
        print(f"  Total samples          : {len(self):,}")
        print(f"  Head class (0) count   : {head_count}")
        print(f"  Tail class (99) count  : {tail_count}")
        print(f"  Target imbalance ratio : {imbalance_ratio}")
        print(f"  Achieved IR            : {actual_ir:.2f}")
        print(f"  Augmentations           : {'Yes' if train else 'No'}")
        if skip_longtail:
            print(f"  (long-tail subsampling skipped — balanced set)")

    def _count_per_class(self, targets: np.ndarray) -> np.ndarray:
        """Return array of per-class sample counts (ordered 0..C-1)."""
        counts = np.zeros(self.n_classes, dtype=np.int64)
        for c in range(self.n_classes):
            counts[c] = int((targets == c).sum())
        return counts

    def _exponential_counts(
        self, n_max: int, n_classes: int, ir: float
    ) -> np.ndarray:
        """
        Compute n_i = n_max * ir ^ (-i / (n_classes - 1)).

        Uses int() truncation (floor for positives), matching the standard
        long-tail CIFAR protocol used by LDAM (Cao et al., NeurIPS 2019),
        RIDE (Wang et al., ICLR 2021), and PaCo (Cui et al., ICCV 2021).
        Guarantees at least 1 sample per class.
        """
        indices = np.arange(n_classes, dtype=np.float64)          # i = 0 .. 99
        exp = -indices / (n_classes - 1)                         # exponent
        raw = n_max * (ir ** exp)                                # float counts
        counts = np.maximum(raw.astype(np.int64), 1)             # ≥ 1, truncate
        return counts

    def get_class_counts(self) -> np.ndarray:
        """Return array of per-class sample counts in this dataset."""
        return self._count_per_class(self.sample_targets)

    def __len__(self) -> int:
        return len(self.sample_images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int] | tuple[list[torch.Tensor], int]:
        img = self.sample_images[idx]
        label = int(self.sample_targets[idx].item())
        if self.two_view:
            # Support both single-transform (applied twice) and per-view transforms (list of 2)
            if isinstance(self.transform, (list, tuple)):
                view1 = self.transform[0](img)
                view2 = self.transform[1](img)
            else:
                view1 = self.transform(img)
                view2 = self.transform(img)
            return [view1, view2], label
        else:
            if isinstance(self.transform, (list, tuple)):
                # If transform is a list but two_view=False, warn and use the first transform
                import warnings
                warnings.warn(
                    f"transform is a {type(self.transform).__name__} of length {len(self.transform)} "
                    f"but two_view=False. Using transform[0] only."
                )
                img_tensor = self.transform[0](img)
            else:
                img_tensor = self.transform(img)
            return img_tensor, label
