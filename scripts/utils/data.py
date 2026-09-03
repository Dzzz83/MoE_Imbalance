"""
Data loading utilities for CIFAR-100-LT experiments.

Provides:
  - load_expert_checkpoint: Load any expert model from checkpoint
  - create_cifar_loader: Create DataLoader for train/val/test
  - get_class_groups: Head/medium/tail class grouping
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.cifar_lt import LongTailCIFAR100
from models.resnet32 import ResNet32, PaCoResNet32

EPS = 1e-12

# Default checkpoint directory
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"


# ── Model Loading ────────────────────────────────────────────────────────


def load_expert_checkpoint(
    expert_name: str,
    checkpoint_path: str | None = None,
    device: str = "cpu",
) -> torch.nn.Module:
    """Load a trained expert model from checkpoint.

    Args:
        expert_name: 'LAL', 'Mixup', 'PaCo', 'CE', or 'BalancedSoftmax'.
        checkpoint_path: Path to .pt file. If None, uses
            ``checkpoints/{expert_name}_best.pt``.
    Returns:
        Loaded model in eval mode on the specified device.
    """
    if checkpoint_path is None:
        checkpoint_path = str(DEFAULT_CHECKPOINT_DIR / f"{expert_name}_best.pt")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if expert_name.upper() == "PACO":
        model = PaCoResNet32(num_classes=100, dim=32, K=2048)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model = ResNet32(num_classes=100)
        model.load_state_dict(ckpt["model_state_dict"])

    model = model.to(device)
    model.eval()

    # Optionally attach metadata for convenience
    model._expert_name = expert_name
    model._checkpoint_epoch = ckpt.get("epoch", "?")
    model._checkpoint_ba = ckpt.get("best_metric_val",
                                     ckpt.get("log", {}).get("val_ba", None))

    return model


def load_all_experts(
    expert_names: list[str] | None = None,
    checkpoint_dir: str | None = None,
    device: str = "cpu",
) -> dict[str, torch.nn.Module]:
    """Load all expert models and return as a dict keyed by name.

    Args:
        expert_names: List like ['LAL', 'Mixup', 'PaCo']. Defaults to all available.
        checkpoint_dir: Override checkpoint directory.
    Returns:
        Dict mapping expert name → loaded model. Skips missing checkpoints with a warning.
    """
    if expert_names is None:
        expert_names = ["LAL", "Mixup", "PaCo", "CE", "BalancedSoftmax"]

    if checkpoint_dir is None:
        checkpoint_dir = str(DEFAULT_CHECKPOINT_DIR)

    models = {}
    for name in expert_names:
        ckpt_path = os.path.join(checkpoint_dir, f"{name}_best.pt")
        if not os.path.exists(ckpt_path):
            print(f"  [Warning] Checkpoint not found: {ckpt_path} — skipping {name}")
            continue
        models[name] = load_expert_checkpoint(name, ckpt_path, device)

    if not models:
        raise FileNotFoundError(
            f"No expert checkpoints found in {checkpoint_dir}. "
            f"Searched for: {expert_names}"
        )

    return models


# ── Data Loading ─────────────────────────────────────────────────────────


def create_cifar_loader(
    dataset_type: str = "train",
    data_root: str = "./data",
    batch_size: int = 128,
    shuffle: bool | None = None,
    num_workers: int = 2,
    pin_memory: bool = True,
    expert_name: str | None = None,
) -> tuple[DataLoader, np.ndarray]:
    """Create a DataLoader for CIFAR-100-LT splits.

    Args:
        dataset_type: 'train' | 'val' | 'test'
        data_root: Path to data directory (must contain ``processed/*.npy``).
        batch_size: Batch size.
        shuffle: Whether to shuffle. Auto-set for train/val/test if None.
        num_workers: DataLoader workers.
        pin_memory: Pin memory for GPU transfer.
        expert_name: Optional — if 'PaCo', returns two-view augmentations.
    Returns:
        (loader, class_counts_array)
    """
    root = Path(data_root)
    processed = root / "processed"

    if dataset_type == "test":
        # Original CIFAR-100 test set (10K balanced)
        dataset = LongTailCIFAR100(
            root=str(root),
            train=False,
            download=False,
            use_test_set=True,
        )
    elif dataset_type == "val":
        # LT validation set (held-out portion of LT indices)
        val_idx = np.load(str(processed / "lt_val_indices.npy"))
        dataset = LongTailCIFAR100(
            root=str(root),
            base_train_indices=val_idx,
            imbalance_ratio=100.0,
            train=False,
            download=False,
            already_subsampled=True,
        )
    else:
        # LT training set
        train_idx = np.load(str(processed / "lt_train_indices.npy"))
        dataset = LongTailCIFAR100(
            root=str(root),
            base_train_indices=train_idx,
            imbalance_ratio=100.0,
            train=(dataset_type == "train"),
            download=False,
            already_subsampled=True,
        )

    # Apply PaCo-specific transforms if needed
    if expert_name is not None and expert_name.upper() == "PACO" and dataset_type == "train":
        from scripts.train_paco import _augmentation_regular, _augmentation_sim_cifar
        dataset.transform = [_augmentation_regular, _augmentation_sim_cifar]
        dataset.two_view = True

    if shuffle is None:
        shuffle = (dataset_type == "train")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(pin_memory and torch.cuda.is_available()),
    )

    return loader, dataset.get_class_counts()


# ── Class Groups ─────────────────────────────────────────────────────────


def get_class_groups(
    class_counts: np.ndarray,
    many_thresh: int = 100,
    few_thresh: int = 20,
) -> dict[str, np.ndarray]:
    """Group class indices into head / medium / tail by sample count.

    Args:
        class_counts: Per-class sample count array, shape (100,).
        many_thresh: Classes with >= this many samples are 'Head'.
        few_thresh: Classes with <= this many samples are 'Tail'.
                    Classes in between are 'Medium'.
    Returns:
        Dict with keys 'Head', 'Med', 'Tail' mapping to arrays of class indices.
    """
    groups = {}
    groups["Head"] = np.where(class_counts >= many_thresh)[0]
    groups["Med"] = np.where((class_counts > few_thresh) & (class_counts < many_thresh))[0]
    groups["Tail"] = np.where(class_counts <= few_thresh)[0]
    return groups


def print_data_info(
    train_counts: np.ndarray,
    val_counts: np.ndarray | None = None,
    test_size: int | None = None,
) -> None:
    """Print a summary of the dataset splits."""
    print("=" * 55)
    print("DATA SPLIT SUMMARY")
    print("=" * 55)
    print(f"  LT Train: {train_counts.sum():,} samples "
          f"(head={train_counts[0]}, tail={train_counts[99]}, "
          f"IR={train_counts[0]/max(train_counts[99],1):.1f})")
    if val_counts is not None:
        print(f"  LT Val:   {val_counts.sum():,} samples "
              f"(head={val_counts[0]}, tail={val_counts[99]})")
    if test_size is not None:
        print(f"  Test:     {test_size:,} samples (balanced)")
    print()
