"""
Data loading utilities for CIFAR-100-LT experiments.

Provides:
  - find_expert_checkpoint: resolve one run's final-epoch checkpoint
  - load_expert_checkpoint: load any expert model from a checkpoint
  - load_all_experts: load a pool of experts
  - create_cifar_loader: DataLoader over the canonical train/test splits
  - get_class_groups: Head/Medium/Tail class grouping (delegates to the one
    frozen definition in ``scripts.base_trainer``)

Protocol
--------
Splits come from :mod:`data.protocol_splits`, the single owner of the
long-tailed index artifact. There is **no validation split**: asking for one
raises. ``data/cifar_lt.py`` is the dataset; ``data/lt_datamodule.py`` is the
training loader; this module exists for the evaluation-side helpers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.cifar_lt import LongTailCIFAR100
from data.protocol_splits import ProtocolError, load_lt_train_indices
from models.resnet32 import ResNet32, PaCoResNet32

EPS = 1e-12

# Default checkpoint directory
DEFAULT_CHECKPOINT_DIR = Path(__file__).resolve().parent.parent.parent / "checkpoints"
DEFAULT_DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "data"

#: Suffix of the final-epoch checkpoint written by ``scripts/base_trainer.py``.
CHECKPOINT_SUFFIX = "_final.pt"

#: The experts of the canonical pool (see ``scripts/evaluate_experts.py``).
DEFAULT_EXPERTS: tuple[str, ...] = ("CE", "LAL", "BalancedSoftmax", "Mixup")


# ── Model Loading ────────────────────────────────────────────────────────


def _checkpoints_for(
    expert_name: str,
    checkpoint_dir: str | Path | None = None,
    seed: int | None = None,
) -> list[Path]:
    """Every final-epoch checkpoint matching one expert (and optionally a seed)."""
    directory = Path(checkpoint_dir) if checkpoint_dir else DEFAULT_CHECKPOINT_DIR
    pattern = (f"{expert_name}_seed{seed}{CHECKPOINT_SUFFIX}" if seed is not None
               else f"{expert_name}_seed*{CHECKPOINT_SUFFIX}")
    return sorted(directory.glob(pattern))


def find_expert_checkpoint(
    expert_name: str,
    checkpoint_dir: str | Path | None = None,
    seed: int | None = None,
) -> Path:
    """Resolve one expert run's final-epoch checkpoint.

    Checkpoints are named ``{expert}_seed{N}_final.pt`` by
    :class:`scripts.base_trainer.BaseTrainer`. A directory holding several seeds
    is an error, not a guess — the same rule ``ExpertPool`` enforces, so an
    evaluation can never silently mix runs.

    Raises:
        FileNotFoundError: when no run, or more than one, matches.
    """
    matches = _checkpoints_for(expert_name, checkpoint_dir, seed)
    directory = Path(checkpoint_dir) if checkpoint_dir else DEFAULT_CHECKPOINT_DIR
    if not matches:
        wanted = (f"{expert_name}_seed{seed}{CHECKPOINT_SUFFIX}" if seed is not None
                  else f"{expert_name}_seed<N>{CHECKPOINT_SUFFIX}")
        raise FileNotFoundError(
            f"no {expert_name} checkpoint in {directory} (looked for {wanted})"
        )
    if len(matches) > 1:
        seeds = sorted(m.name for m in matches)
        raise FileNotFoundError(
            f"{len(matches)} {expert_name} runs on disk ({seeds}); pass seed=... "
            f"so the evaluation reports a single run"
        )
    return matches[0]


def load_expert_checkpoint(
    expert_name: str,
    checkpoint_path: str | None = None,
    device: str = "cpu",
    seed: int | None = None,
    checkpoint_dir: str | Path | None = None,
) -> torch.nn.Module:
    """Load a trained expert model from checkpoint.

    Args:
        expert_name: 'CE', 'LAL', 'BalancedSoftmax', 'Mixup' (or retired 'PaCo').
        checkpoint_path: Path to a .pt file. When None, the run's
            ``{expert_name}_seed{seed}_final.pt`` is resolved from
            ``checkpoint_dir``.
        device: torch device string.
        seed: which run to load when the directory holds several.
        checkpoint_dir: checkpoint directory (defaults to ``./checkpoints``).
    Returns:
        Loaded model in eval mode on the specified device.
    """
    if checkpoint_path is None:
        checkpoint_path = str(find_expert_checkpoint(
            expert_name, checkpoint_dir=checkpoint_dir, seed=seed))

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if expert_name.upper() == "PACO":
        model = PaCoResNet32(num_classes=100, dim=32, K=1024)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model = ResNet32(num_classes=100)
        model.load_state_dict(ckpt["model_state_dict"])

    model = model.to(device)
    model.eval()

    # Optionally attach metadata for convenience
    model._expert_name = expert_name
    model._checkpoint_epoch = ckpt.get("epoch", "?")
    model._checkpoint_seed = ckpt.get("seed", seed)
    model._checkpoint_path = str(checkpoint_path)

    return model


def load_all_experts(
    expert_names: list[str] | None = None,
    checkpoint_dir: str | None = None,
    device: str = "cpu",
    seed: int | None = None,
) -> dict[str, torch.nn.Module]:
    """Load all expert models and return as a dict keyed by name.

    Args:
        expert_names: expert labels; defaults to the canonical pool.
        checkpoint_dir: override the checkpoint directory.
        device: torch device string.
        seed: which run to load for every expert.
    Returns:
        Dict mapping expert name -> loaded model. An expert with no checkpoint is
        skipped with a warning; an *ambiguous* expert (several seeds on disk while
        ``seed`` is None) raises, because that is not a missing run.
    """
    names = list(expert_names) if expert_names else list(DEFAULT_EXPERTS)
    directory = Path(checkpoint_dir) if checkpoint_dir else DEFAULT_CHECKPOINT_DIR

    models: dict[str, torch.nn.Module] = {}
    for name in names:
        matches = _checkpoints_for(name, directory, seed)
        if not matches:
            print(f"  [Warning] no checkpoint for {name} in {directory} — skipping")
            continue
        if len(matches) > 1:
            raise FileNotFoundError(
                f"{len(matches)} {name} runs on disk in {directory} "
                f"({[m.name for m in matches]}); pass seed=... to choose one"
            )
        models[name] = load_expert_checkpoint(name, str(matches[0]), device)

    if not models:
        raise FileNotFoundError(
            f"No expert checkpoints found in {directory}. Searched for: {names}"
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
    """Create a DataLoader over a canonical protocol split.

    Args:
        dataset_type: ``'train'`` or ``'test'``. The protocol has **no** validation
            split, so ``'val'`` raises :class:`~data.protocol_splits.ProtocolError`.
        data_root: Path to data directory (must contain ``processed/``).
        batch_size: Batch size.
        shuffle: Whether to shuffle. Auto-set per split when None.
        num_workers: DataLoader workers.
        pin_memory: Pin memory for GPU transfer.
        expert_name: Optional — if 'PaCo', returns two-view augmentations.
    Returns:
        (loader, class_counts_array)
    """
    root = Path(data_root)

    if dataset_type == "val":
        raise ProtocolError(
            "the CIFAR-100-LT protocol has no validation split: experts train on the "
            "full 10,847-sample long-tailed set and the final-epoch model is the "
            "reported model. Use 'train' or 'test'."
        )
    if dataset_type not in ("train", "test"):
        raise ValueError(
            f"dataset_type must be 'train' or 'test', got {dataset_type!r}"
        )

    if dataset_type == "test":
        # Original CIFAR-100 test set (10K balanced) — the only evaluation set
        dataset = LongTailCIFAR100(
            root=str(root),
            train=False,
            download=False,
            use_test_set=True,
        )
    else:
        # The canonical long-tailed training set, read through its one owner.
        train_idx = load_lt_train_indices(str(root))
        dataset = LongTailCIFAR100(
            root=str(root),
            base_train_indices=train_idx,
            imbalance_ratio=100.0,
            train=True,
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
    """Group class indices into Head / Med / Tail by sample count.

    A thin adapter over :func:`scripts.base_trainer.compute_class_groups`, which
    owns the frozen definition (AGENTs.md section 6): Head >= 100 training
    samples, Medium 20-100, Tail < 20 — so a class with exactly 20 samples is
    Medium. Only the key names differ (``Head``/``Med``/``Tail`` here), and that
    rename lives here so the thresholds exist in exactly one place.

    Args:
        class_counts: Per-class sample count array, shape (100,).
        many_thresh: Classes with >= this many samples are 'Head'.
        few_thresh: Classes with < this many samples are 'Tail'; the classes with
            exactly ``few_thresh`` samples are 'Med'.
    Returns:
        Dict with keys 'Head', 'Med', 'Tail' mapping to arrays of class indices.
    """
    from scripts.base_trainer import compute_class_groups

    canonical = compute_class_groups(
        np.asarray(class_counts), many_thresh=many_thresh, few_thresh=few_thresh)
    return {
        "Head": canonical["head"],
        "Med": canonical["medium"],
        "Tail": canonical["tail"],
    }


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
