"""
Stratified balanced validation split for CIFAR-100.

Loads the standard CIFAR-100 training set (50,000 samples, 500 per class),
reserves exactly 50 samples per class (5,000 total) as a balanced validation set,
and saves the indices of both splits for downstream use.

Usage:
    python utils/split_cifar100.py
"""

import numpy as np
import torchvision
from pathlib import Path


def main(
    data_root: str = "./data",
    val_per_class: int = 50,
    seed: int = 42,
) -> None:
    # ── reproducibility ──
    rng = np.random.default_rng(seed)

    # ── load full CIFAR-100 training set ──
    trainset = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=True
    )
    targets = np.array(trainset.targets)      # shape (50000,)
    n_classes = len(trainset.classes)          # 100
    n_total = len(targets)                     # 50000

    assert n_total == 50000, f"Expected 50000 training samples, got {n_total}"
    assert len(trainset.classes) == 100

    base_train_indices = []
    balanced_val_indices = []

    print(f"Splitting {n_total} samples into:")
    print(f"  - Balanced validation: {val_per_class} per class = {val_per_class * n_classes} total")
    print(f"  - Base training pool:  {n_total - val_per_class * n_classes} total")
    print()

    for cls in range(n_classes):
        # all indices belonging to this class
        cls_indices = np.where(targets == cls)[0]
        assert len(cls_indices) >= val_per_class, (
            f"Class {cls} has only {len(cls_indices)} samples, "
            f"cannot reserve {val_per_class}"
        )

        # shuffle and split
        shuffled = cls_indices.copy()
        rng.shuffle(shuffled)

        val_idx = shuffled[:val_per_class]
        train_idx = shuffled[val_per_class:]

        balanced_val_indices.extend(val_idx.tolist())
        base_train_indices.extend(train_idx.tolist())

    # convert to numpy arrays
    balanced_val_indices = np.array(sorted(balanced_val_indices), dtype=np.int64)
    base_train_indices = np.array(sorted(base_train_indices), dtype=np.int64)

    # ── validation ──
    val_targets = targets[balanced_val_indices]
    unique, counts = np.unique(val_targets, return_counts=True)
    print("=" * 50)
    print("VALIDATION SET — Class distribution")
    print("=" * 50)
    for c in range(n_classes):
        cnt = counts[unique == c][0] if c in unique else 0
        status = "✓" if cnt == val_per_class else "✗"
        print(f"  Class {c:3d} ({trainset.classes[c]:20s}): {cnt} samples {status}")
    print(f"\n  Total validation samples: {len(balanced_val_indices)}")
    assert np.all(counts == val_per_class), "Validation set is not balanced!"
    print("  ✓ Every class has exactly", val_per_class, "samples.")

    # ── save ──
    out_dir = Path(data_root) / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "balanced_val_indices.npy", balanced_val_indices)
    np.save(out_dir / "base_train_indices.npy", base_train_indices)
    np.save(out_dir / "val_targets.npy", val_targets)  # handy for quick checks

    print(f"\nSaved to {out_dir.resolve()}:")
    print(f"  - balanced_val_indices.npy   ({len(balanced_val_indices)} indices)")
    print(f"  - base_train_indices.npy     ({len(base_train_indices)} indices)")
    print(f"  - val_targets.npy            (labels for validation set)")
    print("Done.")


if __name__ == "__main__":
    main()
