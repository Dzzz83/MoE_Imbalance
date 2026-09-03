"""
Create proper CIFAR-100-LT train/val split (standard benchmark protocol).

Applies exponential subsampling (IR=100) to the FULL 50K CIFAR-100 training set
(no pre-holdout), then splits the resulting ~10,847 samples into train (80%)
and val (20%). Both splits follow the long-tailed distribution.

This replaces the old split_cifar100.py which held out 50/class as balanced
validation BEFORE creating the long tail — that was non-standard.

Usage:
    python utils/create_lt_split.py
    # Creates: data/processed/lt_train_indices.npy  (~8,678)
    #          data/processed/lt_val_indices.npy    (~2,169)
    #          data/processed/lt_all_indices.npy    (~10,847)
"""

import numpy as np
from pathlib import Path
from torchvision import datasets


def exponential_counts(n_max: int, n_classes: int, ir: float) -> np.ndarray:
    """
    Compute n_i = n_max * ir ^ (-i / (n_classes - 1)).

    Matches the standard long-tail CIFAR protocol used by:
      - LDAM (Cao et al., NeurIPS 2019)
      - RIDE (Wang et al., ICLR 2021)
      - PaCo (Cui et al., ICCV 2021)
      - LAL (Menon et al., ICLR 2021)

    Guarantees at least 1 sample per class via floor + max(1).
    """
    indices = np.arange(n_classes, dtype=np.float64)
    exp = -indices / (n_classes - 1)
    raw = n_max * (ir ** exp)
    return np.maximum(raw.astype(np.int64), 1)


def main(
    data_root: str = "./data",
    ir: float = 100.0,
    val_split: float = 0.2,
    seed: int = 42,
):
    # ── load full CIFAR-100 training set (50K, balanced) ──
    full = datasets.CIFAR100(root=data_root, train=True, download=True)
    targets = np.array(full.targets)
    n_classes = len(full.classes)
    n_total = len(targets)

    print(f"Full CIFAR-100 training set: {n_total} samples, {n_classes} classes")
    print(f"  Per class: {n_total // n_classes} samples (balanced)")
    print(f"  Target IR: {ir}")
    print()

    # ── compute per-class LT target counts ──
    n_per_class = np.array([(targets == c).sum() for c in range(n_classes)])
    n_max = int(n_per_class.max())
    target_counts = exponential_counts(n_max, n_classes, ir)
    target_counts = np.minimum(target_counts, n_per_class)

    print("Target per-class counts (first 10 / last 5):")
    for i in list(range(10)) + list(range(95, 100)):
        print(f"  Class {i:3d}: {target_counts[i]:3d} samples")
    print(f"  Total target: {target_counts.sum()}")
    print()

    # ── sample without replacement for each class ──
    rng = np.random.default_rng(seed)
    lt_indices: list[int] = []

    for cls in range(n_classes):
        cls_positions = np.where(targets == cls)[0]
        n_keep = int(target_counts[cls])
        sampled = rng.choice(cls_positions, size=n_keep, replace=False)
        lt_indices.extend(sampled.tolist())

    lt_indices = np.array(sorted(lt_indices), dtype=np.int64)

    # ── shuffle and split into train / val ──
    shuffled = lt_indices.copy()
    rng.shuffle(shuffled)
    split = int(len(shuffled) * (1 - val_split))
    train_idx = np.array(sorted(shuffled[:split]), dtype=np.int64)
    val_idx = np.array(sorted(shuffled[split:]), dtype=np.int64)

    # ── verify no overlap ──
    overlap = np.intersect1d(train_idx, val_idx)
    assert len(overlap) == 0, f"Train/val overlap: {len(overlap)} samples!"

    # ── verify LT distribution is preserved in both splits ──
    train_targets = targets[train_idx]
    val_targets = targets[val_idx]
    for name, tgt in [("Train", train_targets), ("Val", val_targets)]:
        counts = np.array([(tgt == c).sum() for c in range(n_classes)])
        actual_ir = counts.max() / max(counts.min(), 1)
        print(f"{name}: {len(tgt)} samples, head={counts[0]}, tail={counts[99]}, IR={actual_ir:.1f}")

    # ── save ──
    out_dir = Path(data_root) / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "lt_train_indices.npy", train_idx)
    np.save(out_dir / "lt_val_indices.npy", val_idx)
    np.save(out_dir / "lt_all_indices.npy", lt_indices)

    print()
    print(f"Saved to {out_dir.resolve()}:")
    print(f"  lt_train_indices.npy  ({len(train_idx)} indices) — for training experts")
    print(f"  lt_val_indices.npy    ({len(val_idx)} indices)   — for validation during training")
    print(f"  lt_all_indices.npy    ({len(lt_indices)} indices) — complete LT set")
    print()
    print("✅ Standard CIFAR-100-LT split created. No cheating.")
    print(f"   Train on the LT training set. Validate on the LT validation set.")
    print(f"   Evaluate FINAL results on the original CIFAR-100 test set (10K).")


if __name__ == "__main__":
    main()
