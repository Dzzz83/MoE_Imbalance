"""
Build the canonical CIFAR-100-LT training index set.

Protocol produced here
----------------------
    CIFAR-100 train (50,000, balanced)
      └── apply imbalance factor 0.01 (IR=100) per class  ->  10,847 samples
           └── ALL 10,847 samples ARE the training set
    CIFAR-100 test (10,000, balanced) is the ONLY evaluation set.

There is deliberately **no validation split**. Experts train on the full
long-tailed training set and are evaluated on the balanced test set, which is
the standard CIFAR-100-LT protocol used by LDAM (Cao et al., NeurIPS 2019),
RIDE (Wang et al., ICLR 2021) and PaCo (Cui et al., ICCV 2021).

The per-class count follows

    n_i = n_max * IR ** (-i / (C - 1))

with i = 0 the head class and i = C-1 the tail class. For C=100, n_max=500 and
IR=100 this yields head=500, tail=5, total=10,847.

Reproducibility
---------------
The artifact is a pure function of (source targets, imbalance factor, seed), so
``data/processed/`` does not need to be version-controlled: regenerating with
the same seed reproduces the index array exactly.

Usage:
    python utils/create_lt_split.py                        # IR=100, seed 42
    python utils/create_lt_split.py --imb-factor 0.01 --seed 42
    python utils/create_lt_split.py --output my_indices.npy
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Allow `python utils/create_lt_split.py` from the project root (same bootstrap
# pattern used by scripts/*.py).
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)

from data.protocol_splits import LT_TRAIN_FILENAME  # noqa: E402

#: Imbalance factor = min_class_count / max_class_count (Cui et al. convention).
IMBALANCE_FACTOR_DEFAULT: float = 0.01
#: Imbalance ratio = max_class_count / min_class_count.
IR_DEFAULT: float = 100.0
#: Fixed across the whole project so every comparison shares one LT profile.
SEED_DEFAULT: int = 42


def imbalance_factor_to_ir(imbalance_factor: float) -> float:
    """Convert the `imb_factor` convention to the imbalance ratio.

    ``imb_factor`` is min/max class count (0.01 means 100:1), so the ratio is
    its reciprocal. Kept as a named function because the two conventions are
    used interchangeably in the literature and confusing them silently changes
    the dataset.
    """
    if not 0.0 < imbalance_factor <= 1.0:
        raise ValueError(
            f"imbalance_factor must be in (0, 1], got {imbalance_factor!r}"
        )
    return 1.0 / imbalance_factor


class LongTailProfile:
    """The per-class sample count of an exponentially long-tailed dataset."""

    def __init__(self, n_classes: int, n_max: int, imbalance_ratio: float) -> None:
        if n_classes < 2:
            raise ValueError(f"n_classes must be >= 2, got {n_classes}")
        if n_max < 1:
            raise ValueError(f"n_max must be >= 1, got {n_max}")
        if imbalance_ratio < 1.0:
            raise ValueError(f"imbalance_ratio must be >= 1, got {imbalance_ratio}")
        self.n_classes = n_classes
        self.n_max = n_max
        self.imbalance_ratio = float(imbalance_ratio)

    def counts(self) -> np.ndarray:
        """Per-class target counts, ``n_i = n_max * IR ** (-i / (C-1))``.

        Truncated to int (floor for positives) and floored at 1 so that no class
        is ever empty — matching the reference implementations.
        """
        i = np.arange(self.n_classes, dtype=np.float64)
        raw = self.n_max * (self.imbalance_ratio ** (-i / (self.n_classes - 1)))
        return np.maximum(raw.astype(np.int64), 1)

    @property
    def total(self) -> int:
        return int(self.counts().sum())

    def describe(self) -> str:
        c = self.counts()
        return (
            f"{len(c)} classes, total={int(c.sum())}, head={int(c[0])}, "
            f"tail={int(c[-1])}, IR={c[0] / max(int(c[-1]), 1):.1f}"
        )


class LongTailIndexBuilder:
    """Samples which CIFAR-100 training images survive the long-tail subsample."""

    def __init__(self, targets: np.ndarray, imbalance_ratio: float = IR_DEFAULT) -> None:
        targets = np.asarray(targets)
        if targets.ndim != 1:
            raise ValueError(f"targets must be 1-D, got shape {targets.shape}")
        self.targets = targets
        self.n_classes = int(targets.max()) + 1
        self.imbalance_ratio = float(imbalance_ratio)
        self.available = np.bincount(targets, minlength=self.n_classes)
        self.profile = LongTailProfile(
            n_classes=self.n_classes,
            n_max=int(self.available.max()),
            imbalance_ratio=self.imbalance_ratio,
        )

    def target_counts(self) -> np.ndarray:
        """Wanted per-class counts, capped by what the source actually holds."""
        return np.minimum(self.profile.counts(), self.available)

    def build(self, seed: int = SEED_DEFAULT) -> np.ndarray:
        """Return the sorted, unique training indices of the long-tailed set.

        Classes are sampled in ascending class order from a single seeded RNG,
        so the result is deterministic for a given (targets, IR, seed).
        """
        counts = self.target_counts()
        rng = np.random.default_rng(seed)
        chosen: list[int] = []

        for cls in range(self.n_classes):
            cls_positions = np.where(self.targets == cls)[0]
            n_keep = int(counts[cls])
            if n_keep > len(cls_positions):
                raise ValueError(
                    f"class {cls}: want {n_keep} samples but only "
                    f"{len(cls_positions)} available"
                )
            sampled = rng.choice(cls_positions, size=n_keep, replace=False)
            chosen.extend(int(x) for x in sampled)

        return np.array(sorted(chosen), dtype=np.int64)


def build_lt_indices(
    targets: np.ndarray,
    imbalance_ratio: float = IR_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> np.ndarray:
    """Convenience wrapper around :class:`LongTailIndexBuilder`."""
    return LongTailIndexBuilder(targets, imbalance_ratio).build(seed)


def _load_train_targets(data_root: str, download: bool = False) -> np.ndarray:
    from torchvision import datasets

    try:
        full = datasets.CIFAR100(root=data_root, train=True, download=download)
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            f"Cannot load CIFAR-100 train split from '{data_root}': {exc}"
        ) from exc
    return np.asarray(full.targets, dtype=np.int64)


def main(
    data_root: str = './data',
    imbalance_factor: float = IMBALANCE_FACTOR_DEFAULT,
    seed: int = SEED_DEFAULT,
    output_name: str = LT_TRAIN_FILENAME,
    download: bool = False,
) -> np.ndarray:
    """Build and save the canonical long-tailed training index set."""
    from pathlib import Path

    ir = imbalance_factor_to_ir(imbalance_factor)
    targets = _load_train_targets(data_root, download=download)

    n_per_class = np.bincount(targets)
    print(f"Source: CIFAR-100 train, {len(targets)} samples, "
          f"{len(n_per_class)} classes, {n_per_class.min()}-{n_per_class.max()} per class")
    print(f"Imbalance factor {imbalance_factor} -> IR {ir:g}")

    builder = LongTailIndexBuilder(targets, ir)
    print(f"Target profile: {builder.profile.describe()}")

    indices = builder.build(seed)

    # ── verify the realized profile before writing anything ──
    realized = np.bincount(targets[indices], minlength=builder.n_classes)
    expected = builder.target_counts()
    if not np.array_equal(realized, expected):
        raise AssertionError(
            "realized per-class counts differ from the target profile:\n"
            f"  realized={realized[:5]}...{realized[-5:]}\n"
            f"  expected={expected[:5]}...{expected[-5:]}"
        )
    if realized.min() < 1:
        raise AssertionError(
            f"{(realized == 0).sum()} classes ended up with zero samples — "
            f"the long-tailed profile must cover every class"
        )
    if len(np.unique(indices)) != len(indices):
        raise AssertionError("duplicate indices produced")

    out_dir = Path(data_root) / 'processed'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / output_name
    np.save(out_path, indices)

    print()
    print(f"Wrote {out_path.resolve()}")
    print(f"  {len(indices)} training indices, head={realized[0]}, "
          f"tail={realized[-1]}, IR={realized.max() / realized.min():.1f}")
    print()
    print("✅ Canonical CIFAR-100-LT training set created (no validation split).")
    print("   Train on these indices. Evaluate FINAL results on the CIFAR-100 test set (10K).")
    return indices


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default='./data')
    parser.add_argument(
        '--imb-factor', type=float, default=IMBALANCE_FACTOR_DEFAULT,
        help=f'min/max class-count ratio (default {IMBALANCE_FACTOR_DEFAULT} = IR 100)',
    )
    parser.add_argument('--seed', type=int, default=SEED_DEFAULT)
    parser.add_argument('--output', default=LT_TRAIN_FILENAME)
    parser.add_argument(
        '--download', action='store_true',
        help='allow torchvision to download CIFAR-100 if absent',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(
        data_root=args.data_root,
        imbalance_factor=args.imb_factor,
        seed=args.seed,
        output_name=args.output,
        download=args.download,
    )
