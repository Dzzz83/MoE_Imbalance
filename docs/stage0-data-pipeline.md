# Stage 0: Fix the Data Pipeline

## What's Wrong

Currently, the pipeline holds out 50 samples/class **before** applying long-tail subsampling. This creates a balanced validation set that doesn't exist in real long-tail scenarios, and reduces the training pool from 50K to 45K.

## What Needs to Change

**Old flow:**
```
CIFAR-100 train (50K, balanced)
  → Hold out 50/class (split_cifar100.py)
    → 5K balanced val indices  (balanced_val_indices.npy)
    → 45K base pool indices    (base_train_indices.npy)
      → LT subsampling inside LongTailCIFAR100
        → ~9,754 LT training samples
```

**New flow:**
```
CIFAR-100 train (50K, balanced)
  → LT subsampling on full 50K (create_lt_split.py)
    → ~10,847 LT indices
      → Shuffle & split 80/20
        → ~8,678 LT train indices  (lt_train_indices.npy)
        → ~2,169 LT val indices    (lt_val_indices.npy)

CIFAR-100 test (10K)  → used directly for final evaluation
```

---

## Files to Create

### 1. `utils/create_lt_split.py` — Pre-compute LT indices from full 50K

This script extracts the LT subsampling logic from `LongTailCIFAR100` and runs it on the full 50K training set, then splits into train/val.

```python
"""
Create proper CIFAR-100-LT train/val split.

Applies exponential subsampling (IR=100) to the full 50K training set,
then splits the resulting ~10,847 samples into train (80%) and val (20%).

Usage:
    python utils/create_lt_split.py
    # Creates: data/processed/lt_train_indices.npy
    #          data/processed/lt_val_indices.npy
    #          data/processed/lt_all_indices.npy
"""

import numpy as np
from pathlib import Path
from torchvision import datasets


def exponential_counts(n_max: int, n_classes: int, ir: float) -> np.ndarray:
    """Same logic as LongTailCIFAR100._exponential_counts."""
    indices = np.arange(n_classes, dtype=np.float64)
    exp = -indices / (n_classes - 1)
    raw = n_max * (ir ** exp)
    return np.maximum(raw.astype(np.int64), 1)


def create_lt_indices(
    data_root: str = "./data",
    ir: float = 100.0,
    val_split: float = 0.2,
    seed: int = 42,
):
    # Load full CIFAR-100 training set
    full = datasets.CIFAR100(root=data_root, train=True, download=True)
    targets = np.array(full.targets)
    
    # Compute per-class LT target counts
    n_per_class = np.array([(targets == c).sum() for c in range(100)])
    n_max = int(n_per_class.max())  # 500
    target_counts = exponential_counts(n_max, 100, ir)
    target_counts = np.minimum(target_counts, n_per_class)
    
    # Sample without replacement for each class
    rng = np.random.default_rng(seed)
    lt_indices = []
    for cls in range(100):
        cls_positions = np.where(targets == cls)[0]
        n_keep = int(target_counts[cls])
        sampled = rng.choice(cls_positions, size=n_keep, replace=False)
        lt_indices.extend(sampled.tolist())
    
    lt_indices = np.array(sorted(lt_indices), dtype=np.int64)
    # ~10,847 indices
    
    # Shuffle and split
    shuffled = lt_indices.copy()
    rng.shuffle(shuffled)
    split = int(len(shuffled) * (1 - val_split))
    train_idx = np.array(sorted(shuffled[:split]), dtype=np.int64)
    val_idx = np.array(sorted(shuffled[split:]), dtype=np.int64)
    
    # Save
    out_dir = Path(data_root) / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "lt_train_indices.npy", train_idx)
    np.save(out_dir / "lt_val_indices.npy", val_idx)
    np.save(out_dir / "lt_all_indices.npy", lt_indices)
    
    print(f"Created LT split from full 50K training set (IR={ir}):")
    print(f"  Total LT samples: {len(lt_indices)}")
    print(f"  Train: {len(train_idx)}")
    print(f"  Val:   {len(val_idx)}")
```

---

## Files to Modify

### 2. `data/cifar_lt.py` — Add `already_subsampled` flag

**Change:** Add a parameter that tells the dataset to skip LT subsampling when the indices are already pre-computed.

```python
class LongTailCIFAR100(Dataset):
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
        already_subsampled: bool = False,  # NEW
    ):
        ...
        # ── restrict to the base-training pool ──
        if base_train_indices is not None:
            self.images = all_images[base_train_indices]
            self.targets = all_targets[base_train_indices]
            self.base_indices = base_train_indices.copy()
        else:
            self.images = all_images
            self.targets = all_targets
            self.base_indices = np.arange(len(all_images))

        if skip_longtail:
            # Balanced set — keep ALL samples
            self.sample_images = self.images
            self.sample_targets = self.targets
            self.sample_indices = self.base_indices.copy()
        elif already_subsampled:          # NEW
            # Indices are already LT-subsampled — use directly
            self.sample_images = self.images
            self.sample_targets = self.targets
            self.sample_indices = self.base_indices.copy()
        else:
            # Original behavior: apply LT subsampling
            ...
```

**Why this works:** When `base_train_indices` contains the pre-computed LT indices (e.g., from `lt_train_indices.npy`), `self.images` and `self.targets` already have only ~8,678 samples. Setting `already_subsampled=True` skips the second subsampling step and uses them directly.

### 3. `data/cifar_lt.py` — Add test set support (optional but recommended)

Add a way to load the original CIFAR-100 test set (10K balanced). This can be done by adding a parameter:

```python
class LongTailCIFAR100(Dataset):
    def __init__(
        self,
        ...,
        use_test_set: bool = False,  # NEW: load CIFAR-100 test set
    ):
        ...
        if use_test_set:
            test = datasets.CIFAR100(root=root, train=False, download=download)
            self.images = test.data
            self.targets = np.array(test.targets)
            self.base_indices = np.arange(len(self.images))
            # Keep all, no subsampling
            self.sample_images = self.images
            self.sample_targets = self.targets
            self.sample_indices = self.base_indices.copy()
            # Test transforms
            self.transform = ...
            return
```

---

## How Training Scripts Will Change

### Before (old, flawed):
```python
# train_lal.py
base_idx = np.load('data/processed/base_train_indices.npy')    # 45K
val_idx = np.load('data/processed/balanced_val_indices.npy')   # 5K balanced

train_set = LongTailCIFAR100(
    root='./data',
    base_train_indices=base_idx,   # 45K → internally subsampled to ~9,754
    imbalance_ratio=100.0,
    train=True,
)
val_set = LongTailCIFAR100(
    root='./data',
    base_train_indices=val_idx,    # 5K balanced
    imbalance_ratio=100.0,
    train=False,
    skip_longtail=True,            # keep all 5K
)
```

### After (fixed):
```python
# train_lal.py
train_idx = np.load('data/processed/lt_train_indices.npy')    # ~8,678 (already LT)
val_idx = np.load('data/processed/lt_val_indices.npy')        # ~2,169 (already LT)

train_set = LongTailCIFAR100(
    root='./data',
    base_train_indices=train_idx,   # pre-computed LT indices
    train=True,
    already_subsampled=True,        # skip internal subsampling
)
val_set = LongTailCIFAR100(
    root='./data',
    base_train_indices=val_idx,     # pre-computed LT indices
    train=False,
    already_subsampled=True,        # skip internal subsampling
)
```

For test set evaluation:
```python
test_set = LongTailCIFAR100(
    root='./data',
    train=False,
    use_test_set=True,             # loads CIFAR-100 test set
)
```

---

## All Changes Summary

| File | Action | Lines Changed |
|:-----|:-------|:-------------|
| `utils/create_lt_split.py` | **Create** | ~60 lines |
| `data/cifar_lt.py` | **Modify** — add `already_subsampled`, `use_test_set` params | ~15 lines added |
| `utils/split_cifar100.py` | **Keep** (for reference, no longer used in training) | 0 |
| `scripts/train_lal.py` | **Modify** — load new indices, set `already_subsampled=True` | ~5 lines |
| `scripts/train_paco.py` | **Modify** — same | ~5 lines |
| `scripts/train_mixup.py` | **Modify** — same | ~5 lines |
| `scripts/evaluate_expert.py` | **Modify** — add test set option | ~5 lines |
| `data/processed/*` | **Replace** — remove old `base_train_indices.npy`, `balanced_val_indices.npy`; add new `lt_*` files | N/A |

---

## Verification Steps

After implementing, run these checks:

1. **Check LT distribution matches standard benchmark:**
```python
train_set = LongTailCIFAR100(root='./data',
    base_train_indices=np.load('data/processed/lt_train_indices.npy'),
    train=False, already_subsampled=True)
counts = train_set.get_class_counts()
print(f'Head: {counts[0]}, Tail: {counts[99]}, IR: {counts[0]/max(counts[99],1):.1f}')
# Expected: Head ~500, Tail ~5, IR ~100
```

2. **Check no overlap between train and val:**
```python
train_idx = np.load('data/processed/lt_train_indices.npy')
val_idx = np.load('data/processed/lt_val_indices.npy')
overlap = np.intersect1d(train_idx, val_idx)
print(f'Overlap: {len(overlap)}')  # Expected: 0
```

3. **Check test set loads correctly:**
```python
test_set = LongTailCIFAR100(root='./data', train=False, use_test_set=True)
print(f'Test set size: {len(test_set)}')  # Expected: 10,000
```

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|:-----|:-------|:-----------|
| Different random seed produces different LT split | Reproducibility | Use fixed seed (42) in `create_lt_split.py` |
| Training set is smaller (~8,678 vs ~9,754) | Expert BA may drop slightly | Acceptable — this is the standard protocol |
| Validation set is long-tailed (~2,169, few tail samples) | Noisy early stopping | Use smoothed BA or min loss for checkpoint selection |
| All existing routing scripts use old indices | Many scripts to update | Update systematically, one phase at a time |

---

## Order of Implementation

1. Create `utils/create_lt_split.py`
2. Run it to generate new indices
3. Modify `data/cifar_lt.py` (add `already_subsampled`, `use_test_set`)
4. Test loading with new indices
5. Update `scripts/train_lal.py` (minimal change)
6. Update `scripts/train_paco.py` (minimal change)
7. Update `scripts/train_mixup.py` (minimal change)
8. Update `scripts/evaluate_expert.py` (add test set support)
9. Verify everything works end-to-end
10. Remove old `base_train_indices.npy` and `balanced_val_indices.npy` (or archive them)

---

**Estimated effort:** ~2-3 hours of coding + testing. Most of the logic already exists — we're just moving the LT subsampling step from inside the dataset to a pre-computation script.
