# Stage 0: Fix the Data Pipeline — Implementation Report

> **Status: ✅ Complete**
>
> This document describes what was wrong with the original data pipeline and what
> was done to fix it. All changes have been implemented and verified.

---

## What Was Wrong

The original pipeline held out 50 samples/class **before** applying long-tail subsampling. This created a balanced validation set that doesn't exist in real long-tail scenarios, and reduced the training pool from 50K to 45K.

### Old (Flawed) Flow — Now Deprecated
```
CIFAR-100 train (50K, balanced)
  → Hold out 50/class (split_cifar100.py)
    → 5K balanced val indices  (balanced_val_indices.npy)
    → 45K base pool indices    (base_train_indices.npy)
      → LT subsampling inside LongTailCIFAR100
        → ~9,754 LT training samples
```

### New (Proper) Flow — ✅ Implemented
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

## What Was Created

### `utils/create_lt_split.py` — ✅ New file

Creates the proper CIFAR-100-LT train/val split from the full 50K training set:

- Extracts the exponential subsampling logic from `LongTailCIFAR100._exponential_counts()`
- Applies IR=100 subsampling to the **full 50K** (no pre-holdout)
- Splits the resulting ~10,847 samples into train (80%) and val (20%)
- Both splits follow the long-tailed distribution
- Fixed seed (42) for reproducibility
- Verifies no overlap between train and val
- Saves: `lt_train_indices.npy`, `lt_val_indices.npy`, `lt_all_indices.npy`

**Usage:** `python utils/create_lt_split.py`

---

## What Was Modified

### `data/cifar_lt.py` — ✅ Added two new parameters

**`already_subsampled: bool = False`**
- When `True`, skips internal LT subsampling and uses the provided indices directly
- Used with pre-computed LT indices from `create_lt_split.py`
- Example: `LongTailCIFAR100(..., base_train_indices=lt_train_indices, already_subsampled=True)`

**`use_test_set: bool = False`**
- When `True`, loads the original CIFAR-100 test set (10K, balanced)
- Overrides `base_train_indices` and `already_subsampled`
- Used for final evaluation
- Example: `LongTailCIFAR100(..., use_test_set=True)`

### `utils/split_cifar100.py` — ✅ Removed

Replaced by `utils/create_lt_split.py`. The old script created the flawed balanced validation split.

### `scripts/utils/data.py` — ✅ Updated

The shared `create_cifar_loader()` function now:
- Uses `lt_train_indices.npy` for training data
- Uses `lt_val_indices.npy` for validation data
- Uses `use_test_set=True` for test data
- Handles PaCo-specific two-view augmentations via `expert_name` parameter

### `scripts/train.py` — ✅ Updated

The unified training script uses the shared data loading from `scripts/utils/data.py`, so it automatically uses the proper data split. No per-trainer changes were needed.

---

## How Training Scripts Changed

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

### After (fixed) — via `scripts/train.py`:
```python
# No manual data loading needed — scripts/utils/data.py handles it:
train_loader, train_counts = create_cifar_loader('train', data_root)
val_loader, val_counts = create_cifar_loader('val', data_root)
```

### Test set evaluation:
```python
# scripts/evaluate.py --dataset test
test_loader, _ = create_cifar_loader('test', data_root)
```

---

## Verification Results

All verification steps pass:

1. **LT distribution matches standard benchmark:**
   - Head class (0): ~500 samples
   - Tail class (99): ~5 samples
   - IR: ~100:1

2. **No overlap between train and val:**
   - Overlap: 0 samples ✅

3. **Test set loads correctly:**
   - Test set size: 10,000 ✅

4. **Train set size:**
   - ~8,678 samples (from full 50K, no pre-holdout) ✅

---

## Files Created/Modified Summary

| File | Action | Status |
|:-----|:-------|:-------|
| `utils/create_lt_split.py` | **Create** — proper LT train/val split | ✅ Done |
| `data/cifar_lt.py` | **Modify** — add `already_subsampled`, `use_test_set` params | ✅ Done |
| `utils/split_cifar100.py` | **Remove** — replaced by create_lt_split.py | ✅ Done |
| `scripts/utils/data.py` | **Modify** — use new indices, add PaCo transforms | ✅ Done |
| `scripts/train.py` | **Modify** — uses shared data loading (automatic) | ✅ Done |
| `data/processed/lt_train_indices.npy` | **New** — ~8,678 LT training indices | ✅ Done |
| `data/processed/lt_val_indices.npy` | **New** — ~2,169 LT validation indices | ✅ Done |
| `data/processed/lt_all_indices.npy` | **New** — ~10,847 complete LT set | ✅ Done |
| `data/processed/base_train_indices.npy` | **Deprecated** — kept for reference | ⏸️ |
| `data/processed/balanced_val_indices.npy` | **Deprecated** — kept for reference | ⏸️ |

---

## What's Next

The data pipeline is fixed. The next steps are:

1. **Retrain all three experts** on the proper LT split using `scripts/train.py` (requires GPU)
2. **Evaluate** on the CIFAR-100 test set using `scripts/evaluate.py`
3. **Run routing benchmark** using `scripts/benchmark.py --dataset test`

See `docs/redo-plan.md` for the full redo plan and `docs/project-context.md` for the current project status.

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|:-----|:-------|:-----------|
| Different random seed produces different LT split | Reproducibility | Fixed seed (42) in `create_lt_split.py` |
| Training set is smaller (~8,678 vs ~9,754) | Expert BA may drop slightly | Acceptable — this is the standard protocol |
| Validation set is long-tailed (~2,169, few tail samples) | Noisy early stopping | Use smoothed BA or min loss for checkpoint selection |
| Old indices still in repo | Confusion | Kept for backward compatibility; clearly marked as deprecated |
