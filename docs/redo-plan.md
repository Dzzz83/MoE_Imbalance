# Redo Plan — Proper CIFAR-100-LT Benchmark

> **Status Update — Phase 0 Complete, Phases 1–6 Pending**
>
> Phase 0 (data pipeline fix) has been implemented. See `docs/stage0-data-pipeline.md` for
> implementation details and `docs/project-context.md` for the current project state.
> The remaining phases (retrain experts, re-establish baselines, re-run routing experiments)
> require GPU time on Kaggle.

---

## What Needs to Change

### Old (Flawed) Setup — ✅ Now Deprecated
```
CIFAR-100 train (50K)
  ├── 5K balanced validation ← held out BEFORE LT (NOT standard)
  └── 45K base pool
       └── LT subsampling → 9,754 training samples
```

### New (Proper) Setup — ✅ Implemented
```
CIFAR-100 train (50K)
  └── LT subsampling (IR=100) → ~10,847 training samples
       ├── ~8,678 training (80%)
       └── ~2,169 validation (20%) ← also long-tailed

CIFAR-100 test (10K) ← final evaluation only
```

---

## Phase 0: Recreate the Data Pipeline — ✅ COMPLETE

### 0.1 Fix `data/cifar_lt.py` — ✅ Done

**Changes made:**
- Added `already_subsampled: bool = False` parameter — when True, skips internal LT subsampling and uses the provided indices directly
- Added `use_test_set: bool = False` parameter — loads the original CIFAR-100 test set (10K balanced)
- All existing behavior preserved for backward compatibility (old training scripts still work with `skip_longtail=True`)

### 0.2 Create `utils/create_lt_split.py` — ✅ Done (replaces `utils/split_cifar100.py`)

**New file** creates the proper CIFAR-100-LT train/val split:
- Applies exponential subsampling (IR=100) to the **full 50K** training set
- Splits resulting ~10,847 samples into train (80%, ~8,678) and val (20%, ~2,169)
- Both splits follow the long-tailed distribution (no cheating)
- Fixed seed (42) for reproducibility
- Verifies no overlap between train and val indices
- Saves `lt_train_indices.npy`, `lt_val_indices.npy`, `lt_all_indices.npy`

The old `utils/split_cifar100.py` has been removed.

### 0.3 Files Affected — ✅ Done

| File | Change | Status |
|:-----|:-------|:-------|
| `data/cifar_lt.py` | Added `already_subsampled`, `use_test_set` params | ✅ |
| `utils/create_lt_split.py` | **New** — creates proper LT train/val split | ✅ |
| `utils/split_cifar100.py` | Removed (replaced by create_lt_split.py) | ✅ |
| `data/processed/lt_train_indices.npy` | **New** — ~8,678 LT training indices | ✅ |
| `data/processed/lt_val_indices.npy` | **New** — ~2,169 LT validation indices | ✅ |
| `data/processed/lt_all_indices.npy` | **New** — ~10,847 complete LT indices | ✅ |
| `data/processed/base_train_indices.npy` | Still present (deprecated) | ⏸️ |
| `data/processed/balanced_val_indices.npy` | Still present (deprecated) | ⏸️ |
| `scripts/utils/data.py` | Updated to use new indices | ✅ |
| `scripts/train.py` | Updated to use new data loading | ✅ |

---

## Phase 1: Retrain All Experts

### 1.1 Training Setup

All experts train on the **same** LT training set (~8,678 samples) and validate on the **LT validation set** (~2,169 samples, long-tailed).

| Expert | Loss | Epochs | Key Hyperparameters | Est. Time |
|:-------|:-----|:------:|:--------------------|:---------:|
| **LAL** | LAL (τ=1.0) | 200 | lr=0.1, batch=128 | 1.5h GPU |
| **PaCo** | PaCo (α=0.01, t=0.05) | 400 | lr=0.05, batch=256, K=2048 | 3h GPU |
| **Mixup** | CE + Mixup (α=1.0) | 200 | lr=0.1, batch=128 | 1.5h GPU |

**Important:** The validation set is now long-tailed. Early stopping and checkpoint selection will use BA on this LT validation set. The BA numbers will be LOWER than before because the validation set is imbalanced (tail classes have few samples → noisy BA estimates).

### 1.2 Expected Performance Change

| Expert | Old (balanced val) | New (LT val, estimated) | Notes |
|:-------|:------------------:|:-----------------------:|:------|
| LAL | 43.98% | ~38-42% | LT val has fewer tail samples → noisier BA |
| PaCo | 49.28% | ~43-47% | Same |
| Mixup | 40.80% | ~35-39% | Same |

The BA will be lower AND noisier because the validation set has very few tail-class samples (maybe 1-2 per class). This makes early stopping less reliable.

**Mitigation:** Use a smoothed BA (average over last N epochs) for checkpoint selection, or use the minimum loss epoch.

### 1.3 Files to Modify

| File | Change |
|:-----|:-------|
| `scripts/train_lal.py` | Point to new data indices; use LT validation set |
| `scripts/train_paco.py` | Same |
| `scripts/train_mixup.py` | Same |
| `scripts/base_trainer.py` | No changes needed (already supports any DataLoader) |

---

## Phase 2: Re-establish Baselines on the Test Set

### 2.1 Individual Expert Performance

Evaluate all three experts on the **CIFAR-100 test set** (10K, balanced). This gives the real numbers.

**Script:** `scripts/diversity_analysis.py` — modify to use test set loader.

### 2.2 Uniform Average Baseline

```
Uniform average BA on test set: ___%
```
This is the new baseline that all routing methods must beat.

### 2.3 Oracle Analysis

```
Oracle (at least one expert correct): ___%
Oracle gap: ___%
```

This tells us the routing headroom under the proper setup.

### 2.4 Diversity Metrics

| Pair | Cohen's κ | Class-acc correlation |
|:-----|:---------:|:---------------------:|
| LAL ↔ PaCo | ___ | ___ |
| LAL ↔ Mixup | ___ | ___ |
| PaCo ↔ Mixup | ___ | ___ |

### 2.5 Files to Create/Modify

| File | Change |
|:-----|:-------|
| `scripts/evaluate_on_test.py` | **New.** Evaluate any expert on the 10K test set |
| `scripts/diversity_analysis.py` | Add option to use test set instead of validation set |
| `scripts/base_trainer.py` | Add `_forward_for_eval` test-set evaluation helper |

---

## Phase 3: Rebuild Trust Meters & Error Analysis

### 3.1 Trust Meters (Correctness Prediction)

Train correctness-prediction trust meters on the **LT training set features**, evaluate on the **test set**.

**What to rebuild:**
| Trust Meter | Input Features | Training Data | Eval Data |
|:------------|:---------------|:--------------|:----------|
| Per-expert trust | 64-d backbone features | LT training set | Test set |
| 3-way classifier | 192-d concatenated features | LT training set | Test set |
| Calibrated confidence | Logits → temperature | LT training set | Test set |

### 3.2 Error Pattern Analysis

**Key metrics to recompute:**
- Feature learning gap (oracle-weighted routing vs learned gate)
- All-wrong ceiling (fraction where all 3 experts wrong)
- Lone Dissenter Paradox stats
- Label ambiguity (fraction with no unique best expert)
- Per-class error rates

### 3.3 Files to Create/Modify

| File | Change |
|:-----|:-------|
| `scripts/build_trust_meters.py` | **New.** Train trust meters on LT features |
| `scripts/analyze_errors.py` | **New.** Comprehensive error analysis on test set |
| `scripts/root_cause_analysis.py` | Update to use test set |

---

## Phase 4: Re-run All Routing Experiments

### 4.1 Routing Methods to Re-evaluate

All routing methods need to be re-run on the test set with the new experts. This is the biggest redo.

**Priority order** (start with the most promising, skip methods that were clearly inferior):

| # | Method | Old Result | Script | Priority |
|:-:|:--------|:----------:|:-------|:--------:|
| 1 | Uniform average | 51.12% | `diversity_analysis.py` | ✅ Critical (new baseline) |
| 2 | Optimal fixed weights | 52.58% | `gate_routing_diagnostic.py` | ✅ Critical |
| 3 | Product combination | ~52.4% | `gate_routing_diagnostic.py` | ✅ Critical |
| 4 | Calibrated confidence | 51.44% | `debug_routing.py` | ✅ Important |
| 5 | 89-d correctness routing | **52.42%** | `final_verify_89d.py` | ✅ Important (best method) |
| 6 | 92-d combined routing | **52.49%** | `multi_seed_92d_verify.py` | ✅ Important |
| 7 | Selective 92-d routing | **52.70%** | `selective_hybrid_routing.py` | ✅ Important |
| 8 | MLP router on 192-d features | 49.32% | `gate_routing_3seeds.py` | ⚠️ Optional |
| 9 | Confidence routing (raw) | 50.64% | `debug_routing.py` | ⚠️ Optional |
| 10 | Entropy routing | 50.10% | `debug_routing.py` | ❌ Skip (already inferior) |
| 11 | 3-way classifier | 26.25% | `deep_debug_routing.py` | ❌ Skip (already inferior) |
| 12 | Tournament soft (pairwise) | 52.10% | `pairwise_routing.py` | ⚠️ Optional |
| 13 | MLP pairwise | 50.66% | `pairwise_mlp_combined.py` | ❌ Skip |
| 14 | Meta-router (9-d) | 52.40% | `refined_routing_test.py` | ⚠️ Optional |
| 15 | Trust-weighted product | 52.43% | `gate_routing_diagnostic.py` | ⚠️ Optional |
| 16 | GDDR (gradient alignment) | 46.98% | `gradient_alignment_routing.py` | ❌ Skip |
| 17 | Cluster routing | 52.56% | `cluster_routing.py` | ⚠️ Optional |
| 18 | TTA routing variants | 53.00-53.34% | `tta_routing.py` etc. | ⚠️ Optional |
| 19 | 392-d hybrid TTA | 53.22% | `hybrid_tta_routing.py` | ⚠️ Optional |

**Total to re-run:** ~10-12 methods (prioritized).

### 4.2 Files to Modify

Each routing script needs to be updated to:
1. Load the new expert checkpoints (from Phase 1)
2. Use the test set for evaluation instead of the balanced validation set
3. Recompute all trust meters and routing features on the new data

---

## Phase 5: Verify All 12 Root Cause Problems

The 12 root cause problems documented in `docs/problem.md` need to be re-verified:

| # | Problem | Needs Re-verification? | Notes |
|:-:|:--------|:----------------------|:------|
| 1 | Feature learning gap (19.06%) | ✅ Yes | Depends on validation set |
| 2 | Expert miscalibration | ✅ Yes | Depends on validation set |
| 3 | 37.1% all-wrong ceiling | ✅ Yes | Depends on validation set |
| 4 | Insufficient training data | ❌ No | Ruled out via experiment |
| 5 | Distribution mismatch | ✅ Yes | Core issue being fixed |
| 6 | 3-way comparison problem | ✅ Yes | Depends on validation set |
| 7 | Lone Dissenter Paradox | ✅ Yes | Depends on validation set |
| 8 | Product captures most signal | ✅ Yes | Depends on validation set |
| 9 | 69.4% label ambiguity | ✅ Yes | Depends on validation set |
| 10 | Disagreement routing fails | ✅ Yes | Depends on validation set |
| 11 | Gradient orthogonality | ❌ No | Mathematical property |
| 12 | Feature clusters misaligned | ✅ Yes | Depends on validation set |

---

## Phase 6: Re-evaluate the Boosting Idea

After all baselines are re-established, we can assess whether the boosting approach is still worth pursuing.

### Key Questions to Answer

1. **Is the feature learning gap still 19%+ under the proper setup?** If the gap is smaller, the routing problem might be easier than we thought.

2. **Is the Lone Dissenter Paradox still 83.9%?** This depends on the expert calibration, which might change with the new training setup.

3. **Does the all-wrong ceiling change?** With different training data, the overlap of expert errors might be different.

4. **Is the 69.4% label ambiguity still present?** This is a fundamental property of the ensemble, not the evaluation set.

### Decision

If the core problems are confirmed under the proper setup → proceed with boosting (or another novel approach).
If the problems are less severe → the existing routing methods might already work better than we thought.

---

## Effort Summary

| Phase | Description | Scripts to Modify | GPU Hours | CPU Hours | Status |
|:-----:|:------------|:-----------------:|:---------:|:---------:|:------:|
| 0 | Fix data pipeline | 2 | 0 | 1 | ✅ Complete |
| 1 | Retrain 3 experts | 3 | 6 | 0 | ⏳ Pending |
| 2 | Re-establish baselines | 2 | 0.5 | 2 | ⏳ Pending |
| 3 | Rebuild trust meters | 2 | 1 | 2 | ⏳ Pending |
| 4 | Re-run ~12 routing methods | 10-12 | 2 | 4 | ⏳ Pending |
| 5 | Re-verify 12 root causes | 1 | 0.5 | 2 | ⏳ Pending |
| 6 | Re-evaluate boosting | 0 | 0 | 1 | ⏳ Pending |
| **Total** | | **~20** | **~10** | **~12** | **Phase 0 done** |

### Notes on Current Status

- **Phase 0** is fully implemented. The codebase now supports both the old (backward-compatible) and new data protocols.
- **Phases 1–6** require GPU access (Kaggle T4) to execute.
- The code infrastructure for Phases 2–4 is already in place: `scripts/train.py`, `scripts/evaluate.py`, `scripts/benchmark.py`, and `scripts/analyze.py` are all ready to use with the new data split.
- No additional scripting is needed for Phases 2–4 — just run the existing entry points with the new expert checkpoints.

---

## Open Questions — Mostly Resolved

1. **Do we keep the existing experts and just re-evaluate on the test set?** ❌ **No — must retrain from scratch.** The existing experts were trained on 9,754 samples from the 45K base pool. The proper setup trains on ~10,847 samples from the full 50K. The training distribution is different (~1,093 more samples, different class counts). Re-evaluating old experts on the test set would not give valid benchmark numbers.

2. **Do we re-run ALL 25+ routing methods, or just the most important ones?** ✅ **Use `scripts/benchmark.py` which runs all 9 routers automatically.** The benchmark script handles all routing methods in a single command. Individual experiment scripts from the old rounds can be consulted for specific details but the primary evaluation is through the benchmark.

3. **Do we fix the 10 code bugs first?** ✅ **Already done.** The data index bug, PaCo hyperparams, device placement, checkpoint overwrite, and all other infrastructure bugs were fixed during the original experiments. The refactored codebase incorporates all these fixes.

4. **What's the goal?** The primary goal is to obtain correct, reproducible numbers on the standard CIFAR-100-LT benchmark protocol. The refactored codebase supports this directly.
