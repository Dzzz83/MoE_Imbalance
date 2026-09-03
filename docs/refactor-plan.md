# Scripts Refactor Plan — Implementation Status

> **Status: ✅ Mostly Complete**
>
> This document originally described a plan to refactor 40+ ad-hoc scripts into a clean,
> maintainable framework. That plan has been largely executed. This file now documents
> what was implemented, what changed from the original plan, and what remains.

---

## Current State (Post-Refactor)

### ✅ What Was Built

| Category | Before | After | Status |
|:---------|:------:|:-----:|:------:|
| Training scripts | 5 standalone | `scripts/train.py --method {lal, mixup, paco, ce, balanced_softmax}` | ✅ |
| Evaluation scripts | 1 standalone | `scripts/evaluate.py --expert LAL --dataset {train, val, test}` | ✅ |
| Benchmarking | ~23 routing scripts | `scripts/benchmark.py` + 9 OOP routers in `scripts/router/` | ✅ |
| Analysis scripts | 7 overlapping | `scripts/analyze.py --mode {diversity, root_cause, calibration, all}` | ✅ |
| Shared utilities | Duplicated everywhere | `scripts/utils/{data, metrics, features}.py` | ✅ |
| Weighted training | Not available | `scripts/train_lal_weighted.py`, `scripts/train_paco_weighted.py`, `data/weighted_dataset.py` | ✅ |

### Actual Structure vs Original Plan

The original plan proposed a target structure. Here's what was actually implemented:

```
scripts/
  __init__.py
  train.py                 ← UNIFIED entry point (as planned)
  evaluate.py              ← UNIFIED entry point (as planned)
  benchmark.py             ← UNIFIED entry point (as planned)
  analyze.py               ← UNIFIED entry point (as planned)
  base_trainer.py          ← KEPT (abstract training base)
  train_lal.py             ← KEPT (individual trainer)
  train_mixup.py           ← KEPT (individual trainer)
  train_paco.py            ← KEPT (individual trainer)
  train_lal_weighted.py    ← NEW (boosting support)
  train_paco_weighted.py   ← NEW (boosting support)
  train_ce.py              ← KEPT (legacy)
  train_balanced_softmax.py ← KEPT (legacy)

  utils/                   ← NEW (shared utilities)
    __init__.py
    data.py                ← Data loading, model loading, class groups
    metrics.py             ← BA, per-class acc, group acc, ECE, routing metrics
    features.py            ← Logit extraction, 24-d/89-d/92-d features, PCA

  router/                  ← NEW (OOP router framework)
    __init__.py            ← Router registry
    base.py                ← BaseRouter (abstract)
    uniform.py             ← UniformRouter
    confidence.py          ← ConfidenceRouter (calibrated)
    product.py             ← ProductRouter (geometric mean)
    correctness.py         ← CorrectnessRouter (trust meters)
    pairwise.py            ← PairwiseRouter (tournament)
    cluster.py             ← ClusterRouter (per-cluster weights)
    gate.py                ← GateRouter (learned MLP gate)
    tta.py                 ← TTARouter (test-time augmentation)
    selective.py           ← SelectiveRouter (abstain on low confidence)
```

### Differences from Original Plan

| Planned | Actual | Reason |
|:--------|:-------|:-------|
| Remove all individual trainers | Kept `train_lal.py`, `train_mixup.py`, `train_paco.py` | `train.py` dispatches to them internally; individual trainers still needed for custom logic |
| Create `scripts/router/hybrid.py` | Not created | Hybrid routing (e.g., TTA + 92-d) is handled by combining features in `benchmark.py` |
| `scripts/router/tta.py` as separate | Created as planned | TTA has specific input preprocessing needs |
| Remove all old routing scripts | Old scripts still in repo | Preserved for reference; can be archived later |
| `utils/split_cifar100.py` → keep | `utils/split_cifar100.py` removed, replaced by `utils/create_lt_split.py` | The old split script was the source of the data leakage |

---

## What Changed from the Original Plan During Implementation

### 1. Data Loading Simplification

The original plan proposed `create_data_loader(dataset_type, data_root, batch_size)`. The actual implementation (`scripts/utils/data.py`) uses `create_cifar_loader(dataset_type, data_root, batch_size, expert_name)` — the `expert_name` parameter was added to handle PaCo's two-view augmentations, which wasn't anticipated in the original plan.

### 2. Router Framework API Refinements

The original `BaseRouter` interface specified:
```python
def train(val_logits, val_labels, val_features) → Self
def predict(logits, features) → np.ndarray
```

The actual implementation adds:
- `predict_proba(logits, features)` — returns routing weights per expert (for soft routing)
- `predict_class(logits, features)` — returns predicted class labels (for uniform/product routers that combine before argmax)
- `evaluate(logits, labels, class_counts, features)` — comprehensive evaluation with head/med/tail breakdown

### 3. Feature Extraction Consolidation

The original plan proposed separate `extract_logits()`, `extract_24d_features()`, `extract_89d_features()`, `extract_92d_features()` functions. The actual implementation (`scripts/utils/features.py`) has all of these plus:
- `extract_all_experts()` — runs all models on a DataLoader and collects logits + features in one pass
- `compute_energy()` — energy score for logits
- `softmax()` — stable NumPy softmax

### 4. Benchmarking Instead of Individual Scripts

The original plan proposed removing all 23 routing scripts. In practice, the old scripts remain in the repo for reference, but all routing methods are now accessible through `scripts/benchmark.py`. The benchmark script:
- Loads all expert models
- Extracts features from validation and evaluation sets
- Runs each router's `train()` and `evaluate()` methods
- Produces a formatted comparison table
- Optionally saves results to JSON

---

## Remaining Work

### Phase 4: Remove Obsolete Files (Optional)

The old routing scripts are still in the repo. They can be archived or removed once the new framework is verified to produce identical results on the proper data split. The obsolete scripts include:

| File | Superseded By | Priority |
|:-----|:--------------|:---------|
| `correctness_routing.py` | `scripts/router/correctness.py` + `benchmark.py` | Low |
| `debug_routing*.py` | `scripts/analyze.py --mode root_cause` | Low |
| `deep_debug_routing.py` | `scripts/analyze.py` | Low |
| `gate_routing_*.py` | `scripts/router/gate.py` | Low |
| `gradient_*routing.py` | `scripts/benchmark.py` (as variants) | Low |
| `cluster_routing.py` | `scripts/router/cluster.py` | Low |
| `pairwise_*.py` | `scripts/router/pairwise.py` | Low |
| `tta_routing.py`, `hybrid_tta_routing.py` | `scripts/router/tta.py` | Low |
| `selective_hybrid_routing.py` | `scripts/router/selective.py` | Low |
| `eval_router*.py` | `scripts/benchmark.py` | Low |
| `verify_*.py`, `final_*.py` | `scripts/benchmark.py` | Low |
| `novel_routing_test.py`, `refined_routing_test.py` | `scripts/benchmark.py` | Low |
| `mock_test.py` | Obsolete test file | Low |
| `rotation_routing.py` | `scripts/router/tta.py` | Low |
| `root_cause_light.py`, `diagnose_loss_oracle.py` | `scripts/analyze.py --mode root_cause` | Low |
| `kaggle_root_cause.py` | `scripts/analyze.py` (Kaggle-specific) | Low |
| `augmentation_consistency_analysis.py` | `scripts/analyze.py --mode augmentation` | Low |
| `per_class_calibration.py` | `scripts/analyze.py --mode calibration` | Low |
| `root_cause_analysis.py` | `scripts/analyze.py --mode root_cause` | Low |

### Verification Needed

1. **Equivalence testing** — Verify that the refactored routers produce the same results as the original scripts on the same data split
2. **Multi-seed testing** — Run `benchmark.py` with multiple seeds to verify statistical properties
3. **New expert training** — Use `train.py` to train experts on the proper LT split (requires GPU)

---

## Summary

| Metric | Before | After |
|:-------|:------:|:-----:|
| Total scripts | 40+ | ~15 core + 9 router modules |
| Routing scripts | 23 standalone | 9 OOP routers + 1 benchmark |
| Lines of routing code | ~12,000 | ~2,500 |
| Duplicated data loading | 25 files | 1 utility function |
| Duplicated BA computation | 37 files | 1 utility function |
| Time to add a new router | Copy 500-line script | Subclass BaseRouter (~30 lines) |
| Data pipeline | Flawed (balanced val) | Fixed (standard CIFAR-100-LT) |
