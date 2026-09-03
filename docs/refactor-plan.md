# Scripts Refactor Plan

## Current State — 40 Scripts, Massive Duplication

| Category | Count | Problem |
|:---------|:-----:|:--------|
| Training | 5 | 2 weighted variants could be merged with flags |
| Base | 1 | OK — keep |
| Evaluation | 1 | OK — keep |
| **Routing** | **23** | **Same boilerplate copied 23× — data loading, model loading, metrics** |
| Router Eval | 3 | Same thing, 3 iterations |
| Analysis | 7 | Overlapping analyses |
| Other | 1 | `mock_test.py` — obsolete |

**Key numbers:**
- 39/40 files import `base_trainer` but most just copy its patterns
- 25 files have their own data loading logic (duplicated)
- 37 files have their own BA computation (duplicated)
- 23 routing scripts average ~500 lines each = **~12,000 lines of duplicated routing code**

---

## Target Structure

```
scripts/
  __init__.py
  
  # ── Core framework ──
  base_trainer.py          ← KEEP (abstract training base)
  
  # ── Utilities (shared code, no duplication) ──
  utils/
    __init__.py
    data.py                ← Data loading (train/val/test loaders)
    metrics.py             ← BA, per-class acc, head/med/tail grouping
    features.py            ← Feature extraction (24-d, 89-d, 92-d, logits)
  
  # ── Router framework (OOP) ──
  router/
    __init__.py
    base.py                ← Abstract BaseRouter
    confidence.py          ← Confidence-based routing
    correctness.py         ← Correctness-prediction routing
    product.py             ← Product-of-experts routing
    pairwise.py            ← Pairwise tournament + MLP routing
    cluster.py             ← Cluster-based routing
    gate.py                ← Learned gate routing (MLP, calibrated)
    tta.py                 ← Test-time augmentation routing
    hybrid.py              ← Hybrid / composite routing
    selective.py           ← Selective routing (abstain on hard samples)
  
  # ── Entry points (thin, minimal) ──
  train.py                 ← Unified: --method {lal, mixup, paco, weighted}
  evaluate.py              ← Unified: --expert LAL --dataset {train, val, test}
  benchmark.py             ← Run all routers, produce comparison table
  analyze.py               ← --mode {diversity, root_cause, calibration}
```

---

## Refactor Phases

### Phase 1: Extract Shared Utilities

**What:** Move duplicated boilerplate into `scripts/utils/`.

**`utils/data.py`** — Data loading:
```python
def load_expert_checkpoint(expert_name, checkpoint_path, device):
    """Load any expert (LAL, Mixup, PaCo) from checkpoint."""
    ...

def create_data_loader(dataset_type='train', data_root='./data', batch_size=128):
    """Return DataLoader for train/val/test."""
    ...

def get_class_groups(class_counts):
    """Return head/medium/tail class indices based on counts."""
    ...
```

**`utils/metrics.py`** — Evaluation metrics:
```python
def balanced_accuracy(predictions, targets, num_classes=100):
    ...

def per_class_accuracy(predictions, targets, num_classes=100):
    ...

def group_accuracies(predictions, targets, class_counts, many_thresh=100, few_thresh=20):
    ...

def confidence_calibration(confidences, correct, num_bins=15):
    """ECE, MCE for calibration analysis."""
    ...
```

**`utils/features.py`** — Feature extraction:
```python
def extract_logits(model, loader, device):
    """Run model on loader, return all logits."""
    ...

def extract_24d_features(logits_list):
    """24-d feature vector per sample: 3×{max_logit, max2nd-max, confidence, margin, ...}"""
    ...

def extract_89d_features(logits_list, class_labels):
    """89-d: 3×{confidence} + 3×{error_binary} + 1×{true_label} + ..."""
    ...

def extract_92d_features(logits_list, class_labels):
    """92-d: 89-d + 3×{margin}"""
    ...
```

**Files to create:** `scripts/utils/__init__.py`, `data.py`, `metrics.py`, `features.py`

---

### Phase 2: Build Router Framework

**What:** Define `BaseRouter` abstract class, implement each routing method as a subclass.

**`router/base.py`** — Abstract base:
```python
class BaseRouter(ABC):
    """Abstract base for all routing methods."""

    def __init__(self, expert_names: list[str]):
        self.expert_names = expert_names

    @abstractmethod
    def train(self, val_logits: np.ndarray, val_labels: np.ndarray,
              val_features: dict | None = None) -> Self:
        """Train router on validation data (if needed)."""
        ...

    @abstractmethod
    def predict(self, logits: np.ndarray, features: dict | None = None) -> np.ndarray:
        """Return expert_index for each sample (argmax over experts)."""
        ...

    def evaluate(self, logits: np.ndarray, labels: np.ndarray,
                 features: dict | None = None) -> dict:
        """Evaluate routing performance. Returns {ba, head_acc, med_acc, tail_acc, ...}."""
        predictions = self.predict(logits, features)
        return compute_routing_metrics(predictions, labels, logits)
```

**`router/confidence.py`** — Confidence-based routing:
```python
class ConfidenceRouter(BaseRouter):
    def train(self, val_logits, val_labels, **kwargs):
        # Compute per-expert temperature scaling on val set
        return self
    def predict(self, logits, **kwargs):
        # Pick expert with highest calibrated confidence
        ...
```

**`router/correctness.py`** — Correctness prediction:
```python
class CorrectnessRouter(BaseRouter):
    def train(self, val_logits, val_labels, val_features):
        # Train per-expert trust meters (MLP on features → correctness prob)
        return self
    def predict(self, logits, features):
        # Pick expert with highest predicted correctness
        ...
```

**`router/product.py`** — Product of experts:
```python
class ProductRouter(BaseRouter):
    def predict(self, logits, **kwargs):
        # Combine via soft product: softmax per expert, multiply, argmax
        ...
```

**`router/pairwise.py`** — Pairwise tournament + MLP:
```python
class PairwiseRouter(BaseRouter):
    def train(self, val_logits, val_labels, **kwargs):
        # Learn per-pair preference weights
        return self
    def predict(self, logits, **kwargs):
        # Run tournament: each pair votes, pick winner
        ...

class PairwiseMLPRouter(BaseRouter):
    def train(self, val_logits, val_labels, val_features):
        # Train MLP per pair: (features_A, features_B) → preference
        return self
    def predict(self, logits, features):
        ...
```

**`router/cluster.py`** — Cluster-based routing:
```python
class ClusterRouter(BaseRouter):
    def train(self, val_logits, val_labels, val_features):
        # Cluster feature space, learn per-cluster expert preferences
        return self
    def predict(self, logits, features):
        # Assign to cluster, use cluster's preferred expert
        ...
```

**`router/gate.py`** — Learned gate:
```python
class GateRouter(BaseRouter):
    def train(self, val_logits, val_labels, val_features):
        # Train MLP gate: features → expert weights
        return self
    def predict(self, logits, features):
        # Gate weights → weighted combination → argmax
        ...
```

**`router/tta.py`** — Test-time augmentation:
```python
class TTARouter(BaseRouter):
    def predict(self, logits, **kwargs):
        # Average over TTA views, pick expert with most consistent prediction
        ...
```

**`router/hybrid.py`** — Hybrid composite:
```python
class HybridRouter(BaseRouter):
    def __init__(self, routers: list[BaseRouter], meta_learner=None):
        # Combine multiple routers
        ...
```

**`router/selective.py`** — Selective / abstention:
```python
class SelectiveRouter(BaseRouter):
    def predict(self, logits, confidence_threshold=0.5):
        # Route only when confidence > threshold; abstain otherwise
        ...
```

**Files to create:** `scripts/router/__init__.py`, `base.py`, `confidence.py`, `correctness.py`, `product.py`, `pairwise.py`, `cluster.py`, `gate.py`, `tta.py`, `hybrid.py`, `selective.py`

---

### Phase 3: Consolidate Entry Points

**What:** Thin scripts that use the framework above.

**`train.py`** — Unified training:
```python
# Usage:
#   python scripts/train.py --method lal
#   python scripts/train.py --method paco --epochs 400
#   python scripts/train.py --method lal_weighted --weight-source mixup_confidence

parser.add_argument('--method', choices=['lal', 'mixup', 'paco',
                                         'lal_weighted', 'paco_weighted'])
# Routes to the appropriate trainer, shares data loading from utils.data
```

**`evaluate.py`** — Unified evaluation:
```python
# Usage:
#   python scripts/evaluate.py --expert LAL --dataset test
#   python scripts/evaluate.py --expert PaCo --dataset train --save-logits

# Uses utils.data for loader, utils.metrics for BA
```

**`benchmark.py`** — Run all routers:
```python
# Usage:
#   python scripts/benchmark.py --routers all --dataset test

routers = {
    'uniform': UniformRouter(experts),
    'confidence': ConfidenceRouter(experts),
    'product': ProductRouter(experts),
    'correctness': CorrectnessRouter(experts),
    'pairwise': PairwiseRouter(experts),
    'cluster': ClusterRouter(experts),
    'gate': GateRouter(experts),
    'tta': TTARouter(experts),
    'selective': SelectiveRouter(experts),
}

results = {}
for name, router in routers.items():
    router.train(val_logits, val_labels, val_features)
    results[name] = router.evaluate(test_logits, test_labels, test_features)

# Print comparison table: method | BA | head | med | tail | oracle_gap
```

**`analyze.py`** — Unified analysis:
```python
# Usage:
#   python scripts/analyze.py --mode diversity
#   python scripts/analyze.py --mode root_cause
#   python scripts/analyze.py --mode calibration

# Each mode calls the appropriate analysis from utils + router framework
```

**Files to create/modify:** `scripts/train.py`, `scripts/evaluate.py`, `scripts/benchmark.py`, `scripts/analyze.py`

---

### Phase 4: Remove Obsolete Files

**Remove entirely** (replaced by framework):

| File | Reason |
|:-----|:-------|
| `correctness_routing.py` | → `router/correctness.py` |
| `debug_routing.py` | → `router/*` (one-off debug, not needed) |
| `debug_routing_root_cause.py` | → `router/*` + `analyze.py --mode root_cause` |
| `deep_debug_routing.py` | → `router/*` (one-off debug) |
| `gate_routing_3seeds.py` | → `router/gate.py` |
| `gate_routing_diagnostic.py` | → `router/gate.py` |
| `gradient_alignment_routing.py` | → `router/product.py` (gradient is a variant) |
| `gradient_routing.py` | → `router/product.py` |
| `hybrid_tta_routing.py` | → `router/hybrid.py` |
| `cluster_routing.py` | → `router/cluster.py` |
| `novel_routing_test.py` | → `router/*` (exploration, consolidated) |
| `pairwise_routing.py` | → `router/pairwise.py` |
| `pairwise_mlp_combined.py` | → `router/pairwise.py` (MLP variant) |
| `refined_routing_test.py` | → `router/*` (exploration, consolidated) |
| `rotation_routing.py` | → `router/tta.py` (rotation is TTA variant) |
| `selective_hybrid_routing.py` | → `router/selective.py` |
| `tta_routing.py` | → `router/tta.py` |
| `final_routing_push.py` | → `benchmark.py` |
| `final_verify_89d.py` | → `benchmark.py` |
| `multi_seed_89d_verify.py` | → `benchmark.py` (multi-seed variant) |
| `multi_seed_92d_verify.py` | → `benchmark.py` (multi-seed variant) |
| `verify_routing_hypotheses.py` | → `benchmark.py` |
| `verify_routing_target.py` | → `benchmark.py` |
| `eval_router.py` | → `benchmark.py` |
| `eval_router_v2.py` | → `benchmark.py` |
| `eval_router_bigdata.py` | → `benchmark.py` |
| `root_cause_light.py` | → `analyze.py --mode root_cause` |
| `diagnose_loss_oracle.py` | → `analyze.py --mode root_cause` |
| `kaggle_root_cause.py` | → `analyze.py --mode root_cause` (Kaggle-specific, removed) |
| `mock_test.py` | → Obsolete test file |

**Keep** (no change needed):

| File | Reason |
|:-----|:-------|
| `base_trainer.py` | Foundation for all training |
| `__init__.py` | Package marker |
| `diversity_analysis.py` | → `analyze.py --mode diversity` (consolidate) |
| `augmentation_consistency_analysis.py` | → `analyze.py --mode augmentation` (consolidate) |
| `per_class_calibration.py` | → `analyze.py --mode calibration` (consolidate) |
| `root_cause_analysis.py` | → `analyze.py --mode root_cause` (consolidate) |

---

## Summary: Before vs After

| Metric | Before | After |
|:-------|:------:|:-----:|
| Total scripts | 40 | ~15 |
| Routing scripts | 23 | 9 (clean OOP) |
| Lines of routing code | ~12,000 | ~2,500 |
| Duplicated data loading | 25 files | 1 utility |
| Duplicated BA computation | 37 files | 1 utility |
| Time to add a new router | Copy 500-line script | Subclass BaseRouter (30 lines) |

---

## OOP Architecture Diagram

```
BaseRouter (abstract)
  ├── train(val_logits, val_labels, val_features) → Self
  ├── predict(logits, features) → np.ndarray
  └── evaluate(logits, labels, features) → dict

Concrete Routers:
  ├── UniformRouter        — no train needed
  ├── ConfidenceRouter     — calibrate on val
  ├── ProductRouter        — no train needed  
  ├── CorrectnessRouter    — train trust meters on val
  ├── PairwiseRouter       — learn pairwise preferences
  ├── PairwiseMLPRouter    — train pairwise MLPs
  ├── ClusterRouter        — cluster + per-cluster preferences
  ├── GateRouter           — train MLP gate
  ├── TTARouter            — no train needed
  ├── SelectiveRouter      — wrapps any router with abstention
  └── HybridRouter         — combines multiple routers
```

Each router is **testable in isolation**:
```python
# Unit-test pattern
router = ConfidenceRouter(['LAL', 'Mixup', 'PaCo'])
router.train(val_logits, val_labels)
result = router.evaluate(test_logits, test_labels)
assert result['ba'] > 0.5
```

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|:-----|:-------|:-----------|
| Breaking changes during refactor | Can't run experiments | Implement phase by phase, keep old scripts until new ones are verified |
| Router behavior changes slightly | Different results | Write equivalence tests: old script vs new router produce same outputs |
| Too much restructuring | Never finished | Start with utility extraction (highest ROI), then router framework, then remove old files |
| OOP overhead for simple routers | More code than needed | Keep simple routers (Uniform, Product) as thin subclasses; no over-engineering |

---

## Recommendation

**Start with Phase 1** (utility extraction) — it's the highest ROI with lowest risk. Every script immediately benefits from shared data loading and metrics. Then Phase 2 (router framework) can be done incrementally — convert one routing method at a time, verifying against the old script's output.
