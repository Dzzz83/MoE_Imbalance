# Routing Mechanisms — Everything Tried, and What Happened

> Complete inventory of every routing / combination mechanism this project has
> evaluated, with a one-line description and its measured result.
>
> Results are labelled by the protocol they were measured on. **"Full-data"** =
> the original canonical protocol (10,847-sample long-tailed training set,
> balanced 10,000 test set, 3 seeds). **"Old split"** = the discarded *flawed* protocol that held out a
> balanced validation set before long-tail subsampling, where 3-expert uniform
> averaging sat at **51.12**; those numbers are historical and not comparable.
> (A second superseded protocol, the non-standard 80/20 split, had a uniform of
> 45.68 — do not confuse the two.)
>
> This catalogue records the original full-data/test-track mechanisms. The
> separate OOF development analyses are authoritative in
> [`oof-results.md`](../docs/oof-results.md).
>
> Related: [`results.md`](../docs/results.md) · [`problem.md`](../docs/problem.md) ·
> [`routing-preregistration.md`](routing-preregistration.md) (the frozen
> candidate set) · [`research.md`](../docs/research.md) (literature)

> **Audit status (2026-09-23).** The historical TTA BA/Tail row below is
> retained for provenance but is **invalid** because the earlier
> `data/tta.py` implementation used incorrect padding for normalized tensors
> and batch-shared crop semantics. It must not support a conclusion pending a
> separately authorized reevaluation. Historical routing ECE/average-confidence
> fields for Uniform, Confidence and TTA are also invalid from the old
> distribution mismatch; Probability routing calibration and per-expert
> calibration are unaffected.

---

## 1. Active rules — parameter-free, four of them

These need no held-out data: none of them fits anything, so none can be accused
of tuning on the test set. The interface enforces this — `BaseRouter` has **no**
`train()`, `fit()` or `calibrate()` method, and a test asserts no module under
`scripts/router/` references held-out labels.

| Rule | What it does | BA (3 seeds) | Tail |
|:--|:--|:--:|:--:|
| **Uniform** | Mean of expert **logits**, then argmax | **46.98 ±0.69** | 18.76 ±0.86 |
| Probability | Mean of expert **softmax probabilities**, then argmax | 45.95 ±0.55 | 18.70 ±0.50 |
| Confidence | Take the single most-confident expert's answer | 44.27 ±0.46 | 18.69 ±0.36 |
| TTA | Confidence, over 10 augmented views per image | **INVALID — 44.15 ±0.68** | **INVALID — 19.38 ±0.97** |

Among the valid historical rows, neither Probability nor Confidence beats
Uniform on both BA and Tail. The TTA comparison is withdrawn, so its former
significance and Tail statements cannot support the pre-registered conclusion.
See [`results.md`](../docs/results.md) §3.

The current TTA implementation renders ten augmented views, averages their
per-view softmax probabilities, and recovers logits as log-mean-probability.
That implementation has been corrected, but the historical row was produced
before the normalized-padding and per-sample-crop fixes. A new value requires a
separately authorized reevaluation. The router contract now also exposes
`predict_distribution`, which must use the same classifier as `predict_class`;
calibration code must consume that distribution rather than expert weights.

## 2. Removed mechanisms — needed a validation split

These results belong to the original full-data protocol. The nested-OOF
pipeline now provides held-out development predictions. No fitted router was
available for this original full-data/test-track section; the later Task 3F-A
Ridge fit is an exploratory OOF development analysis, not a validated test-track
router. Its diagnostics are recorded in [oof-results.md](../docs/oof-results.md).

Each of these five mechanisms fitted parameters on held-out labels. With no
validation split there
is no honest data to fit them on (`routing_dev` is carved from the same 10,847
samples the experts trained on, so its correctness labels are memorisation), and
the test set may not be fitted on. The code was deleted; the results are kept
here so the knowledge survives and the approaches are never silently
re-proposed.

| Mechanism | Fitting step it needed | Measured result | Verdict |
|:--|:--|:--|:--|
| **CorrectnessRouter** | logistic/MLP "trust meters" fitted on held-out correctness labels | Old split: 24-d 51.70 (+0.58), **89-d 52.42 (+1.30)**, **92-d 52.49 (+1.37)** vs uniform 51.12. Re-evaluated on the canonical split: no gain | ❌ Failed |
| **PairwiseRouter** | pairwise logistic comparators fitted on held-out preferences | Old split: tournament-soft 52.10 (+0.98); MLP comparators 50.66 (−0.46) | ❌ Failed |
| **ClusterRouter** | k-means plus per-cluster optimal weights | Old split: 52.56, agreement-pattern 51.76, soft 52.40 — all ties, inside noise | ❌ Failed |
| **GateRouter** | MLP gate trained on held-out labels | Old split: gated mixture 43.98 (**−7.14**); collapsed to one expert. DACE variant picked expert B on 90–95.7% of samples | ❌ Collapsed |
| **SelectiveRouter** | confidence threshold tuned on held-out labels | Old split: 92-d at thresh 0.35 reached 52.70 (+1.58) — the best method then recorded, but the gain vanished on the canonical split | ❌ Failed |

**`ProductRouter` was also removed**, not because it needed fitting but because
it is provably the *same classifier* as Uniform: `∏ₑ softmax(zₑ) ∝ exp(Σₑ zₑ)`,
and that normaliser does not depend on the class, so its argmax is the argmax of
the mean logits. Confirmed identical on all 3 seeds and on 64/64 synthetic
samples. Retained as a test (`test_product_is_provably_the_logit_average`) so
the justification outlives the code.

## 3. Historical catalogue — old split, 25+ methods

Measured when uniform averaging sat at **51.12**. Included for completeness;
none of these numbers are comparable with §1.

**Round 1 — basic routing**

| Method | BA | vs uniform |
|:--|:--:|:--:|
| Confidence routing (raw) | 50.64 | −0.48 |
| Confidence routing (calibrated) | 51.44 | +0.32 |
| Entropy routing | 50.10 | −1.02 |
| MLP router (192-d features, 5K train) | 49.32 | −1.80 |
| MLP router (192-d features, 45K train) | 49.82 | −1.30 |
| 3-way classifier | 26.25 | −24.87 |
| Disagreement routing | 40.72 | −10.40 |
| Gated mixture (24-d, NLL) | 43.98 | −7.14 |
| Trust-weighted product (36-d) | 52.43 | +1.31 |
| Optimal fixed weights | 52.58 | +1.46 |

*Deltas are recomputed against the stated baseline of 51.12. The original record
showed **+1.19** for the trust-weighted product, which matches no baseline in the
table (52.43 − 51.12 = 1.31); the same record also claimed a "+0.29% routing
contribution" that its own rows do not support. The BAs are as measured; only the
derived deltas were recomputed.*

**Round 2 — enriched features**

| Method | BA | vs uniform |
|:--|:--:|:--:|
| 24-d correctness prediction | 51.70 | +0.58 |
| **89-d enriched correctness** | **52.42** | **+1.30** |
| KL-divergence routing | 38.58 | −12.54 |
| Energy-based routing | 49.28 | −1.84 |
| Pairwise majority voting | 50.84 | −0.28 |
| Bayesian prior routing | 52.34 | +1.22 |
| Temperature-scaled features | 52.00 | +0.88 |
| MLP trust meters | 52.10 | +0.98 |
| Adaptive threshold routing | 52.42 | +1.30 |

**Round 3 — learning to rank**

| Method | BA | vs uniform |
|:--|:--:|:--:|
| Augmentation consistency | 51.30 | +0.18 |
| Tournament soft (pairwise LR) | 52.10 | +0.98 |
| MLP pairwise comparators | 50.66 | −0.46 |
| **92-d combined (89-d + pairwise)** | **52.49** | **+1.37** |
| Meta-router (9-d features) | 52.40 | +1.28 |

**Round 4 — TTA, gradients, selective**

| Method | BA | vs uniform |
|:--|:--:|:--:|
| TTA uniform avg (N=32) | 52.54 | +1.42 |
| TTA optimal fixed | 53.54 | +2.42 |
| 60-d TTA routing | 53.00 | +0.46 vs TTA uniform |
| Gradient sensitivity (95-d log_grad) | 52.52 | +1.40 |
| **Selective 92-d (thresh 0.35)** | **52.70** | **+1.58** |
| 392-d hybrid TTA | 53.22 | +0.68 vs TTA uniform |

**Round 5 — gradient alignment and clustering**

| Method | BA | verdict |
|:--|:--:|:--|
| GDDR (gradient alignment) | 46.98 | ❌ gradients near-orthogonal in 3072-d (cos ≈ 0.03) |
| Cluster routing (feature clustering) | 52.56 | ❌ ties global optimal fixed weights |
| Cluster routing (agreement pattern) | 51.76 | ❌ below optimal fixed weights |
| Cluster routing (soft) | 52.40 | ❌ ties optimal fixed weights |

**On the canonical split, all of these were re-evaluated and none beat uniform
(45.68 at the time).**

## 4. DACE — disagreement-aware cascade experts

An architecture that trained three experts *sequentially* (A → B → C), each with
a routing head trained by a contrastive loss on "does my prediction distribution
agree with the previous expert's?", plus class-group partitioning (A head, B
medium, C tail) to reduce the all-wrong floor. Full design in the project
history; the implementation is retired.

| Result | Value |
|:--|:--:|
| DACE 3-expert uniform (test) | 39.84 |
| Original 3-expert uniform (test) | 45.68 |
| Gap | −5.84 |
| Routing score vs "expert correct", AUROC | **0.327** (anti-predictive) |
| Score scales across experts | ~8× apart → argmax picked expert B on 90% of samples |
| Cosine-similarity fallback (threshold 0.7) | fired on **0.00%** of samples |

**Why it failed:** the routing objective was anti-predictive — with an AUROC of
0.327 for correctness (0.673 for *wrongness*), inference selected the expert most
likely to be wrong, so learning the label better made routing worse. Secondary
defects: incomparable score scales across experts, and a fallback threshold so
high it never fired. Separately, partitioning weakened every expert (each
near-zero on its non-target groups; expert B had **0.0000** tail recall — no
tail specialist existed at all). See [`problem.md`](../docs/problem.md)
§4.

## 5. Verdict

- **20+ tested mechanisms over three mechanism families** (parameter-free rules;
  fitted routers; the DACE cascade) failed to beat uniform averaging under the
  recorded protocols. This catalogue is not a mathematical proof that every
  future learned router is infeasible; fitted methods still require an honest
  OOF/held-out design.
- The binding constraint is structural, not architectural for hard expert
  selection: the historical **39.7% all-wrong floor** on the four-expert pool
  limits what a hard-selection router can recover, but it is not a ceiling for
  adaptive soft-mixture routing. [Task 3E-B](../docs/oof-results.md#5-task-3e-b--adaptive-soft-mixture-oracle)
  found 321 images correctable by convex logit combinations even though every
  expert's individual top-1 prediction was wrong. For hard selection, the
  accessible headroom (60.27 oracle − 46.98 uniform = **13.3 points**) has not
  been captured by any rule.
- Two conditions must hold for routing to win: a useful per-image signal must
  be predictable, and the resulting expert scores must be comparable. The
  historical hard-selection results did not establish the first condition;
  Tasks 3C–3F show complementary correctness but no validated predictive
  signal. The retired routing score also failed the second condition because
  it was anti-predictive and poorly scaled across experts.

The completed OOF work refines this historical verdict: complementary
correctness and convex-mixture feasibility exist on the OOF development
partition, but no fitted router has yet shown that the useful weights are
predictable. See [`oof-results.md`](../docs/oof-results.md).
