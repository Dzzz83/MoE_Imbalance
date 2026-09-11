# Routing Results Record — Removed Mechanisms

> Record of routing mechanisms that were **removed from the codebase** because
> their fitting step required a held-out (validation) split, which this project
> no longer has. The code is gone; the measured results are kept here so the
> knowledge survives and the approaches are never silently re-proposed
> (`AGENTs.md` §4).
>
> Removed on 2026-09-11. Superseding design: `docs/specs/parameter-free-routing.md`.

---

## Why they were removed

The project now trains experts on the **full** 10,847-sample long-tailed
training set with **no validation split** (see `docs/specs/data-protocol.md`).
Every one of the five mechanisms below fitted parameters on held-out labels —
Platt scaling, thresholds, per-cluster weights, learned gates or trust meters.
There is no honest held-out data left to fit them on:

- `routing_dev` is carved *from* the same 10,847 samples the experts trained on,
  so its correctness labels are memorisation, not generalisation;
- the 10K test set may not be fitted on, by definition.

Keeping the code would invite exactly the leak the protocol exists to prevent,
so the interface itself no longer offers a `train()` method.

---

## Removed mechanisms

| # | Mechanism | Fitting step it needed | Measured result | Verdict |
|:-:|:--|:--|:--|:--|
| 1 | **CorrectnessRouter** (trust meters on 89-d / 92-d features) | logistic/MLP classifiers fitted on held-out correctness labels | Old split: 24-d 51.70% (+0.58%), **89-d 52.42% (+1.30%)**, **92-d 52.49% (+1.37%)** vs uniform 51.12%. Re-evaluated on the proper split: **none beat uniform** (`results.md` §5.4) | ❌ Failed on the honest protocol |
| 2 | **PairwiseRouter** (tournament of learned comparators) | pairwise logistic regressions fitted on held-out preferences | Old split: tournament-soft 52.10% (+0.98%); MLP pairwise comparators 50.66% (−0.46%). Proper split: no gain | ❌ Failed |
| 3 | **ClusterRouter** (feature clustering + per-cluster weights) | k-means plus per-cluster optimal weights fitted on held-out data | Old split: 52.56%, agreement-pattern 51.76%, soft 52.40% — all tie global optimal-fixed weights, inside noise | ❌ Failed |
| 4 | **GateRouter** (learned MLP gate) | MLP trained on held-out labels | Old split: gated mixture (24-d, NLL) 43.98% (**−7.14%**); collapsed to always picking one expert. DACE variant selected expert B on 90–95.7% of samples | ❌ Failed (collapse) |
| 5 | **SelectiveRouter** (abstain below a confidence threshold) | threshold tuned on held-out labels | Old split: 92-d selective at thresh=0.35 reached 52.70% (+1.58%), the best routing method then recorded — but on the **proper** split its gain disappeared along with every other fitted method | ❌ Failed on the honest protocol |

*All percentages above are Balanced Accuracy on the split named in the cell.*
Numbers from the "old split" come from `research-findings.md` §2, where uniform
averaging sat at 51.12%. That split held out a balanced validation set *before*
long-tail subsampling and has since been discarded as non-standard, so those
figures are **historical only** and are not comparable with current numbers.

---

## The consolidated verdict

- **18 methods over 2 expert pools failed to beat uniform beyond noise**
  (`phase0-results.md`). This record's five mechanisms are part of that count.
- The binding constraint is structural, not architectural: a **44.2% all-wrong
  floor** on the 3-expert pool caps what any router can recover, and the
  accessible headroom is smaller than the measurement noise.
- Root causes recorded in `research-findings.md` §1 and
  `dace-diagnosis-findings.md`: the routing objective was anti-predictive of
  correctness (AUROC 0.327), the lone-dissenter paradox put the correct expert
  last in confidence in 83.9% of savable cases, and tail-group recall of 0.114
  meant the router could not recognise the samples it needed to act on.

## What survives, and why

Only mechanisms with **no fitted parameters** remain:

| Mechanism | Why it needs no held-out data |
|:--|:--|
| `UniformRouter` | plain logit averaging |
| `ProductRouter` | geometric mean of expert probabilities |
| `ConfidenceRouter` | raw max-softmax selection (the temperature-calibration path was removed with the rest) |
| `TTARouter` | delegates to a parameter-free router over TTA-averaged logits |

Their candidate set is frozen in `docs/routing-preregistration.md` **before**
any test-set evaluation, so choosing among them cannot become selection-on-test.
