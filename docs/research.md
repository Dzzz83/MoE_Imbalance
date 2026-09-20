# Research — Literature and What It Was Used For

> Every paper surveyed for this project, with an overview, the key numbers, and
> what (if anything) the project did with it. The framing is deliberately
> "literature → decision": each entry says what was concluded or tried.
>
> Related: [`project-context.md`](project-context.md) ·
> [`results.md`](results.md) · [`problem.md`](problem.md) ·
> [`routing_mechanism.md`](../records/routing_mechanism.md)

---

## Decision summary

- **Loss diversity alone has not yielded useful routing diversity** in this project; same-objective and different-objective expert pairs occupy a similar measured agreement range.
- **Sampling/controlled diversity has stronger literature support** than simply adding another related loss (RIDE, SADE/TADE, MDCS).
- **Fitted routers generally rely on balanced/held-out supervision**, which the current no-validation protocol does not provide.
- **Mixup is valuable mainly for calibration/head complementarity here**, but vanilla Mixup suppresses tail labels; LOB Mixup/MAMix explain or address that weakness.
- **Uniform logit averaging remains the active baseline**; probability averaging is measurably worse on this expert pool.
- **Sinkhorn is an allocation mechanism, not a source of routing signal**: it can coordinate weak suitability scores globally, but cannot invent comparative expertise that is absent from the expert outputs.
- **The preferred new branch is soft, residual and unbalanced**: preserve Uniform when evidence is weak, learn deviations from it with Ridge, and use soft-capacity OT rather than forcing 25/25/25/25 expert usage.
- **Any fitted Ridge router requires out-of-fold expert predictions** under the current no-validation protocol; in-sample training predictions are only an explicitly labelled exploratory shortcut.
- **Run the soft-mixture oracle and label-free OT controls before expensive OOF retraining** so the project first establishes that adaptive soft weighting/global allocation has headroom.
- Before proposing a new mechanism, check the detailed entries below and [`problem.md`](problem.md) so ruled-out ideas are not repeated.

## 1. Multi-expert architectures for long-tailed recognition

### RIDE — *Long-Tailed Recognition by Routing Diverse Distribution-Aware Experts*
Wang et al., ICLR 2021 · [arXiv:2010.01809](https://arxiv.org/abs/2010.01809)

Multiple experts on a **shared** ResNet-32 backbone with expert-specific batch
norm; diversity comes from *distribution-aware sampling* (each expert sees a
different class distribution), not from different losses. A learned router
assigns each sample to an expert.

| Method | Overall | Many | Medium | Few |
|:--|:--:|:--:|:--:|:--:|
| Cross-Entropy | 39.1 | 66.1 | 37.3 | 10.6 |
| Decouple (cRT) | 43.3 | 64.0 | 44.8 | 18.1 |
| **RIDE (3 experts)** | **48.6** | **67.0** | **49.9** | **25.7** |
| RIDE + Distill (4 experts) | 49.4 | 67.7 | 51.3 | 25.7 |
| Teacher (6 experts) | 50.2 | 69.3 | 52.1 | 25.8 |

**Used for.** The reference point for "ensemble + routing beats a single model",
and the source of the ensemble-size prior: 3 → 4 → 6 experts adds roughly +0.8
and +0.8 overall. RIDE's shared backbone differs from this project's separate
backbones with different losses.

### SADE / TADE — *Self-Supervised Aggregation of Diverse Experts*
Zhou et al., NeurIPS 2022 · [arXiv:2107.09249](https://arxiv.org/abs/2107.09249)

Three experts trained with **different supervised losses** (CE, LDAM, Balanced
Softmax) plus a self-supervised rotation-prediction task used to weight them at
test time without knowing the test distribution. ≈49.1% on CIFAR-100-LT.

**Used for.** The closest prior art to this project's premise. Its authors report
"CE + LDAM + BS" plateauing at 49% while adding random sampling reaches 50% —
i.e. **sampling diversity mattered more than loss diversity**, anticipating this
project's own finding (§6, [`problem.md`](problem.md) §3).

### ACE — *Ally Complementary Experts*
Cai et al., ICCV 2021 · [arXiv:2108.02385](https://arxiv.org/abs/2108.02385)

Multiple classifier heads on one network, trained one-shot with a complementary
loss that penalises heads for agreeing. Combines by **averaging**, not routing.

**Used for.** Evidence that an explicit diversity penalty helps an ensemble.
Never implemented.

### BalPoE / BPCE — *Balanced Product of Calibrated Experts*
Aimar et al., CVPR 2023 · [arXiv:2206.05260](https://arxiv.org/abs/2206.05260)

Trains experts on balanced subsets and combines them by a product that is
Fisher-consistent for balanced error. Combines experts in **log space** — the
mean of expert scores, eq. (10) — and proves the ensemble is unbiased when the
average logit-adjustment parameter is zero.

**Used for.** Two things. (a) It is the long-tail precedent for *logit*
averaging, which is the baseline this project uses. (b) It is the source of the
argument that **calibration is a prerequisite** for combining logit-adjusted
experts, and that **mixup is what makes experts calibrated** on CIFAR-100-LT —
the reason Mixup is the third expert here.

### MDCS — *More Diverse Experts with Consistency Self-Distillation*
Zhao et al., ICCV 2023 · [arXiv:2308.09917](https://arxiv.org/abs/2308.09917)

Enforces agreement on head classes while permitting disagreement on tail
classes. ≈50.3%.

**Used for.** The idea that *controlled* diversity beats unconstrained
diversity — relevant here, where diversity turned out to be mostly
initialisation noise.

### ELF — *An Early-Exiting Framework for Long-Tailed Classification*
Ghosh et al., ICASSP 2021 · [arXiv:2006.11979](https://arxiv.org/abs/2006.11979)

Multiple early-exit classifiers on one backbone; easy samples exit early, hard
ones go deeper. Routing is by confidence.

**Used for.** The origin of confidence-based routing as a baseline, measured here
at 44.27 BA — below uniform averaging.

## 2. Routing and mixture-of-experts

| Paper | Overview | Relevance here |
|:--|:--|:--|
| **Routers in Vision MoE: An Empirical Study**, Liu & Blondel, TMLR 2024 · [OpenReview](https://openreview.net/forum?id=5vSXd8cogo) | Compares softmax gating, top-k, linear and MLP routers. Hard routing with an MLP trained on a **balanced validation set** works best under imbalance; soft gating is unstable. | Direct guidance for a fitted router — and precisely what this project can no longer do, having no validation split. |
| **Divide, Weight, and Route**, Wei et al. 2025 · [arXiv:2508.19630](https://arxiv.org/abs/2508.19630) | A difficulty estimator scores each sample; hard samples go to balanced/margin-loss experts, easy ones to the CE expert. 50.4% on CIFAR-100-LT. | Confirms different losses give experts *difficulty-specific* strengths. Not implemented. |
| **RICASSO**, Zhang et al. 2024 · [arXiv:2410.10548](https://arxiv.org/abs/2410.10548) | RL (PPO) router with reward from a balanced validation set. 49.7%. | RL routing needs a validation set and costs ≈2× expert training. Not implemented. |
| **Sparsely-Gated MoE**, Shazeer et al., ICLR 2017 · [arXiv:1701.06538](https://arxiv.org/abs/1701.06538) | The original load-balancing auxiliary loss for MoE. | Proposed here as a fix for router collapse; unnecessary, because collapse was caused by an anti-predictive objective, not by imbalance. |

## 3. Long-tail loss functions

| Loss | Paper | Key idea |
|:--|:--|:--|
| CE | — | Unmodified softmax cross-entropy; biased toward head classes. |
| **LAL** | Menon et al., ICLR 2021 · [arXiv:2007.07314](https://arxiv.org/abs/2007.07314) | Adds `τ·log(π_y)` to logits; Bayes-optimal under prior shift. **Used, τ=1.0.** |
| **Balanced Softmax** | Ren et al., NeurIPS 2020 · [arXiv:2007.10740](https://arxiv.org/abs/2007.10740) | Adds `log(n_y)` before the softmax. **Used.** |
| LDAM | Cao et al., NeurIPS 2019 · [arXiv:1906.07413](https://arxiv.org/abs/1906.07413) | Margin `∝ n_i^(-1/4)`. Also the source of this project's training schedule. |
| Focal | Lin et al., ICCV 2017 · [arXiv:1708.02002](https://arxiv.org/abs/1708.02002) | Down-weights easy examples. |
| SupCon | Khosla et al., NeurIPS 2020 · [arXiv:2004.11362](https://arxiv.org/abs/2004.11362) | Same-class attraction on a hypersphere. |
| KCL | Kang et al., ICLR 2021 · [arXiv:2102.10078](https://arxiv.org/abs/2102.10078) | SupCon with k sampled positives. |
| TSC | Li et al., CVPR 2022 · [arXiv:2111.13998](https://arxiv.org/abs/2111.13998) | KCL plus uniform class targets. |
| PaCo | Cui et al., ICCV 2021 · [arXiv:2107.12028](https://arxiv.org/abs/2107.12028) | SupCon with learnable parametric class centres. |

**Controlled comparison** (same backbone, schedule and augmentation; TSC Table 1,
CIFAR-100-LT IR=100):

| Method | Acc |
|:--|:--:|
| CE | 38.3 |
| Focal Loss | 38.4 |
| LDAM | 39.6 |
| KCL | 43.4 |
| **TSC** | **44.3** |

**The diversity argument that shaped the expert choice** (TSC, pairwise agreement
on tail classes): classification-boundary losses agree with each other at
**κ ≈ 0.82** (CE vs LDAM), whereas a classification loss against a contrastive
loss gives **κ ≈ 0.61–0.63**. Conclusion drawn at the time: include a contrastive
expert, or the router has nothing to exploit.

**PaCo vs Balanced Softmax** (PaCo Table 6, controlled): PaCo beats BS by
**+1.2%** at IR=100.

**What happened in practice.** This project did *not* keep a contrastive expert
(PaCo was dropped from the 4-expert set). The predicted diversity advantage did
not translate into routing benefit: measured κ between the four experts spans
only 0.42–0.49, and two experts with mathematically identical objectives sit in
the middle of that band.

## 4. The mixup family

| Paper | Overview | Relevance |
|:--|:--|:--|
| **Mixup**, Zhang et al., ICLR 2018 · [code](https://github.com/facebookresearch/mixup-cifar10) | Convex combinations of image pairs and their labels; improves generalisation and calibration. | **Used**, α=1.0 (the official default). Improves head accuracy and calibration, *hurts* tail — see below. |
| **Manifold Mixup**, Verma et al., ICML 2019 · [arXiv:1806.05236](https://arxiv.org/abs/1806.05236) | Interpolates hidden states rather than inputs. | Not implemented. |
| **Remix**, Chou et al., ECCV 2020 Workshops | Relabels mixup pairs toward the minority class using `τ` and `P`-majority (suggested 0.5 and 3). | Not implemented. |
| **LOB Mixup**, Zhang et al. 2021 · [arXiv:2110.04964](https://arxiv.org/abs/2110.04964) | Diagnoses **label suppression**: with random pairing under imbalance, tail samples are mixed with head samples and their labels diluted — "introduces noise to tail data". Fixes it with two class-balanced samplers. CIFAR-100-LT IR=100 error: ERM 60.7, vanilla Mixup 59.1, **LOB 53.8**. | Explains this project's measured Mixup result exactly: head **+5.5** points over CE, tail **−2.9** points (5.69 vs 8.67). Not implemented. |
| **MAMix** (Margin-Aware Mixup), Cheng, Mai & Lin, TAAI 2023 · [arXiv:2308.15457](https://arxiv.org/abs/2308.15457) | Keeps mixup's input mixing unchanged and recomputes only the **label** factor from class-frequency weights `ηᵢ = 1/nᵢ^ω`, pushing mixed labels toward the minority class and so enforcing LDAM-style uneven margins. Released as [`imbalanced-DL`](https://github.com/ntuclab/imbalanced-DL). | The cheapest fix for Mixup's tail weakness: a label-only change to `data/mixup.py`, no sampler or architecture change. Not implemented. |

**Cross-check on published accuracy.** LOB's ERM (39.3%) and vanilla Mixup
(40.9%) are close to this project's CE (37.7%) and Mixup (38.7%) — independent
evidence that the pipeline reproduces the benchmark.

## 5. Ensembling

| Paper | Overview | Relevance |
|:--|:--|:--|
| **The Effects of Ensembling on Long-Tailed Data**, Buchanan et al., NeurIPS 2023 Heavy Tails workshop · [code](https://github.com/ekellbuch/longtail_ensembles) | Systematic comparison of logit vs probability ensembling. **No difference on balanced data**, but **differences on imbalanced data** depending on diversity. Also: ensembles outperform common long-tail methods. | The reason this project reports **both** baselines. Measured here: logit averaging wins at every ensemble size (46.98 vs 45.95 at k=4). |
| Tassi & Gawlikowski, *The impact of averaging logits over probabilities on ensembles of neural networks* · [CEUR-WS](http://sunsite.informatik.rwth-aachen.de/Publications/CEUR-WS/Vol-3215/19.pdf) | Studies the same question from the probability-averaging side. | Supporting reference for reporting both baselines. |

## 6. What was predicted versus what was measured

| Idea (source) | Predicted | Measured here |
|:--|:--|:--|
| Include a contrastive expert for diversity (TSC κ evidence) | lower κ, routable | PaCo dropped from the final set; remaining κ range 0.42–0.49 and routing still failed |
| MLP router trained on a balanced validation set (Liu & Blondel; RIDE) | best-under-imbalance router | Impossible — no validation split; all fitted routers removed |
| Different losses → diverse experts (SADE) | routable diversity | Two same-objective experts (LAL, BS) sit mid-band; diversity is largely init noise |
| Mixup fixes expert calibration (BalPoE) | calibrated experts help combination | **Confirmed**: ECE 9.43 vs 28.96–37.89 for the others |
| Mixup improves long-tail accuracy | helps | **Partly false**: +5.5 head, −2.9 tail; label suppression (LOB) explains it |
| Logit vs probability averaging is a free choice | interchangeable | **False on this data**: they differ by 1.03 points |
| Product-of-experts is a distinct combination rule | alternative baseline | **False**: mathematically identical to logit averaging |
| Rebalanced mixup (LOB, MAMix) recovers the tail | ≈+5 error points on CIFAR-100-LT | Not implemented; would likely move Mixup *toward* LAL/BS and so reduce complementarity |

## 7. Sinkhorn + Ridge as the next routing research direction

### 7.1 Why this direction is different from the failed routers

The current negative result is not simply "routing is impossible". It is more
specific: **independent per-sample signals such as confidence have not exposed a
reliable comparative-expertise signal**, while fixed averaging remains strong.
The proposed Sinkhorn + Ridge branch separates two questions:

1. **Ridge:** can expert outputs/features predict *relative expert suitability*?
2. **Sinkhorn / optimal transport:** if those suitability scores are weak but
   useful, does *global allocation* make them more exploitable than independent
   gating?

This decomposition is important. Sinkhorn does **not** discover which expert is
correct. It only converts a sample–expert score matrix into a globally coherent
transport/weight matrix. If the suitability scores contain no predictive
structure, Sinkhorn will only organise bad routing more cleanly.

For one seed, let

\[
Z\in\mathbb{R}^{N\times E\times C},\qquad N=10{,}000,\;E=4,\;C=100,
\]

be the stacked expert logits. A Ridge mapper produces sample–expert suitability
scores

\[
X\xrightarrow{\text{ridge}}S\in\mathbb{R}^{N\times E},
\]

and the OT stage produces row-stochastic expert weights

\[
S\xrightarrow{\text{Sinkhorn/UOT}}W\in\mathbb{R}^{N\times E}.
\]

The final predictor should usually be a **soft probability mixture**

\[
q_n(c)=\sum_e W_{ne}p_{ne}(c),
\]

rather than a hard single-expert choice. This is deliberately aligned with the
project's strongest empirical fact: **combining experts works better than
selecting one by confidence**.

### 7.2 Literature added for this branch

#### Sinkhorn–Knopp and entropic OT

Sinkhorn & Knopp (1967) established the alternating row/column matrix-scaling
result behind Sinkhorn iterations. Cuturi (NeurIPS 2013) made the method central
to modern machine learning by adding entropy regularisation to OT, allowing the
transport plan to be computed efficiently by Sinkhorn scaling.

**Decision.** Use ordinary entropy-regularised Sinkhorn as the reference OT
primitive. Implement it in the log domain because the project only needs a
small `10000 × 4` transport matrix and numerical transparency matters more than
adding a large dependency.

#### BASE Layers — balanced expert assignment

Lewis et al. (ICML 2021) formulate token-to-expert routing as a balanced linear
assignment so that every expert receives equal load.

**Decision.** This is useful precedent for treating routing as a **global
assignment problem**, but its motivation is partly compute/load balancing. In
`MoE_Imbalance`, the four frozen classifiers have unequal quality and no
capacity bottleneck, so strict 25/25/25/25 usage is a baseline, not the preferred
inductive bias.

#### Unbalanced optimal transport (UOT)

Chizat et al. extend Sinkhorn-style scaling to unbalanced OT, where the marginal
constraints are relaxed rather than enforced exactly. Chapel et al. (NeurIPS
2021) further show that UOT with marginal penalties can be reformulated as a
non-negative penalised linear-regression problem, including efficient treatment
of quadratic penalties.

**Decision.** **Unbalanced/soft-capacity OT should be the primary Sinkhorn
variant** for this project. It prevents trivial expert collapse without assuming
that CE, LAL, BalancedSoftmax and Mixup each deserve exactly 25% of the routing
mass.

#### Selective Sinkhorn Routing (SSR)

Nguyen et al. (2026) revisit sparse-MoE token assignment as entropy-regularised
OT. Their Selective Sinkhorn Routing uses transport-map values as expert weights
and applies Sinkhorn only selectively rather than on every routing step. Their
training-time sparse-MoE setting is not directly comparable to four frozen
ResNet classifiers.

**Decision.** The transferable ideas are (a) use the **transport values as
weights**, not only top-1 assignments, and (b) Sinkhorn need not control every
sample. This motivates the project's **Selective Residual Ridge + UOT** proposal
below.

#### Smooth and Sparse OT

Blondel, Seguy & Rolet (AISTATS 2018) show that entropy produces dense positive
transport plans while squared-L2 and related strongly convex regularisers can
produce sparse plans.

**Decision.** Quadratic/L2 OT is a useful *regulariser ablation*, especially if
soft Sinkhorn remains too diffuse. It is **not vanilla Sinkhorn** and should not
be described as such.

#### Temperature scaling

Guo et al. (ICML 2017) show that temperature scaling is a simple effective
post-hoc calibration method for neural classifiers.

**Decision.** Because the four experts can have different logit scales, a
`TemperatureCalibratedRidge` ablation is justified before feeding confidence,
entropy or margin features to Ridge. Temperatures must be estimated from honest
OOF predictions, not from the test labels.

#### R2-T2 — local test-time rerouting

Li, Li & Zhou (ICML 2025) improve multimodal MoE routing at test time by moving
routing weights toward those of correctly handled neighbouring calibration
samples.

**Decision.** This supports exploring **local/neighbourhood routing structure**
when a single global Ridge mapping is too simple. The project's proposed local
Ridge variant is an adaptation, not a reproduction of R2-T2.

#### ROAM — graph-regularised Sinkhorn

Tian et al. (2026 preprint) use capacity-constrained entropic OT and graph
regularisation so neighbouring regions prefer coherent expert assignments.

**Decision.** This motivates an exploratory **Graph-Smoothed Ridge → Sinkhorn**
variant in which similarity is between CIFAR images/routing features rather than
spatial pathology regions.

### 7.3 Protocol constraint: fitted Ridge requires honest OOF predictions

The current experts train on the entire fixed 10,847-image long-tailed training
set. Therefore fitting Ridge on predictions from those same models on those same
training images would teach the router from **in-sample expert behaviour**.
Different random seeds do not make the images held out.

The clean fitted protocol is:

1. split the 10,847 training images into `K` folds;
2. for each fold, train all four experts without that fold;
3. predict the held-out fold;
4. concatenate those out-of-fold (OOF) predictions;
5. fit all Ridge-based routers on this one shared OOF artifact;
6. discard the fold experts and evaluate the router on the existing full-data
   experts.

The OOF artifact should be reused by every fitted routing method so the study
does not repeatedly retrain experts for each router.

### 7.4 Cheap diagnostics before OOF retraining

Before paying the OOF training cost, run these diagnostics on the already cached
expert logits.

| Diagnostic | Question answered | Decision use |
|:--|:--|:--|
| **Soft-mixture oracle** | Can *any convex probability mixture* correct substantially more samples than the 60.27% hard-choice oracle? | If not, adaptive soft weighting has limited headroom. |
| **Label-free UOT vs same-score independent softmax** | Does global coupling itself help when the suitability signal is identical? | Separates Sinkhorn value from score-design value. |
| **Top-1 vs top-2 vs all-4 soft oracle/mixture** | Is the useful regime hard selection, sparse mixture, or dense adaptive ensembling? | Determines whether later Ridge should target `k=1`, `k=2`, or all experts. |
| **Strict vs soft expert capacities** | Does 25/25/25/25 itself damage routing? | Determines whether balanced Sinkhorn is only a control. |
| **Savable-sample score AUC/ranking** | On `Uniform wrong ∩ some expert correct`, does any label-free score rank the correct expert above the others? | If every score is near/random or anti-predictive, do not expect Sinkhorn to invent signal. |

The **soft-mixture oracle** should be added to `HeadroomAnalyzer`: for each
sample, solve whether there exists `w` on the 4-simplex such that the true class
has maximal mixed probability. This oracle may exceed the hard-choice oracle
because several individually wrong experts can still form a correct convex
mixture.

---

## 8. Candidate Sinkhorn + Ridge formulations, ranked

### 8.1 Ranking criteria

The ranking below is specific to the current four-expert CIFAR-100-LT pool. It
weights:

- chance of beating **Uniform logit averaging (46.98 BA)**;
- chance of helping **Tail** without destroying overall BA;
- methodological cleanliness under the **no-validation** protocol;
- compute cost;
- implementation complexity;
- scientific novelty/diagnostic value.

`UOT` below means entropy-regularised **unbalanced/soft-capacity optimal
transport** unless stated otherwise. `OOF` means the Ridge model is fitted only
on out-of-fold expert predictions.

| Rank | Method | Main idea | Supervision | Output | Main reason for rank |
|:--:|:--|:--|:--|:--|:--|
| **1** | **Selective Residual Ridge → UOT** | Start from Uniform and let Ridge predict only justified deviations; apply OT only to samples with strong evidence. | OOF labels | Soft | Protects the strongest baseline and avoids forcing a routing decision on every image. |
| **2** | **Class-Balanced Advantage Ridge → UOT** | Weighted Ridge predicts each expert's advantage over Uniform/mean expert, with class-balanced fitting. | OOF labels | Soft | Directly aligned with BA/tail goals and prevents head classes dominating the gate fit. |
| **3** | **Advantage Ridge → UOT** | Ridge predicts relative expert benefit, then UOT globally coordinates the weights. | OOF labels | Soft | Cleanest main Ridge+Sinkhorn formulation and essential reference model. |
| **4** | **Top-2 Advantage Ridge → UOT** | Keep only the two strongest transport weights per sample and renormalise. | OOF labels | Sparse soft | Middle ground between successful averaging and unsuccessful hard routing. |
| **5** | **Shrink-to-Uniform Ridge → UOT** | Penalise routing weights for moving far from `[.25,.25,.25,.25]`. | OOF labels | Soft | Conservative adaptive ensembling; weak evidence falls back toward the known-good ensemble. |
| **6** | **Label-free Unbalanced Sinkhorn** | Hand-designed consensus/disagreement suitability, with soft capacities. | None | Soft | Cheapest meaningful OT prototype; should be run before any OOF retraining. |
| **7** | **Temperature-Calibrated Ridge → UOT** | Calibrate each expert on OOF data before constructing Ridge features/scores. | OOF labels | Soft | Removes a plausible logit-scale confound before comparative-expertise modelling. |
| **8** | **Hierarchical H/M/T Ridge → UOT** | Global Ridge plus shrinkage/group offsets for Head/Medium/Tail regimes. | OOF labels | Soft | Tail-aware without trying to fit 100 data-starved class-specific gates. |
| **9** | **Performance-Prior Marginal UOT** | Set expert capacity priors from OOF expert usefulness rather than equal 25% mass. | OOF labels for prior | Soft | More realistic demand prior than equal capacities; simple once OOF data exists. |
| **10** | **Ridge → Dual-Price Gate** | Use Sinkhorn on reference data to learn expert dual biases/prices, then route future samples independently. | OOF or unlabeled reference | Soft | Converts transductive OT into a deployable inductive gate. |
| **11** | **Local/Neighbourhood Ridge → UOT** | Predict suitability from nearby OOF calibration samples instead of one global linear model. | OOF labels | Soft | Can capture local specialisation missed by global Ridge, but adds complexity/variance. |
| **12** | **Graph-Smoothed Ridge → Sinkhorn** | Add a graph penalty so similar samples prefer similar expert weights. | OOF labels or label-free graph | Soft | Novel transductive structure; useful if routing patterns cluster locally. |
| **13** | **TTA-Consistency Ridge → UOT** | Add augmentation-stability features to Ridge. | OOF labels | Soft | Extra information may help, but consistency is not correctness and inference is expensive. |
| **14** | **Label-free strict balanced Sinkhorn** | Use a hand-designed score and force exact 25/25/25/25 soft mass. | None | Soft/hard | Important cheap control, but equal expert demand is a weak prior for this pool. |
| **15** | **Direct penalised-regression UOT** | Put quadratic marginal penalties and suitability directly into one UOT/regression-style objective. | Depends on score | Soft | Elegant link between Ridge-like penalties and UOT, but harder to interpret causally. |
| **16** | **Pairwise Ridge → Sinkhorn** | Learn pairwise expert preferences, aggregate them into suitability, then run OT. | OOF labels | Soft/hard | Possible, but prior pairwise routing results reduce the expected payoff. |
| **17** | **Strict 25/25/25/25 Ridge → Sinkhorn** | Standard Ridge suitability with exact equal expert marginals. | OOF labels | Soft/hard | Necessary balanced-OT baseline; likely to waste mass on weaker/redundant experts. |
| **18** | **Quadratic/L2 OT on Ridge scores** | Replace entropy with squared-L2 regularisation to obtain potentially sparse plans. | OOF labels | Sparse soft | Good regulariser ablation, but it is no longer ordinary Sinkhorn. |
| **19** | **Entropy + quadratic-plan OT** | Combine entropy with a quadratic penalty toward a prior plan. | OOF labels | Soft | Flexible but introduces another regularisation axis before simpler variants are understood. |
| **20** | **EM/self-refining Ridge–Sinkhorn** | Alternate routing/pseudo-target updates and refit the gate. | OOF seed + pseudo-labels | Soft/hard | Highest confirmation-bias risk given this project's history of anti-predictive routing signals. |

### 8.2 The top recommended methods in detail

#### #1 Selective Residual Ridge → UOT — **primary proposal**

The current result says the safest prediction rule is already Uniform. Therefore
Ridge should not be asked to construct a routing policy from scratch. Let

\[
w_0=(0.25,0.25,0.25,0.25)
\]

and learn only a residual

\[
\Delta w_n=f_{\text{ridge}}(X_n).
\]

The candidate weights are

\[
\tilde w_n=w_0+\alpha_n\Delta w_n,
\]

where `α_n` is a **predefined or OOF-derived evidence strength**. If the gate is
uncertain, `α_n≈0` and the sample stays close to Uniform. Samples with meaningful
predicted expert advantage can deviate more strongly. UOT then coordinates the
selected residual changes across the population without enforcing exact equal
expert use.

Two controls are mandatory:

- the same residual Ridge **without UOT**;
- the same UOT applied to **all** samples rather than selectively.

If selective residual routing works while full routing does not, the useful
finding is that **most samples should remain ensembles and only a subset should
be adaptively reweighted**.

#### #2 Class-Balanced Advantage Ridge → UOT — **best tail-aware formulation**

Ordinary Ridge on the long-tailed OOF data is dominated by head-class examples.
Instead fit

\[
\min_B\sum_n \alpha_{y_n}\|Y_n-X_nB\|^2+\lambda\|B\|_F^2,
\qquad \alpha_c\propto 1/N_c,
\]

where the target should preferably represent **relative expert advantage**, e.g.

\[
Y_{ne}=\log p_{ne,y_n}-\frac1E\sum_j\log p_{nj,y_n}.
\]

This asks Ridge to predict *how much expert `e` improves or worsens the true
label relative to the pool*, rather than merely predicting a binary
correct/incorrect flag. UOT then converts these scores into soft expert weights.

This is the most direct Ridge+Sinkhorn method for the project's balanced-accuracy
objective.

#### #3 Advantage Ridge → UOT — **main simple baseline**

Fit the same advantage target without class reweighting, then use UOT. This must
be retained even if the class-balanced version is preferred because it isolates
whether class weighting itself is responsible for any tail gain.

#### #4 Top-2 Advantage Ridge → UOT — **sparse adaptive ensemble**

After UOT returns `W_n`, keep the largest two entries and renormalise. This tests
whether the project really needs all four experts per image or whether a small
adaptive mixture is enough. Report `k=1`, `k=2`, and `k=4` from the same score
matrix where possible.

#### #5 Shrink-to-Uniform Ridge → UOT — **conservative control**

Instead of selective routing, add an explicit penalty toward Uniform:

\[
\gamma\sum_n\|W_n-w_0\|_2^2.
\]

This directly encodes the empirical prior that Uniform is hard to beat. It is
simpler than #1 but less interpretable about *which* samples should remain
unchanged.

### 8.3 Other useful variants

#### Temperature-Calibrated Ridge → UOT

Estimate one temperature per expert from OOF predictions, transform
`z_e → z_e/T_e`, then compute confidence/entropy/margin/disagreement features.
Compare against the exact same Ridge feature set without temperature scaling.

#### Hierarchical H/M/T Ridge → UOT

Use shared global coefficients plus strongly regularised Head/Medium/Tail
adjustments. Do **not** fit 100 independent class gates: tail classes have too
few OOF observations for a stable model.

#### Performance-Prior Marginal UOT

Instead of `b=(.25,.25,.25,.25)`, derive a fixed expert-demand prior from OOF
performance only. A stronger extension uses H/M/T-specific priors. Do not derive
`b` from test accuracy.

#### Ridge → Dual-Price Gate

Run Sinkhorn on a calibration/reference population and retain expert-side dual
biases. At deployment, use a per-sample softmax over Ridge suitability plus
those frozen expert prices. This sacrifices exact batch-level marginals but
removes the transductive requirement.

#### Local/Neighbourhood Ridge → UOT

Use neighbours in OOF routing-feature space to estimate a local Ridge mapping or
local correction to the global Ridge scores. This is inspired by R2-T2's
neighbour-based test-time rerouting, but must be evaluated as a new adaptation.

#### Graph-Smoothed Ridge → Sinkhorn

Construct a k-NN graph over samples using routing features or frozen
representations and add a smoothness term such as

\[
\mu\sum_{(i,j)}A_{ij}\|W_i-W_j\|_2^2.
\]

This is inspired by graph-regularised OT routing in ROAM. It is transductive and
should only be attempted after simpler Ridge/UOT variants establish a positive
signal.

#### TTA-Consistency Ridge → UOT

Add expert stability across augmentations as features. This can only be a
supporting feature because a model can be stably wrong.

#### Direct penalised-regression UOT

Use the regression interpretation of UOT to solve sample–expert assignment with
quadratic marginal penalties directly. This is mathematically attractive but
should be an ablation after the separated `Ridge → UOT` design, because otherwise
it becomes difficult to tell whether gains came from better suitability
prediction or the OT regulariser.

#### Pairwise Ridge → Sinkhorn

Fit pairwise preferences such as `CE > LAL`, aggregate the comparisons into four
expert scores, and apply OT. This is lower priority because earlier pairwise
routers did not expose reliable comparative expertise.

#### Quadratic and mixed OT

Use squared-L2 OT, or entropy plus a quadratic penalty toward a prior plan, only
as regulariser studies. They answer **which transport geometry works best**, not
whether a useful routing signal exists in the first place.

#### EM/self-refining Ridge–Sinkhorn

Alternate between producing routing weights, generating pseudo-targets and
refitting the gate. This is deliberately last because an initially wrong gate
can reinforce its own errors. Any such experiment needs strong stopping rules
and must remain exploratory.

### 8.4 Essential controls

Every fitted Sinkhorn result must be paired with controls that isolate *where the
gain came from*.

| Control | What it isolates |
|:--|:--|
| **Uniform logit averaging** | Existing strongest fixed baseline. |
| **Probability averaging** | Existing probability-mixture baseline. |
| **RidgeGate without Sinkhorn** | Does the learned suitability model work at all? |
| **Same Ridge scores + independent softmax** | Does global OT coupling help beyond an ordinary gate? |
| **Balanced Sinkhorn vs UOT** | Is exact 25% capacity beneficial or harmful? |
| **Hard top-1 vs soft top-2 vs soft all-4** | Is the useful regime selection or adaptive ensembling? |
| **Uncalibrated vs temperature-calibrated features** | Is expert logit scale driving the apparent gain? |
| **Ordinary Ridge vs class-balanced Ridge** | Is any tail gain caused by the BA-aligned fit? |
| **Residual Ridge vs free Ridge weights** | Does protecting Uniform prevent destructive over-routing? |
| **OOF-fitted vs cheap in-sample Ridge (clearly marked)** | Measures optimism caused by training the gate on expert in-sample behaviour. |

### 8.5 Recommended staged experiment plan

**Stage A — no retraining.**

1. Add the soft-mixture oracle.
2. Cache/verify the full `(N,E,C)` logits for all three seeds.
3. Implement log-domain entropic Sinkhorn.
4. Compare label-free independent softmax vs balanced Sinkhorn vs UOT using the
   exact same suitability score.
5. Compare hard top-1, top-2 mixture and all-4 mixture.
6. If global OT does not improve the same-score control at all, treat that as a
   warning before spending on OOF training.

**Stage B — one shared OOF calibration artifact.**

7. Train `K`-fold OOF versions of the four experts once.
8. Save OOF logits/probabilities, labels, fold IDs and feature statistics.
9. Fit `AdvantageRidge` and `ClassBalancedAdvantageRidge`.
10. Compare `RidgeGate`, independent softmax, balanced Sinkhorn and UOT.

**Stage C — top research variants.**

11. Add **Selective Residual Ridge → UOT**.
12. Add **Shrink-to-Uniform Ridge → UOT**.
13. Add **Top-2 Advantage Ridge → UOT**.
14. Add temperature calibration and H/M/T hierarchy only if the basic Ridge
    scores contain measurable comparative-expertise signal.

**Stage D — exploratory structure.**

15. Try performance-prior marginals and dual-price inductive routing.
16. Try local/neighbourhood or graph-smoothed routing if routing success clusters
    in feature space.
17. Only then study L2/quadratic OT or iterative self-refinement.

### 8.6 Decision rule for continuing the branch

Proceed to expensive OOF Ridge work only if at least one cheap result indicates
that **allocation or soft weighting has genuine headroom**:

- the soft-mixture oracle materially exceeds the hard-choice oracle;
- top-2/all-4 mixtures rescue samples that hard expert selection cannot;
- label-free UOT improves over the same suitability followed by independent
  softmax gating; or
- a label-free suitability score has non-trivial ranking power on the
  `Uniform-wrong, some-expert-correct` subset.

After honest OOF fitting:

- if `RidgeGate > Uniform` but `Ridge+UOT <= RidgeGate`, the result is **adaptive
  ensembling works; OT coupling does not**;
- if `Ridge+UOT > RidgeGate`, there is evidence that **global allocation adds
  value beyond suitability prediction**;
- if selective/residual routing works while free routing fails, the correct
  framing is **route only when there is evidence; otherwise preserve Uniform**;
- if no honest fitted method improves Uniform, the negative conclusion becomes
  stronger: **the current expert pool contains diversity useful for averaging
  but not sufficiently predictable sample-level specialisation**.

---

## 9. References

1. Zhang, H. et al. (2018). *mixup: Beyond Empirical Risk Minimization*. ICLR. [code](https://github.com/facebookresearch/mixup-cifar10)
2. Cai, J., Wang, Y., Hwang, J.-N. (2021). *ACE: Ally Complementary Experts*. ICCV. [arXiv:2108.02385](https://arxiv.org/abs/2108.02385)
3. Zhou, Z. et al. (2022). *SADE: Self-Supervised Aggregation of Diverse Experts*. NeurIPS. [arXiv:2107.09249](https://arxiv.org/abs/2107.09249)
4. Aimar, E. S. et al. (2023). *Balanced Product of Calibrated Experts*. CVPR. [arXiv:2206.05260](https://arxiv.org/abs/2206.05260)
5. Zhao, Q. et al. (2023). *MDCS: More Diverse Experts with Consistency Self-Distillation*. ICCV. [arXiv:2308.09917](https://arxiv.org/abs/2308.09917)
6. Liu, Z., Blondel, M. (2024). *Routers in Vision Mixture of Experts*. TMLR. [OpenReview](https://openreview.net/forum?id=5vSXd8cogo)
7. Ghosh, A. et al. (2021). *ELF: An Early-Exiting Framework for Long-Tailed Classification*. ICASSP. [arXiv:2006.11979](https://arxiv.org/abs/2006.11979)
8. Menon, A. K. et al. (2021). *Long-tail learning via logit adjustment*. ICLR. [arXiv:2007.07314](https://arxiv.org/abs/2007.07314) · [code](https://github.com/google-research/google-research/tree/master/logit_adjustment)
9. Ren, J. et al. (2020). *Balanced Meta-Softmax*. NeurIPS. [arXiv:2007.10740](https://arxiv.org/abs/2007.10740) · [code](https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification)
10. Cao, K. et al. (2019). *LDAM: Label-Distribution-Aware Margin Loss*. NeurIPS. [arXiv:1906.07413](https://arxiv.org/abs/1906.07413) · [code](https://github.com/kaidic/LDAM-DRW)
11. Kang, B. et al. (2020). *Decoupling Representation and Classifier*. ICLR. [arXiv:1910.09217](https://arxiv.org/abs/1910.09217)
12. Wei, X. et al. (2025). *Divide, Weight, and Route*. [arXiv:2508.19630](https://arxiv.org/abs/2508.19630)
13. Zhang, X. et al. (2024). *RICASSO*. [arXiv:2410.10548](https://arxiv.org/abs/2410.10548)
14. Khosla, P. et al. (2020). *Supervised Contrastive Learning*. NeurIPS. [arXiv:2004.11362](https://arxiv.org/abs/2004.11362)
15. Shazeer, N. et al. (2017). *Outrageously Large Neural Networks*. ICLR. [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)
16. Li, T. et al. (2022). *Targeted Supervised Contrastive Learning*. CVPR. [arXiv:2111.13998](https://arxiv.org/abs/2111.13998)
17. Kang, B. et al. (2021). *K-positive Contrastive Learning*. ICLR. [arXiv:2102.10078](https://arxiv.org/abs/2102.10078)
18. Cui, J. et al. (2021). *Parametric Contrastive Learning*. ICCV. [arXiv:2107.12028](https://arxiv.org/abs/2107.12028)
19. Lin, T.-Y. et al. (2017). *Focal Loss*. ICCV. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
20. He, K. et al. (2020). *MoCo v2*. CVPR. [arXiv:2003.04297](https://arxiv.org/abs/2003.04297)
21. Zhang, S., Chen, C., Zhang, X., Peng, S. (2021). *Label-Occurrence-Balanced Mixup*. [arXiv:2110.04964](https://arxiv.org/abs/2110.04964)
22. Cheng, W.-C., Mai, T.-H., Lin, H.-T. et al. (2023). *From SMOTE to Mixup for Deep Imbalanced Classification* (MAMix). TAAI. [arXiv:2308.15457](https://arxiv.org/abs/2308.15457) · [code](https://github.com/ntuclab/imbalanced-DL)
23. Chou, H.-P. et al. (2020). *Remix: Rebalanced Mixup*. ECCV Workshops.
24. Verma, V. et al. (2019). *Manifold Mixup*. ICML. [arXiv:1806.05236](https://arxiv.org/abs/1806.05236)
25. Buchanan, E. K., Pleiss, G., Wang, Y., Cunningham, J. P. (2023). *The Effects of Ensembling on Long-Tailed Data*. NeurIPS Heavy Tails Workshop. [code](https://github.com/ekellbuch/longtail_ensembles)
26. Tassi, C. R., Gawlikowski, J. *The impact of averaging logits over probabilities on ensembles of neural networks*. [CEUR-WS Vol-3215](http://sunsite.informatik.rwth-aachen.de/Publications/CEUR-WS/Vol-3215/19.pdf)
27. Wang, X. et al. (2021). *RIDE: Long-Tailed Recognition by Routing Diverse Distribution-Aware Experts*. ICLR. [arXiv:2010.01809](https://arxiv.org/abs/2010.01809)
28. Sinkhorn, R., Knopp, P. (1967). *Concerning Nonnegative Matrices and Doubly Stochastic Matrices*. Pacific Journal of Mathematics 21(2):343–348. [DOI](https://doi.org/10.2140/pjm.1967.21.343)
29. Cuturi, M. (2013). *Sinkhorn Distances: Lightspeed Computation of Optimal Transport*. NeurIPS. [proceedings](https://proceedings.neurips.cc/paper_files/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html)
30. Peyré, G., Cuturi, M. (2019). *Computational Optimal Transport*. Foundations and Trends in Machine Learning. [book](https://optimaltransport.github.io/book/)
31. Hoerl, A. E., Kennard, R. W. (1970). *Ridge Regression: Biased Estimation for Nonorthogonal Problems*. Technometrics 12(1):55–67. [DOI](https://doi.org/10.1080/00401706.1970.10488634)
32. Lewis, M. et al. (2021). *BASE Layers: Simplifying Training of Large, Sparse Models*. ICML. [PMLR](https://proceedings.mlr.press/v139/lewis21a.html)
33. Chizat, L., Peyré, G., Schmitzer, B., Vialard, F.-X. (2018). *Scaling Algorithms for Unbalanced Optimal Transport Problems*. Mathematics of Computation. [arXiv:1607.05816](https://arxiv.org/abs/1607.05816)
34. Chapel, L., Flamary, R., Wu, H., Févotte, C., Gasso, G. (2021). *Unbalanced Optimal Transport through Non-negative Penalized Linear Regression*. NeurIPS. [proceedings](https://proceedings.neurips.cc/paper_files/paper/2021/hash/c3c617a9b80b3ae1ebd868b0017cc349-Abstract.html)
35. Blondel, M., Seguy, V., Rolet, A. (2018). *Smooth and Sparse Optimal Transport*. AISTATS. [PMLR](https://proceedings.mlr.press/v84/blondel18a.html)
36. Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks*. ICML. [PMLR](https://proceedings.mlr.press/v70/guo17a.html)
37. Li, Z., Li, Z., Zhou, T. (2025). *R2-T2: Re-Routing in Test-Time for Multimodal Mixture-of-Experts*. ICML. [PMLR](https://proceedings.mlr.press/v267/li25bc.html)
38. Nguyen, D. A. et al. (2026). *Selective Sinkhorn Routing for Improved Sparse Mixture of Experts*. ICML 2026 AdaptFM Workshop. [OpenReview](https://openreview.net/pdf?id=qRQU6W1vJ4) · [arXiv:2511.08972](https://arxiv.org/abs/2511.08972)
39. Tian, X. et al. (2026). *Region-Graph Optimal Transport Routing for Mixture-of-Experts Whole-Slide Image Classification (ROAM)*. preprint. [arXiv:2604.07298](https://arxiv.org/abs/2604.07298)

