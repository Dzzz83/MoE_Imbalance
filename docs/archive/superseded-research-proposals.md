# Superseded Ridge/Sinkhorn Proposals

> Historical archive. This material was written before the completed Task 3C,
> Task 3E-A and Task 3E-B analyses. The rankings, formulations and staged
> implementation sequence are hypotheses preserved for traceability. They are
> not validated conclusions, current instructions or approval to implement
> Ridge, Sinkhorn, new experts or a full nested experiment.

## Original context

The proposal was created when the project had only the original full-data
experts and no honest held-out router data. It argued that a fitted Ridge model
would require cross-fitted expert predictions, and that Sinkhorn could
coordinate sample–expert scores but could not create predictive signal absent
from those scores.

The intended decomposition was:

1. Ridge predicts relative expert suitability from inference-time features.
2. Sinkhorn or unbalanced optimal transport turns suitability scores into
   globally coordinated soft weights.

The final predictor was proposed as a soft probability mixture rather than
hard single-expert selection. This archive retains that reasoning, but the
completed soft oracle now belongs in [oof-results.md](../oof-results.md), and
the current research boundary is in [research.md](../research.md).

## Historical ranked variants

| Rank | Method | Main idea | Supervision | Output |
|:--:|:--|:--|:--|:--|
| 1 | Selective Residual Ridge → UOT | Start from Uniform and predict only justified deviations; apply OT selectively | OOF labels | Soft |
| 2 | Class-Balanced Advantage Ridge → UOT | Predict expert advantage with inverse-class-frequency fitting | OOF labels | Soft |
| 3 | Advantage Ridge → UOT | Predict relative expert benefit, then coordinate with UOT | OOF labels | Soft |
| 4 | Top-2 Advantage Ridge → UOT | Retain two largest transport weights and renormalize | OOF labels | Sparse soft |
| 5 | Shrink-to-Uniform Ridge → UOT | Penalize movement away from equal weights | OOF labels | Soft |
| 6 | Label-free Unbalanced Sinkhorn | Hand-designed suitability with relaxed expert capacities | None | Soft |
| 7 | Temperature-Calibrated Ridge → UOT | Calibrate expert logits before constructing features | OOF labels | Soft |
| 8 | Hierarchical Head/Medium/Tail Ridge → UOT | Add strongly regularized group offsets | OOF labels | Soft |
| 9 | Performance-Prior Marginal UOT | Derive expert demand priors from OOF performance | OOF labels for prior | Soft |
| 10 | Ridge → Dual-Price Gate | Convert reference-population OT prices into an inductive gate | OOF or unlabeled reference | Soft |
| 11 | Local/Neighbourhood Ridge → UOT | Use local OOF feature neighborhoods for suitability | OOF labels | Soft |
| 12 | Graph-Smoothed Ridge → Sinkhorn | Encourage similar samples to have similar weights | OOF or graph | Soft |
| 13 | TTA-Consistency Ridge → UOT | Add augmentation-stability features | OOF labels | Soft |
| 14 | Label-free strict balanced Sinkhorn | Force exact 25/25/25/25 mass as a control | None | Soft or hard |
| 15 | Direct penalized-regression UOT | Combine suitability regression and marginal penalties | Depends on score | Soft |
| 16 | Pairwise Ridge → Sinkhorn | Predict pairwise expert preferences, then aggregate | OOF labels | Soft or hard |
| 17 | Strict 25/25/25/25 Ridge → Sinkhorn | Standard Ridge with exact equal marginals | OOF labels | Soft or hard |
| 18 | Quadratic/L2 OT on Ridge scores | Replace entropy with a sparse-producing transport regularizer | OOF labels | Sparse soft |
| 19 | Entropy plus quadratic-plan OT | Add a quadratic prior-plan penalty | OOF labels | Soft |
| 20 | EM/self-refining Ridge–Sinkhorn | Alternate pseudo-target generation and refitting | OOF plus pseudo-labels | Soft or hard |

The historical ordering weighted expected BA/Tail value, methodological
cleanliness, compute, complexity and diagnostic value. It was not an
experimentally validated ranking.

## Historical primary proposal

The first proposal, Selective Residual Ridge → UOT, kept the known strong
baseline as the default:

\[
w_0=(0.25,0.25,0.25,0.25), \qquad
\tilde w_n=w_0+\alpha_n\Delta w_n.
\]

The intended interpretation was that uncertain samples remain close to
Uniform, while only samples with a strong predicted advantage receive larger
deviations. Mandatory controls were the same residual Ridge without UOT and
the same UOT applied to every sample.

The second proposal used a class-balanced objective:

\[
\min_B\sum_n \alpha_{y_n}\lVert Y_n-X_nB\rVert^2+\lambda\lVert B\rVert_F^2,
\qquad \alpha_c\propto 1/N_c,
\]

with a possible relative-advantage target such as

\[
Y_{ne}=\log p_{ne,y_n}
-\frac{1}{E}\sum_j\log p_{nj,y_n}.
\]

These equations remain historical hypotheses. The target, feature set and
regularization have not been frozen for a current experiment.

## Historical controls

The proposal required every fitted result to be compared with:

- uniform logit averaging;
- probability averaging;
- the learned gate without Sinkhorn;
- independent softmax gating from the same scores;
- balanced Sinkhorn versus unbalanced/soft-capacity OT;
- hard top-1, soft top-2 and soft all-4 mixtures;
- uncalibrated versus OOF temperature-calibrated features;
- ordinary versus class-balanced Ridge;
- residual Ridge versus free weights; and
- an explicitly labeled in-sample shortcut to quantify optimism.

The proposal also treated strict 25/25/25/25 allocation as a control, not as a
preferred prior, because the four experts have unequal quality and there is no
compute-capacity constraint.

## Historical staged plan

### Stage A — no retraining

1. Add a soft-mixture oracle.
2. Verify aligned expert logits.
3. Implement log-domain entropic Sinkhorn.
4. Compare independent softmax, balanced Sinkhorn and UOT with the same score.
5. Compare hard top-1, top-2 and all-4 mixtures.
6. Stop before OOF retraining if allocation does not help the same-score control.

### Stage B — shared OOF artifact

1. Train K-fold OOF versions of the four experts once.
2. Save logits, probabilities, labels, fold IDs and feature statistics.
3. Fit AdvantageRidge and ClassBalancedAdvantageRidge.
4. Compare a learned gate, independent softmax, balanced Sinkhorn and UOT.

### Stage C — conservative variants

1. Add Selective Residual Ridge → UOT.
2. Add Shrink-to-Uniform Ridge → UOT.
3. Add Top-2 Advantage Ridge → UOT.
4. Add temperature calibration and Head/Medium/Tail hierarchy only if a basic
    suitability signal exists.

### Stage D — exploratory structure

1. Try performance-prior marginals and dual-price inductive routing.
2. Try local or graph-smoothed routing if success clusters in feature space.
3. Study quadratic OT or self-refinement only after simpler variants.

## Historical continuation rule

The branch was to proceed only if a cheap result showed soft-mixture or
allocation headroom, a label-free score ranked the useful expert, or honest OOF
fitting produced a signal. It proposed interpreting outcomes as follows:

- learned gate above Uniform but learned gate plus OT not above it:
  adaptive ensembling works, OT coupling does not;
- learned gate plus OT above the learned gate: global allocation adds value;
- selective/residual success but free routing failure: route only with evidence;
- no honest fitted gain: the pool is useful for averaging but not predictably
  specialized.

Task 3E-B has now answered only the feasibility part of the first question.
The predictive and allocation questions remain open and require a newly frozen
protocol.
