# Expert Routing on Long-Tailed CIFAR-100

## Abstract

This project tests whether predictions from differently trained classifiers can
be combined to improve recognition of rare classes in CIFAR-100-LT at
imbalance ratio 100. It compares uniform logit averaging, fixed expert
mixtures, and routers that use inference-time features. The repository reports
two distinct tracks: a historical three-seed full-data benchmark on the
balanced CIFAR-100 test set, and nested out-of-fold (OOF) development with one
locked outer-fold evaluation. Their metrics use different populations and
must not be compared directly.

The locked residual Ridge candidate exceeded outer-fold uniform logits by
point estimate, but its paired intervals include zero and two frozen
fixed-weight references exceeded it on both Balanced Accuracy and Tail
accuracy. The prespecified expansion gate failed; no adaptive advantage has
been established.

## Problem statement and objectives

Long-tailed training data contains many examples of common classes and few
examples of rare classes. A model can perform well on frequent classes while
missing rare ones. We study whether experts trained with different objectives
make sufficiently complementary predictions, and whether inference-time
information can identify which expert contributions to use.

The primary criterion is improvement in both Balanced Accuracy (BA) and Tail
accuracy over uniform logit averaging, under a declared evaluation protocol.
BA is mean per-class recall. Head, Medium, and Tail metrics are also macro
recall over the canonical class groups. The original balanced test set is
reserved for historical full-data evaluation and has already been accessed;
it is excluded from OOF router development.

## Pipeline architecture

1. **Define the data and metrics.** Use the canonical 10,847-image
   CIFAR-100-LT training population at IR=100. Obtain the 35 Head, 35 Medium,
   and 30 Tail classes through the shared class-group implementation.
2. **Train four experts.** Train ResNet-32 models with CE, logit adjustment
   (LAL, `tau = 1`), BalancedSoftmax, and Mixup (`alpha = 1`). Full-data
   runs use seeds 78, 88, and 1034, have no validation split, and report final
   checkpoints.
3. **Measure full-data baselines.** Evaluate the full-data experts on the
   balanced CIFAR-100 test set and compare uniform logits with parameter-free
   alternatives. These are historical test results, not router-development
   data.
4. **Build held-out OOF predictions.** For outer fold 0, train inner-fold
   experts so each OOF prediction comes from a model that excluded that image.
   Inner folds 1–3 provide 6,507 router-development rows; inner fold 0 has
   2,170 selection rows and had earlier descriptive exposure in Task 3C.
5. **Compare mixtures and routing signals.** Evaluate fixed convex logit
   mixtures and Ridge predictors on the OOF development rows. Label-dependent
   hard-selection and soft-mixture oracles describe feasibility; they are not
   deployable routers.
6. **Lock one candidate.** The staged study selected confidence-only residual
   Ridge around `fixed_007`, without Sinkhorn/optimal transport (OT), before
   loading the outer-evaluation artifacts.
7. **Evaluate the lock once.** The candidate was refit on all four inner OOF
   folds and evaluated on 2,170 held-out outer-fold-0 rows. This fold is
   consumed and cannot select a replacement method. The original balanced
   test set was not accessed for this OOF evaluation.

## Key findings and contributions

- On the historical full-data track, uniform logit averaging is stronger
  than the valid parameter-free probability and confidence rules on BA and
  Tail.
- On the 6,507-row OOF development partition, several fixed convex mixtures
  improve both BA and Tail over uniform logits.
- A label-dependent soft-mixture feasibility analysis reached 55.5569% macro
  feasibility and 26.3644% Tail feasibility. This shows possible headroom but
  not the ability to predict useful weights.
- Full 13-feature Ridge reduced contribution-prediction error in every one
  of 15 matched settings, including Tail, while its primary classification
  result was below confidence-only Ridge: 36.10% / 6.55% versus 36.55% /
  7.38% BA / Tail.
- Frozen-price Sinkhorn failed its development gate. The locked no-OT Ridge
  candidate improved over outer uniform by point estimate, but
  `fixed_007` and `fixed_010` each exceeded it on both metrics. The
  fixed-reference expansion gate failed.

## Results and evidence

All entries are percentages. Full-data results summarize three seeds on the
balanced test set. OOF development metrics use inner folds 1–3 (6,507 rows);
outer metrics use one held-out fold (2,170 rows). These populations and
protocols differ, so the tables are not directly comparable.

### Full-data test benchmark

| Method | BA | Tail |
|:--|--:|--:|
| CE expert | 37.69 ± 0.73 | 8.67 ± 0.42 |
| LAL expert | 42.43 ± 1.53 | 23.49 ± 1.71 |
| BalancedSoftmax expert | 41.34 ± 0.78 | 22.60 ± 0.39 |
| Mixup expert | 38.69 ± 0.61 | 5.69 ± 0.34 |
| Uniform logits | 46.98 ± 0.69 | 18.76 ± 0.86 |
| Probability average | 45.95 ± 0.55 | 18.70 ± 0.50 |
| Confidence selection | 44.27 ± 0.46 | 18.69 ± 0.36 |

The historical TTA BA/Tail row is invalid following a verified augmentation
implementation defect; it is retained only for provenance. Historical
routing calibration values for Uniform, Confidence, and TTA are also invalid.
See [full-data results](docs/results.md) for the full benchmark and audit
details.

### OOF development partition

| Method | BA | Tail |
|:--|--:|--:|
| Uniform logits | 35.9328 | 7.0094 |
| `fixed_006` | 37.2134 | 9.9115 |
| `fixed_007` | 36.4403 | 14.7290 |
| `fixed_010` | 36.4236 | 14.5055 |
| Confidence-only Ridge | 36.5502 | 7.3797 |
| Full 13-feature Ridge | 36.10 | 6.55 |

These are development results, not independent test performance.

### Locked outer-fold-0 evaluation

| Method | BA | Head | Medium | Tail | Sample accuracy |
|:--|--:|--:|--:|--:|--:|
| Uniform original logits | 42.7006 | 67.3885 | 41.0419 | 15.8333 | 63.9171 |
| Locked residual Ridge, no OT | 44.7035 | 68.4018 | 44.7986 | 16.9444 | 65.1613 |
| `fixed_006` | 44.6348 | 68.1301 | 44.6361 | 17.2222 | 65.1613 |
| `fixed_007` | 44.7248 | 65.3856 | 46.4473 | 18.6111 | 62.7650 |
| `fixed_010` | 44.7298 | 64.1379 | 43.6616 | 23.3333 | 61.8433 |

The Ridge-minus-uniform differences were +2.0028 BA points
(paired 95% interval −0.4956 to +4.5771) and +1.1111 Tail points
(−5.0000 to +7.7778). Both intervals include zero. Frozen
`fixed_007` and `fixed_010` each exceeded Ridge on both target metrics.

## Limitations

The full-data results are historical, and the balanced test set is not an
untouched confirmation set. The completed OOF evidence uses one expert seed
and one outer fold; inner experts share training dependence, and inner fold 0
was previously inspected descriptively. The five-fold, three-seed OOF matrix
and new specialized experts have not been run. No adaptive router has passed
the frozen success gates.

## Documentation

- [Protocol](docs/protocol.md): canonical data roles, folds, evaluation
  hygiene, artifact integrity, router contract, and diagnostic definitions.
- [Full-data results](docs/results.md): authoritative three-seed historical
  benchmark on the balanced test set.
- [OOF results](docs/oof-results.md): concise Task 3C–3F synthesis and the
  locked outer-fold evidence.
- [Research](docs/research.md): measured constraints, open questions,
  literature, and references.
- [Reproduction](docs/reproduction.md): commands and artifact locations for
  the completed tracks and studies.
