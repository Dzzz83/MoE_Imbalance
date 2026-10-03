# Ridge/Sinkhorn full-matrix report

**Study:** `ridge_sinkhorn_3seed_v1`  
**Report date:** 2026-10-03  
**Evidence label:** Retrospective nested-OOF evidence; not independent confirmation.

This report brings together the 300-run training health audit, the locked
outer-fold evaluation, and descriptive inner-study diagnostics. The data
population and some inner-fold history informed earlier development. The
results do not establish performance on a new population. No original
balanced CIFAR-100 test examples or labels were loaded or scored. The frozen
evaluation constructs its fold manager from `CIFAR100(train=True)`. In the
installed torchvision version, that constructor's integrity check computes
MD5 over both the training and test split files before it loads the training
batches. Thus the extracted test split file
(`cifar-100-python/test`) was read for MD5 verification only; that pickle was
not unpickled and its examples or labels were not accessed. The study lock
records `test_accessed=false` to indicate that test examples were not
evaluated; it does not track torchvision's checksum-only file read. The inner
diagnostics read their pinned manifests and inner prediction payloads without
loading CIFAR data.

## Training health

The matrix audit checked all 300 inventory jobs. It found no reported
problems, unavailable histories, or runs with an unrecorded gradient norm.

| Stage | Native jobs | Reused jobs | Checked |
|:--|--:|--:|--:|
| Inner | 224 | 16 | 240 |
| Outer | 56 | 4 | 60 |
| **Total** | **280** | **20** | **300** |

The audit used `RunHealthChecker` with the frozen 200-epoch recipe and also
required the exact epoch sequence 1–200, finite learning rates and gradient
norms, and finite training-accuracy values wherever recorded. The
checker validates the LR changes at epochs 161 and 181, finite loss, loss
decrease over training, and the final checkpoint. All 300 histories included
loss, learning-rate, and gradient-norm records. Training accuracy is
unavailable in all 75 Mixup histories (70 native and 5 reused): each has
`train_acc: null` for all 200 epochs, totaling 15,000 absent entries. This
known field omission was not treated as a failed health check; it limits the
training-accuracy history available for Mixup.

The original audit JSON is preserved byte-for-byte at
[training-health-audit-20261003.json](../.rs3-server/logs/training-health-audit-20261003.json).
Its [provenance record](../.rs3-server/logs/training-health-audit-provenance-20261003.json)
links the audit hash to the frozen study and matrix manifests. The source audit
did not include its generator command; the exact audit method is recorded in
the provenance sidecar.

## Sinkhorn solver health

The 15 immutable locks contain 180 selected cross-fit Sinkhorn diagnostic
records and 30 final fitted Sinkhorn states. All 210 records report
`converged=true`. The largest saved log-price residual is `9.991e-11` for
cross-fit records and `9.966e-11` for fitted states; the largest row residual
is `2.221e-16` in both groups.

The largest target-marginal residuals are `0.2851` for cross-fit records and
`0.2835` for fitted states. These are allocation distances in the relaxed-OT
formulation. They are not solver failures; convergence is assessed from the
saved convergence flag and log-price/row residuals. The complete summary and
lock hashes are in
[solver-health-audit-20261003.json](../.rs3-server/logs/solver-health-audit-20261003.json).

## Outer evaluation

Each training seed pools its five disjoint outer folds, covering 10,847
predictions. Balanced accuracy is mean per-class recall. Head, Medium, and
Tail are also macro recall within the canonical training-count groups
(Head `>=100`, Medium `20–99`, Tail `<20`); ordinary accuracy is
sample-weighted.

The study evaluator reports each metric's mean and **population standard
deviation across the three seed-level results** (`ddof=0`). This is the
unchanged calculation in the original evaluator and formal rerun. The five folds and three
seeds share the same underlying population, so the 15 seed/fold pairs are not
independent replicates.

The predeclared performance rule is evaluated separately for each of the five
adaptive methods: both BA and Tail accuracy must exceed uniform logits for
every seed. Fixed-mixture non-domination is a separate check against
`fixed_006`, `fixed_007`, `fixed_010`, and `fixed_011` on both BA and Tail.
The frozen check treats a fixed mixture as disqualifying when its BA is
greater than or equal to the adaptive method's BA **and** its Tail accuracy
is greater than or equal to the adaptive method's Tail accuracy; ties count.
The study uses 10,000 paired, class-then-sample bootstrap replicates with seed
`20260924`; intervals are conditional on the fitted models and do not measure
retraining variability. Each replicate resamples classes within the H/M/T
groups and then samples examples within each selected class, applying the same
sampled rows across methods and the three fixed seed predictions; it does not
resample training seeds.

The formal separate-provenance `study evaluate` and `study report` commands
completed successfully. All 15 fold JSON payloads and every saved NumPy array
match the baseline exactly. Aggregate JSON and Markdown match byte-for-byte,
including the unchanged 10,000-replicate bootstrap.

**None of the five adaptive methods passes the strict all-seed BA + Tail
improvement rule.** All five BA-delta 95% intervals against uniform logits
include zero. These results do not support a consistent improvement over
uniform logits. No settings were tuned using outer results.

All metric values below are percentages. SD is the population SD across
three seeds; deltas and interval endpoints are percentage points.

| Method | BA mean ± population SD | Head | Medium | Tail | Ordinary accuracy | All-seed BA + Tail gate |
|:--|--:|--:|--:|--:|--:|:--|
| `contribution_ridge` | 42.245 ± 0.131 | 68.766 | 40.372 | 13.489 | 65.696 | Fail |
| `contribution_ridge_sinkhorn` | 42.719 ± 0.366 | 67.634 | 41.313 | 15.293 | 64.752 | Fail |
| `residual_ridge` | 42.509 ± 0.466 | 66.522 | 40.964 | 16.296 | 63.677 | Fail |
| `residual_ridge_sinkhorn` | 42.861 ± 0.706 | 66.824 | 41.600 | 16.377 | 64.070 | Fail |
| `selective_residual_ridge_sinkhorn` | 42.832 ± 0.673 | 66.777 | 41.489 | 16.463 | 64.021 | Fail |
| `uniform_logit` | 42.410 ± 0.315 | 67.663 | 40.584 | 15.077 | 64.697 | — |
| `uniform_probability` | 41.556 ± 0.345 | 66.967 | 39.481 | 14.329 | 63.969 | — |
| `fixed_006` | 43.062 ± 0.405 | 68.116 | 41.783 | 15.323 | 65.185 | — |
| `fixed_007` | 42.251 ± 0.955 | 64.047 | 40.357 | 19.033 | 61.442 | — |
| `fixed_010` | 42.566 ± 0.663 | 64.256 | 40.606 | 19.548 | 61.624 | — |
| `fixed_011` | 41.802 ± 0.839 | 60.404 | 39.184 | 23.153 | 58.200 | — |
| `residual_anchor` | 42.251 ± 0.955 | 64.047 | 40.357 | 19.033 | 61.442 | — |
| `prior_only_control` | 42.677 ± 0.186 | 66.993 | 41.154 | 16.084 | 64.174 | — |

Fixed-mixture non-domination is reported separately in the last column.
A pass there does not override a failed uniform-logit gate. Values are
transcribed from canonical `results.json`.

| Seed | Adaptive method | BA | Tail | ΔBA vs uniform logits | ΔTail vs uniform logits | BA + Tail gate | Fixed-mixture non-domination |
|--:|:--|--:|--:|--:|--:|:--|:--|
| 78 | `contribution_ridge` | 42.123 | 13.143 | 0.068 | -1.084 | Fail | Fail |
| 78 | `contribution_ridge_sinkhorn` | 42.905 | 14.972 | 0.849 | 0.744 | Pass | Pass |
| 78 | `residual_ridge` | 42.721 | 16.029 | 0.665 | 1.802 | Pass | Fail |
| 78 | `residual_ridge_sinkhorn` | 43.330 | 16.084 | 1.275 | 1.857 | Pass | Pass |
| 78 | `selective_residual_ridge_sinkhorn` | 43.300 | 16.084 | 1.245 | 1.857 | Pass | Pass |
| 88 | `contribution_ridge` | 42.185 | 13.672 | -0.168 | -2.076 | Fail | Fail |
| 88 | `contribution_ridge_sinkhorn` | 42.207 | 15.736 | -0.146 | -0.011 | Fail | Pass |
| 88 | `residual_ridge` | 41.863 | 15.690 | -0.490 | -0.058 | Fail | Fail |
| 88 | `residual_ridge_sinkhorn` | 41.864 | 15.690 | -0.490 | -0.058 | Fail | Fail |
| 88 | `selective_residual_ridge_sinkhorn` | 41.881 | 15.690 | -0.472 | -0.058 | Fail | Fail |
| 1034 | `contribution_ridge` | 42.427 | 13.651 | -0.393 | -1.606 | Fail | Fail |
| 1034 | `contribution_ridge_sinkhorn` | 43.046 | 15.171 | 0.225 | -0.086 | Fail | Fail |
| 1034 | `residual_ridge` | 42.943 | 17.170 | 0.122 | 1.913 | Pass | Fail |
| 1034 | `residual_ridge_sinkhorn` | 43.391 | 17.357 | 0.570 | 2.100 | Pass | Fail |
| 1034 | `selective_residual_ridge_sinkhorn` | 43.315 | 17.614 | 0.494 | 2.356 | Pass | Fail |

| Adaptive method vs uniform logits | Mean ΔBA | 95% interval for ΔBA | Mean ΔTail | 95% interval for ΔTail |
|:--|--:|:--|--:|:--|
| `contribution_ridge` | -0.165 | [-0.852, 0.357] | -1.589 | [-3.674, -0.253] |
| `contribution_ridge_sinkhorn` | 0.310 | [-0.354, 0.935] | 0.216 | [-1.669, 1.883] |
| `residual_ridge` | 0.099 | [-0.681, 0.858] | 1.219 | [-0.959, 3.270] |
| `residual_ridge_sinkhorn` | 0.452 | [-0.276, 1.162] | 1.300 | [-0.686, 3.156] |
| `selective_residual_ridge_sinkhorn` | 0.422 | [-0.308, 1.141] | 1.385 | [-0.627, 3.250] |

The detailed canonical report retains all per-seed metrics, selections,
diagnostics, expert masses, paired comparisons, and intervals.

## Inner diagnostics

The diagnostics run covers all 15 seed/outer-fold pairs. It keeps three
evidence populations distinct: Expert OOF, Selection CV, and Router fit set.
It does not read outer predictions or labels, fit a replacement router, or
select a method. Inner summaries first aggregate the five outer-fold rows
within each seed, then report the mean and **sample standard deviation** of
the three seed means (`ddof=1`). The 15 pairs are not treated as independent
replicates.

The full diagnostics run completed for all 15 pairs and reports
`complete_matrix=true` (240 inner jobs: 224 native and 16 historical). Expert
OOF, Selection CV, and Router fit set are reported separately. The selection
CV and router-fit summary fractions below are on a 0–1 scale. In these tables,
Head/Medium/Tail are macro recall within the same canonical training-count
groups (Head `>=100`, Medium `20–99`, Tail `<20`); ordinary accuracy is
sample-weighted. Router-fit rows are descriptive fit-set measurements rather
than generalization estimates.

| Inner evidence / method | BA mean | Tail mean |
|:--|--:|--:|
| Expert OOF / CE | 0.29462 | 0.04661 |
| Expert OOF / LAL | 0.31901 | 0.13651 |
| Expert OOF / BalancedSoftmax | 0.31543 | 0.12967 |
| Expert OOF / Mixup | 0.32025 | 0.02845 |
| Selection CV / `contribution_ridge` | 0.37181 | 0.09063 |
| Selection CV / `contribution_ridge_sinkhorn` | 0.37644 | 0.11416 |
| Selection CV / `residual_ridge` | 0.37477 | 0.12287 |
| Selection CV / `residual_ridge_sinkhorn` | 0.37634 | 0.12336 |
| Selection CV / `selective_residual_ridge_sinkhorn` | 0.37661 | 0.12462 |
| Selection CV / `prior_only_control` | 0.37443 | 0.11959 |

Across fold profiles within seeds, the mean L1 change was 0.10329 for
`contribution_ridge`, 0.21821 for `contribution_ridge_sinkhorn`, 0.11927 for
`residual_ridge`, 0.18457 for `residual_ridge_sinkhorn`, and 0.18634 for
`selective_residual_ridge_sinkhorn`. The overall 0.09045 mean includes all 13
reported methods, including fixed controls; it is not a learned-router-only
stability measure. Mean normalized allocation entropy is 0.83573 across the
same 13-method scope.

At least one expert was correct on 0.72893 of inner-OOF samples; all four
experts were wrong on 0.27107. The label-dependent hard-selection oracle had
BA 0.50162. Its 0.17852 gap is averaged against the strongest individual
expert within each seed/fold pair, then across folds within seed and across
seeds; it is a diagnostic opportunity, not deployable performance or a
soft-mixture bound. Mean expert class-profile disagreement was 0.61202.

The mean L1 distance from the selected prior-only control was 0.34405, and
the mean maximum absolute Sinkhorn marginal residual was 0.15493. Mean
selective gate was 0.66821. The marginal residual measures allocation
distance in relaxed OT and is distinct from solver convergence health above.
These are descriptive inner-population results, not outer or original-test
performance.

Matplotlib was absent in the frozen runtime, so figure export was skipped with
`--no-figures`; JSON, CSV, and Markdown outputs were produced. No plotting
dependency was installed in that environment. The complete outputs are
linked below and include fold-level tables and provenance.

## Provenance

The immutable training source is clean commit
`e7357a7c5028c87f49739d0b390a5e4e6a258fba`; the freeze hash is
`09c7d9921a7d7c212192b7d822b99005ea0a9f4b1c324c05c350330936d0d2aa`. The
freeze manifest's `study_config_sha256` is
`d9b37ab823c505eafd6f282f0dfd0a821b7a03e839b6229175b5d514f263f929`.
The main checkout integrated diagnostics in merge commit `85fe4e2` while
preserving master’s `.gitignore` and the frozen training code.

The server was initially provisioned from result snapshot
`eeb14287484dc5a05e73ffdeed14a0f6ff21d58a`, which held 12 native outer jobs
and four historical outer reuses. The server then produced 44 additional
native outer jobs. Final outer-result provenance is preserved in the [baseline outer
provenance](../.rs3-server/logs/outer-result-provenance-final-20261003.json). It
distinguishes the initial snapshot from the subsequent server outputs. The original
pre-aggregate provenance remains unchanged.

For inner diagnostics, all 917 required files (21 study/lock files and 896
native payload files) were verified against the Git trees for both the
documented result pin `5993d26eead9575160886d6130dce744ee9e02e6` and the
server’s initial snapshot `eeb14287484dc5a05e73ffdeed14a0f6ff21d58a`. There
were no missing files or blob mismatches. The 16 historical inner inputs stay
under the separate read-only `rs3` reuse root. The diagnostic runner uses a
clean detached analysis checkout at `85fe4e2` and records the input hashes and
analysis identity in its provenance output.

The formal evaluation and report used evaluator commit
`64401e8e4cf076d618948185a2d008ef650d68c2`. Completion records identify all 300
artifacts, exact fold memberships, scientific settings, runtime versions, commands, and
output hashes. The report is hash-linked to all 30 fold files and both evaluation
records. The baseline remains unchanged.

- [Formal evaluation completion](../.rs3-server/runs/ridge_sinkhorn_3seed_v1/evaluation_runs/rs3-fixed-v1/evaluate.complete.json)
- [Formal report completion](../.rs3-server/runs/ridge_sinkhorn_3seed_v1/evaluation_runs/rs3-fixed-v1/report.complete.json)
- [Formal aggregate results](../.rs3-server/runs/ridge_sinkhorn_3seed_v1/evaluation_runs/rs3-fixed-v1/report/results.json)
- [Formal detailed report](../.rs3-server/runs/ridge_sinkhorn_3seed_v1/evaluation_runs/rs3-fixed-v1/report/report.md)
- [Independent numerical/provenance review](../.rs3-server/logs/rs3-fixed-v1-equivalence-review.json)
- [Fresh 300-history health review](../.rs3-server/logs/rs3-fixed-v1-health-review.json)

Historical logs, audit records, and diagnostics outputs are linked here:

- [Frozen evaluate log](../.rs3-server/logs/study-evaluate-20261003.log)
- [Full inner diagnostics log](../.rs3-server/logs/inner-diagnostics-full-20261003.log)
- [Inner diagnostics report](../.rs3-server/runs/ridge_sinkhorn_3seed_v1/study_analysis/diagnostics/rs3-inner-v2/report.md)
- [Inner diagnostics metrics](../.rs3-server/runs/ridge_sinkhorn_3seed_v1/study_analysis/diagnostics/rs3-inner-v2/analysis.json)
- [Inner diagnostics provenance](../.rs3-server/runs/ridge_sinkhorn_3seed_v1/study_analysis/diagnostics/rs3-inner-v2/provenance.json)
- [Inner diagnostics seed aggregates](../.rs3-server/runs/ridge_sinkhorn_3seed_v1/study_analysis/diagnostics/rs3-inner-v2/seed_aggregates.csv)
- [Inner diagnostics selection CV table](../.rs3-server/runs/ridge_sinkhorn_3seed_v1/study_analysis/diagnostics/rs3-inner-v2/selection_cv_metrics.csv)
- [Inner diagnostics expert OOF table](../.rs3-server/runs/ridge_sinkhorn_3seed_v1/study_analysis/diagnostics/rs3-inner-v2/expert_oof_metrics.csv)
- [Inner diagnostics complementarity table](../.rs3-server/runs/ridge_sinkhorn_3seed_v1/study_analysis/diagnostics/rs3-inner-v2/complementarity.csv)
- [Inner diagnostics frequency associations](../.rs3-server/runs/ridge_sinkhorn_3seed_v1/study_analysis/diagnostics/rs3-inner-v2/frequency_associations.csv)
- [Inner diagnostics oracle opportunity table](../.rs3-server/runs/ridge_sinkhorn_3seed_v1/study_analysis/diagnostics/rs3-inner-v2/oracle_opportunity.csv)
- [Inner diagnostics per-class specialization](../.rs3-server/runs/ridge_sinkhorn_3seed_v1/study_analysis/diagnostics/rs3-inner-v2/per_class_specialization.csv)
- [Inner diagnostics prior diagnostics](../.rs3-server/runs/ridge_sinkhorn_3seed_v1/study_analysis/diagnostics/rs3-inner-v2/prior_diagnostics.csv)
- [Inner diagnostics router class profiles](../.rs3-server/runs/ridge_sinkhorn_3seed_v1/study_analysis/diagnostics/rs3-inner-v2/router_class_profiles.csv)
- [Inner diagnostics router fit-set metrics](../.rs3-server/runs/ridge_sinkhorn_3seed_v1/study_analysis/diagnostics/rs3-inner-v2/router_fit_set_metrics.csv)
- [Inner diagnostics router group allocations](../.rs3-server/runs/ridge_sinkhorn_3seed_v1/study_analysis/diagnostics/rs3-inner-v2/router_group_allocations.csv)
- [Inner diagnostics router stability table](../.rs3-server/runs/ridge_sinkhorn_3seed_v1/study_analysis/diagnostics/rs3-inner-v2/router_stability.csv)
- [Inner diagnostics router weight profiles](../.rs3-server/runs/ridge_sinkhorn_3seed_v1/study_analysis/diagnostics/rs3-inner-v2/router_weight_profiles.csv)
- [Inner diagnostics solver adjustments](../.rs3-server/runs/ridge_sinkhorn_3seed_v1/study_analysis/diagnostics/rs3-inner-v2/sinkhorn_adjustments.csv)
- [Verified inner snapshot audit](../.rs3-server/logs/inner-diagnostics-snapshot-verification.json)
- [Training health provenance](../.rs3-server/logs/training-health-audit-provenance-20261003.json)
- [Solver health audit](../.rs3-server/logs/solver-health-audit-20261003.json)
- [Canonical aggregate report](../.rs3-server/runs/ridge_sinkhorn_3seed_v1/study_analysis/report.md)
- [Canonical machine-readable results, including per-seed metrics](../.rs3-server/runs/ridge_sinkhorn_3seed_v1/study_analysis/results.json)

## Reproduction and validation

The original frozen `study evaluate` command completed in approximately
74 minutes. The original `study report` command was interrupted because of
redundant validation. Baseline aggregation completed through the documented
service runner; the interrupted report CLI is not claimed to have completed.
Its original provenance records remain preserved.

The formal rerun used clean evaluator commit `64401e8e4cf076d618948185a2d008ef650d68c2` in
`/mnt/hdd2/phatht/phat/MoE_Imbalance-evaluation-provenance`. Training identity remains the frozen commit.
No training or lock creation was performed. Use a fresh evaluation ID when
reproducing the run; existing IDs cannot be overwritten.

```bash
cd /mnt/hdd2/phatht/phat/MoE_Imbalance-evaluation-provenance
RS3_EVALUATION_ID=rs3-reproduction-v1
PYTHONDONTWRITEBYTECODE=1 /mnt/hdd2/phatht/phat/MoE_Imbalance/.rs3-server/venv/bin/python -m expert_method \
  --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile /mnt/hdd2/phatht/phat/MoE_Imbalance/.rs3-server/server-gpu2.yaml \
  study evaluate --evaluation-id "$RS3_EVALUATION_ID"
PYTHONDONTWRITEBYTECODE=1 /mnt/hdd2/phatht/phat/MoE_Imbalance/.rs3-server/venv/bin/python -m expert_method \
  --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --profile /mnt/hdd2/phatht/phat/MoE_Imbalance/.rs3-server/server-gpu2.yaml \
  study report --evaluation-id "$RS3_EVALUATION_ID"
```

Formal `evaluate` subprocess runtime: **207.83 seconds** ([process
record](../.rs3-server/logs/rs3-fixed-v1-evaluate-20261003T135050Z.process.json)).
Formal `report` subprocess runtime: **390.46 seconds** ([process
record](../.rs3-server/logs/rs3-fixed-v1-report-20261003T135455Z.process.json)).

These are actual subprocess runtimes, measured separately from implementation
and regression-test time. Stage timing in completion records excludes
initial Python startup and pre-start input inspection.

The full inner diagnostics command used the verified pinned materialization,
the original historical `rs3` reuse root, and a fresh output directory:

```bash
cd /tmp/rs3-analysis-diagnostics-85fe4e2
/mnt/hdd2/phatht/phat/MoE_Imbalance/.rs3-server/venv/bin/python \
  -m expert_method.diagnostics \
  --config configs/studies/ridge_sinkhorn_3seed_v1.yaml \
  --artifact-root /tmp/rs3-inner-diagnostics-inputs-20261003T0537Z \
  --reuse-root rs3=/mnt/hdd2/phatht/phat/MoE_Imbalance/.rs3-server/code/artifacts/oof \
  --stage inner \
  --output-root /mnt/hdd2/phatht/phat/MoE_Imbalance/.rs3-server/runs/ridge_sinkhorn_3seed_v1/study_analysis/diagnostics/rs3-inner-v2 \
  --no-figures
```

Earlier checks after diagnostics integration passed: 39 diagnostics and study tests
passed, 2 skipped. The separate evaluator-hardening review passed 83 tests
after repairing two synthetic CLI fixtures to stub training class counts;
the production evaluator was unchanged. Pytest and its support packages were
isolated under `/tmp/rs3-review-test-deps`, outside the frozen runtime
environment. A reviewer independently recomputed all 115 diagnostics metric
summaries from the CSV fold rows (five folds per seed, then three-seed mean and
sample SD) and matched `analysis.json` within `1e-12`.

The separate-provenance implementation was independently reviewed and tested.
Independent affected-suite checks passed 78 tests with two optional skips; the
post-merge provenance suite passed 15 tests (93 distinct passing tests total). The real
CLI rejected dirty evaluators, mixed code/config checkouts, and default frozen-commit
mismatches without creating evaluation outputs. The formal-output review matched all 15
fold payloads and arrays exactly and both aggregate files byte-for-byte. [Regression
evidence](../.rs3-server/logs/rs3-fixed-v1-tests-review.json)

The affected-suite test subprocess took 76.69 seconds; the final post-merge
provenance test subprocess took 7.57 seconds. These are test runtimes, separate
from the production evaluation and reporting runtimes above.
