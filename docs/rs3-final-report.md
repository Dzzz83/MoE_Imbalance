# Ridge/Sinkhorn three-seed report

**Study:** `ridge_sinkhorn_3seed_v1`  
**Report date:** 2026-10-03  
**Evidence label:** Retrospective nested-OOF evidence; not independent confirmation.

## Conclusion

The completed 300-job study does not establish an adaptive routing advantage.
None of the five adaptive methods improved both BA and Tail accuracy over
uniform logits on all three seeds, and all five paired BA intervals include
zero. Fixed-mixture non-domination is a separate criterion and cannot rescue
a failed uniform-logit gate. The detailed prespecified rules are in the frozen
[study plan](PLAN.md).

The study evaluates the same canonical training population across folds and
seeds. Earlier development exposed inner-fold-0 labels, and seed 78 / outer
fold 0 had already been evaluated in the separate `ridge_sinkhorn_v3` study.
The full matrix therefore supplies retrospective evidence, not confirmation
on a new population.

The v1 evaluation did not load or score original balanced-test examples or
labels. Constructing its fold manager with `CIFAR100(train=True)` caused the
installed torchvision integrity check to compute MD5 over both extracted
split files. Thus `cifar-100-python/test` was read for checksum verification
only; its pickle was not unpickled, and its examples and labels were not loaded. The study lock's `test_accessed=false`
records that test examples were not evaluated; it does not record this
checksum-only file read. Inner diagnostics used saved manifests and OOF
predictions without loading CIFAR data.

## Training and solver integrity

The frozen health audit checked all 300 inventory jobs and found no reported
problems, unavailable histories, or runs with an unrecorded gradient norm.

| Stage | Native jobs | Reused jobs | Checked |
|:--|--:|--:|--:|
| Inner | 224 | 16 | 240 |
| Outer | 56 | 4 | 60 |
| **Total** | **280** | **20** | **300** |

All histories contain the exact 200-epoch sequence, finite losses, learning
rates, and gradient norms, expected LR transitions at epochs 161 and 181,
and a final checkpoint. Training accuracy is unavailable in all 75 Mixup
histories (70 native, five reused); those `null` entries are a known logging
omission, not a failed health check.

The 15 immutable locks contain 180 selected cross-fit Sinkhorn records and
30 final fitted states. All 210 report convergence. The largest saved
log-price residual is `9.991e-11` for cross-fit and `9.966e-11` for fitted
states; the largest row residual is `2.221e-16` in each group. Target-marginal
residuals up to 0.2851 and 0.2835 are allocation distances in the relaxed-OT
formulation, not solver failures.

## Outer evaluation

Each seed pools five disjoint outer folds covering 10,847 predictions.
Metrics are percentages. The reported BA spread is the **population standard
deviation** across the three seed-level results (`ddof=0`), matching the
frozen evaluator.

The strict performance rule requires both BA and Tail accuracy to exceed
uniform logits for every seed. It is assessed separately from fixed-mixture
non-domination, where a fixed mixture with BA and Tail both greater than or
equal to an adaptive method is disqualifying; ties count. The study used
10,000 paired, class-then-sample bootstrap replicates with seed `20260924`.
Intervals are conditional on the fitted models and do not measure retraining
variability. Classes are resampled within Head/Medium/Tail groups, then
examples within each sampled class; paired rows are shared across methods and
the three fixed seed predictions.

| Method | BA mean ± population SD | Head | Medium | Tail | Sample accuracy | All-seed BA + Tail |
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

All five BA-delta intervals against uniform include zero:

| Adaptive method | Mean ΔBA (95% interval) | Mean ΔTail (95% interval) |
|:--|:--|:--|
| `contribution_ridge` | −0.165 [−0.852, 0.357] | −1.589 [−3.674, −0.253] |
| `contribution_ridge_sinkhorn` | 0.310 [−0.354, 0.935] | 0.216 [−1.669, 1.883] |
| `residual_ridge` | 0.099 [−0.681, 0.858] | 1.219 [−0.959, 3.270] |
| `residual_ridge_sinkhorn` | 0.452 [−0.276, 1.162] | 1.300 [−0.686, 3.156] |
| `selective_residual_ridge_sinkhorn` | 0.422 [−0.308, 1.141] | 1.385 [−0.627, 3.250] |

Deltas and intervals are percentage points. By seed, four of five methods
passed the two-metric uniform gate on seed 78, none passed on seed 88, and
three passed on seed 1034. Exact per-seed method comparisons and the separate
fixed-mixture non-domination flags are in the
[canonical results](../.rs3-server/runs/ridge_sinkhorn_3seed_v1/study_analysis/results.json).

The earlier locked seed-78 / outer-fold-0 candidate from `ridge_sinkhorn_v3`
also remains unchanged: its residual Ridge point estimate exceeded outer
uniform, but both paired intervals included zero and `fixed_007` and
`fixed_010` exceeded it on both metrics. That study's expansion gate failed;
the later retrospective matrix does not restore an independent test.

## Inner diagnostics

Inner diagnostics cover all 15 seed/fold pairs and keep **Expert OOF**,
**Selection CV**, and **Router fit set** as distinct evidence populations.
For the summary below, five fold rows are first aggregated within each seed;
the mean and **sample standard deviation** (`ddof=1`) are then calculated
across three seed means. These values are fractions on a 0–1 scale, unlike
the percentage outer results above.

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

The selection results are used evidence, not an untouched confirmation set.
Router fit-set scores are descriptive fit-population measurements, not
generalization estimates. A hard-selection oracle is label-dependent and is
neither an inference method nor a bound on soft mixtures. Sinkhorn marginal
discrepancy describes allocation against its target; solver health is judged
from the saved convergence flag and log-price/row residuals.

## Provenance and detailed records

The training source is clean commit
`e7357a7c5028c87f49739d0b390a5e4e6a258fba`; the study freeze hash is
`09c7d9921a7d7c212192b7d822b99005ea0a9f4b1c324c05c350330936d0d2aa`, and
the resolved study configuration hash is
`d9b37ab823c505eafd6f282f0dfd0a821b7a03e839b6229175b5d514f263f929`.
The evaluator checkout at
`/mnt/hdd2/phatht/phat/MoE_Imbalance-evaluation-provenance` used commit
`64401e8e4cf076d618948185a2d008ef650d68c2`; the frozen training source remained
`e7357a7c5028c87f49739d0b390a5e4e6a258fba`. All 15 fold payloads and saved arrays
matched the baseline; aggregate JSON and Markdown matched byte-for-byte.
Completion records hash-link all 300 artifacts, fold memberships, settings,
environments, and outputs. The 917 required inner-snapshot files (21
study/lock records and 896 native payload files) were verified with no
missing files or blob mismatches; 16 historical payloads remain a separate
read-only input.

Key audit and result records:

- [Training-health audit](../.rs3-server/logs/training-health-audit-20261003.json)
  and [its provenance](../.rs3-server/logs/training-health-audit-provenance-20261003.json)
- [Sinkhorn solver-health audit](../.rs3-server/logs/solver-health-audit-20261003.json)
- [Outer-result provenance](../.rs3-server/logs/outer-result-provenance-final-20261003.json)
- [Verified inner-snapshot audit](../.rs3-server/logs/inner-diagnostics-snapshot-verification.json)
- [Formal evaluation completion](../.rs3-server/runs/ridge_sinkhorn_3seed_v1/evaluation_runs/rs3-fixed-v1/evaluate.complete.json)
- [Formal report completion](../.rs3-server/runs/ridge_sinkhorn_3seed_v1/evaluation_runs/rs3-fixed-v1/report.complete.json)
- [Formal detailed report](../.rs3-server/runs/ridge_sinkhorn_3seed_v1/evaluation_runs/rs3-fixed-v1/report/report.md)
- [Inner diagnostics report](../.rs3-server/runs/ridge_sinkhorn_3seed_v1/study_analysis/diagnostics/rs3-inner-v2/report.md)
- [Inner diagnostics provenance](../.rs3-server/runs/ridge_sinkhorn_3seed_v1/study_analysis/diagnostics/rs3-inner-v2/provenance.json)
- [Independent numerical and provenance review](../.rs3-server/logs/rs3-fixed-v1-equivalence-review.json)

Reproduction commands and artifact locations are in
[reproduction.md](reproduction.md). The formal evaluator command and
provenance contract are in [reproduction.md](reproduction.md#evaluation-id-and-provenance).
