# Inner Ridge/Sinkhorn diagnostics

This guide describes the read-only descriptive analysis of the saved
`ridge_sinkhorn_3seed_v1` inner predictions and locked router states. The
reader validates the frozen study, all 15 seed/fold locks, the 240 inner jobs,
source hashes, fold memberships, saved reuse decisions, and finite prediction
records before numerical diagnostics. It never reads an outer prediction or
label, trains an expert, fits a router, repeats method selection, or opens the
original CIFAR-100 test set.

The canonical data roles and metric definitions remain in
[protocol.md](protocol.md). The saved-study reproduction sequence is in
[reproduction.md](reproduction.md). The v1 matrix is retrospective nested-OOF
evidence; it is not an independent confirmation set.

## Source identities and inputs

The analysis uses three distinct identities:

| Identity | What it identifies | Required handling |
|:--|:--|:--|
| Training source | The clean source commit recorded by the frozen study and matrix manifests | Preserve the saved values; the analysis checkout may have a different commit. |
| Result artifact snapshot | The immutable native run and lock snapshot, pinned to `5993d26eead9575160886d6130dce744ee9e02e6` | Keep it in a separate detached artifact checkout or a verified materialization. |
| Analysis source | The local checkout containing `expert_method.diagnostics` | Preserve its commit and working tree during a run; `dev/study-analysis` is local and no remote availability is assumed. |

The pinned result snapshot contains 224 native inner jobs and all 15 locks. It
does not contain the historical Task 3C prediction payloads. Those 16
historical references are a separate input mounted under the frozen logical
reuse name `rs3`. A clone of the result snapshot alone is insufficient.

For an extracted or materialized artifact root, the supplied
`.snapshot-provenance.json` must be at `ARTIFACT_ROOT` and record schema
`diagnostics_snapshot.v1` or `artifact_snapshot.v1`, the pinned
`snapshot_commit`, and hashes in `file_hashes`. The pinned Git tree has no
snapshot marker; a detached Git checkout supplies its identity through
`HEAD`. Do not edit a marker to bless changed payloads. Keep the separate
historical root's compatibility materialization record and all 16 sources.

The Task 3C export did not include CE, seed 78, outer fold 0, inner fold 0.
The frozen reuse decision points to the separate historical experiment
`task3b_pilot_ce_s78_o0_i0`, at
`task3b_pilot_ce_s78_o0_i0/expert_CE/seed_78/outer_0/inner_0`. The named
`rs3` root must preserve that source path alongside `task3c_oof/`. If the
audited original is missing from a materialization, restore only that exact
source and let the reader verify its metadata and hashes. Do not synthesize
metadata or recreate files from predictions.

## 1. Prepare the local analysis checkout and environment

Use the existing local `dev/study-analysis` checkout containing this guide.
It is not assumed to be published on `origin`; do not switch branches or
substitute a same-named remote branch. Confirm the checkout, create its
environment, and record its current identity:

```bash
ANALYSIS_ROOT=/home/dzzz83/Documents/code/expert_method
cd "$ANALYSIS_ROOT"
test "$(git branch --show-current)" = "dev/study-analysis"
test -f expert_method/diagnostics/__main__.py
python -m venv .venv
git rev-parse --show-toplevel
git rev-parse HEAD
```

If the checkout lives elsewhere, set `ANALYSIS_ROOT` to that existing local
checkout. Keep this analysis checkout separate from the artifact snapshot and
historical OOF root. The report provenance records the analysis commit, dirty
state, source hashes, and installed package versions.

## 2. Obtain the pinned artifact snapshot and historical inputs

The configured repository remote is
`https://github.com/Dzzz83/MoE_Imbalance.git`. These commands create a separate
detached checkout and build an exact file allowlist from the pinned Git tree.
The allowlist includes the 224 native inner jobs and required study metadata,
and excludes `attempts/`, `outer_eval/`, outer compatibility, logs, and
training histories.

```bash
SNAPSHOT_COMMIT=5993d26eead9575160886d6130dce744ee9e02e6
SNAPSHOT_CHECKOUT=/tmp/rs3-result-snapshot
SNAPSHOT_ALLOWLIST=$(mktemp /tmp/rs3-allowlist.XXXXXX)
git clone --filter=blob:none --no-checkout \
  https://github.com/Dzzz83/MoE_Imbalance.git "$SNAPSHOT_CHECKOUT"
git -C "$SNAPSHOT_CHECKOUT" fetch --filter=blob:none origin "$SNAPSHOT_COMMIT"
git -C "$SNAPSHOT_CHECKOUT" ls-tree -r --name-only "$SNAPSHOT_COMMIT" -- \
  runs/ridge_sinkhorn_3seed_v1 | awk '
  /^runs\/ridge_sinkhorn_3seed_v1\/(fold_manifest\.json|job_manifest\.json|manifests\/study-freeze\.json|reuse_compatibility_inner\.json|study_analysis\/(study_config\.json|study_lock\.json|locks\/seed_(78|88|1034)_outer_[0-4]\.json))$/ { print "/" $0; next }
  /^runs\/ridge_sinkhorn_3seed_v1\/expert_(BalancedSoftmax|CE|LAL|Mixup)\/seed_(78|88|1034)\/outer_[0-4]\/inner_[0-3]\/(checkpoints\/[^/]+_final\.pt|predictions\.json|resolved_config\.json|run_metadata\.json)$/ { print "/" $0 }
' > "$SNAPSHOT_ALLOWLIST"
test "$(wc -l < "$SNAPSHOT_ALLOWLIST")" -eq 917
if grep -E '(^|/)(attempts|outer_eval)(/|$)|reuse_compatibility_outer\.json|training_history\.json|execution\.log' "$SNAPSHOT_ALLOWLIST"; then
  echo "unexpected non-diagnostic artifact in snapshot allowlist" >&2
  exit 1
fi
git -C "$SNAPSHOT_CHECKOUT" sparse-checkout init --no-cone
git -C "$SNAPSHOT_CHECKOUT" sparse-checkout set --no-cone --stdin < "$SNAPSHOT_ALLOWLIST"
git -C "$SNAPSHOT_CHECKOUT" checkout --detach "$SNAPSHOT_COMMIT"
rm "$SNAPSHOT_ALLOWLIST"
test "$(git -C "$SNAPSHOT_CHECKOUT" rev-parse HEAD)" = "$SNAPSHOT_COMMIT"
test -f "$SNAPSHOT_CHECKOUT/runs/ridge_sinkhorn_3seed_v1/job_manifest.json"
test ! -e "$SNAPSHOT_CHECKOUT/runs/ridge_sinkhorn_3seed_v1/attempts"
test ! -e "$SNAPSHOT_CHECKOUT/runs/ridge_sinkhorn_3seed_v1/outer_eval"
ARTIFACT_ROOT="$SNAPSHOT_CHECKOUT/runs"
HISTORICAL_ROOT=/tmp/rs3-inputs/artifacts/oof
```

If `fetch` cannot obtain that exact commit, stop and obtain the verified
snapshot from its provider; do not replace it with a current branch tip. Set
`ARTIFACT_ROOT` to the detached checkout's `runs` directory. For a supplied
materialized tree, set it to the directory containing both the marker and
`ridge_sinkhorn_3seed_v1/` instead.

Obtain the separate audited historical bundle and set `HISTORICAL_ROOT` to its
OOF directory. It must contain the 16 exact historical sources and its
compatibility materialization record. The root must expose both `task3c_oof/`
and the separate `task3b_pilot_ce_s78_o0_i0/` source used by the frozen CE
inner-0 decision. A repository clone does not provide these historical
payloads.

If the local audited Task 3B source is available at the path below and the
destination experiment root is absent, copy its shared `fold_manifest.json`
and four required CE inner-0 files byte for byte. This omits the source
directory's logs and training history. Do not copy a Task 3C-side CE inner-0
directory as a substitute, edit metadata, or reconstruct predictions. The
reader's compatibility checks are the final verification.

```bash
ANALYSIS_ROOT=/home/dzzz83/Documents/code/expert_method
HISTORICAL_ROOT=/tmp/rs3-inputs/artifacts/oof
TASK3B_EXPERIMENT="$ANALYSIS_ROOT/artifacts/oof/task3b_pilot_ce_s78_o0_i0"
TASK3B_SOURCE="$TASK3B_EXPERIMENT/expert_CE/seed_78/outer_0/inner_0"
TASK3B_DEST_ROOT="$HISTORICAL_ROOT/task3b_pilot_ce_s78_o0_i0"
TASK3B_DEST="$TASK3B_DEST_ROOT/expert_CE/seed_78/outer_0/inner_0"
test -f "$TASK3B_SOURCE/run_metadata.json"
test -f "$TASK3B_SOURCE/predictions.json"
test -f "$TASK3B_SOURCE/resolved_config.json"
test -f "$TASK3B_SOURCE/checkpoints/CE_seed78_final.pt"
test -f "$TASK3B_EXPERIMENT/fold_manifest.json"
test ! -e "$TASK3B_DEST_ROOT"
mkdir -p "$TASK3B_DEST/checkpoints"
cp -- "$TASK3B_EXPERIMENT/fold_manifest.json" "$TASK3B_DEST_ROOT/"
cp -- "$TASK3B_SOURCE/checkpoints/CE_seed78_final.pt" "$TASK3B_DEST/checkpoints/"
cp -- "$TASK3B_SOURCE/predictions.json" "$TASK3B_SOURCE/resolved_config.json" \
  "$TASK3B_SOURCE/run_metadata.json" "$TASK3B_DEST/"
chmod -R a-w "$TASK3B_DEST_ROOT"
```

The example mount paths are `/tmp/rs3-inputs/runs` and
`/tmp/rs3-inputs/artifacts/oof`; use the corresponding paths where the audited
inputs are mounted. Keep both roots separate from the analysis checkout.

## 3. Install dependencies

From the analysis checkout, install the core project requirements and the
optional plotting dependency:

```bash
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-analysis.txt
```

`requirements-analysis.txt` includes `requirements.txt` and
`matplotlib>=3.9.0`. Matplotlib is only needed for PNG/SVG output. With the
core requirements alone, pass `--no-figures` to write tables and reports
without plots.

## 4. Validate the inputs

From the analysis checkout, set the configuration path. `ARTIFACT_ROOT` and
`HISTORICAL_ROOT` remain set by step 2. If using a verified materialized
snapshot rather than the detached Git checkout, set `ARTIFACT_ROOT` to its
`runs` directory before continuing.

```bash
CONFIG=configs/studies/ridge_sinkhorn_3seed_v1.yaml
```

Validate identities, source decisions, payload hashes, row membership, and
finite saved predictions without assembling aligned logit matrices, running
diagnostic metrics, or creating an output directory. Validation parses each
prediction payload to check its schema and membership and checks checkpoint
metadata; the full run later assembles aligned arrays for analysis:

```bash
.venv/bin/python -m expert_method.diagnostics \
  --config "$CONFIG" \
  --artifact-root "$ARTIFACT_ROOT" \
  --reuse-root "rs3=$HISTORICAL_ROOT" \
  --stage inner \
  --validate-only
```

A successful validation reports the result snapshot commit and counts for 15
locks, 240 inner jobs, 224 native inner jobs, and 16 historical inner jobs.
Validation checks the immutable freeze, study and matrix manifests, fold
membership, native and reused checkpoints and predictions, and the saved
inner compatibility audit. The historical root name `rs3` is part of the
frozen study identity; change its path when mounting elsewhere, but preserve
that logical name.

On the prepared local `/tmp/rs3-inputs` snapshot, the full validation took
about nine minutes. It hashes and parses the saved inputs, so storage
throughput and whether the files are locally cached affect the runtime.

## 5. Run the complete descriptive report

After validation succeeds, run the full 15-pair descriptive report into a new
output directory. The output must not overlap either input root.

```bash
.venv/bin/python -m expert_method.diagnostics \
  --config "$CONFIG" \
  --artifact-root "$ARTIFACT_ROOT" \
  --reuse-root "rs3=$HISTORICAL_ROOT" \
  --stage inner \
  --output-root runs/diagnostics/rs3-inner-v2
```

On the prepared local checkout, `rs3-inner-v1` is an earlier full report from
before the final edge-case and output-safety fixes. It remains as immutable
historical output. The final reviewed report is `rs3-inner-v2`, generated by
the current analysis source; an identical rerun with the same inputs and
arguments succeeds in that directory. If you mount the inputs at different
paths or change command arguments, choose a fresh output root because those
values are recorded in provenance.

For a deterministic smoke run, `--limit-pairs 1 --no-figures` analyzes only
the first seed/fold pair. Give that partial run its own output directory; its
`analysis.json` marks `complete_matrix` false. It cannot produce complete
three-seed summaries and does not replace the full report.

```bash
.venv/bin/python -m expert_method.diagnostics \
  --config "$CONFIG" \
  --artifact-root "$ARTIFACT_ROOT" \
  --reuse-root "rs3=$HISTORICAL_ROOT" \
  --stage inner \
  --output-root runs/diagnostics/rs3-inner-smoke \
  --limit-pairs 1 \
  --no-figures
```

The output root is immutable. Identical reruns may reuse matching file bytes;
conflicts cause an error rather than overwrite. Use a fresh output directory
when analysis code, inputs, or run arguments change, and retain previous
outputs.

## 6. Locate and interpret the outputs

Every complete run writes these files:

| File | Contents |
|:--|:--|
| `analysis.json` | Completion status, evidence labels, descriptive metrics, highlights, and interpretation limits |
| `provenance.json` | Analysis/result/training source identities, input hashes, locks, package versions, command, and access flags |
| `report.md` | Human-readable evidence and metric summary |
| `expert_oof_metrics.csv` | Per-expert inner OOF sample and macro metrics |
| `complementarity.csv` | Correctness overlap, pairwise contingency, agreement patterns, and H/M/T summaries |
| `per_class_specialization.csv` | Per-class expert recall and disagreement summaries |
| `frequency_associations.csv` | Descriptive class-frequency associations for correctness and routing weights |
| `oracle_opportunity.csv` | Label-dependent hard-selection oracle metrics and gaps to the strongest individual expert |
| `selection_cv_metrics.csv` | Metrics copied from the immutable cross-fit selection locks |
| `router_fit_set_metrics.csv` | Saved final router state predictions on rows used to fit that state |
| `router_weight_profiles.csv` | Overall router allocation profiles and entropy summaries |
| `router_class_profiles.csv` | Per-class router allocation profiles |
| `router_group_allocations.csv` | Sample-weighted and class-macro Head/Medium/Tail allocations |
| `router_stability.csv` | Fold-within-seed and seed-within-fold profile comparisons |
| `prior_diagnostics.csv` | Allocation distance and prediction changes against the selected prior-only control |
| `sinkhorn_adjustments.csv` | Raw, smoothed, and adjusted weights, frozen prices, marginal/row KL, decomposition, and scoped transitions |
| `seed_aggregates.csv` | Fold-within-seed summaries, then mean and sample SD across the three seed means |

With figures enabled, `figures/` contains PNG and SVG versions of
`complementarity`, `specialization`, `allocation_entropy`, `stability`, and
`sinkhorn_adjustments` when the corresponding tables have data.

Keep these populations distinct:

- **Expert OOF** uses saved inner predictions from experts that excluded each
  predicted sample from their training membership.
- **Selection CV** reproduces recorded cross-fit metrics from immutable locks.
  It is selection evidence, not an untouched confirmation set.
- **Router fit set** applies a saved final router to rows used to fit it.
  These are descriptive fit-set measurements and do not establish held-out
  generalization.

Aggregation first summarizes the five outer-fold rows within each seed. It
then reports the mean and **sample standard deviation** (`ddof=1`) over the
three seed means. The folds and seeds share training-population structure, so
the 15 pairs are not independent replicates. Do not compute independent
15-fold p-values or treat the three-seed spread as retraining uncertainty.

The hard-selection oracle is label-dependent and is not an inference method
or a soft-mixture bound. Sinkhorn marginal KL/residual measures the adjusted
fit-population marginal against the saved target prior; it is not by itself a
solver-convergence diagnostic. Read the saved solver convergence/objective
fields separately. These descriptive analyses do not select or retune a
method, establish a performance gain, or support claims about another
population.

## Troubleshooting

- **Missing Torch or project modules:** run the command with `.venv/bin/python`
  from the project checkout and install `requirements.txt` or
  `requirements-analysis.txt`.
- **Historical reuse is missing or incompatible:** check that the named
  `rs3` root exposes the exact 16 payload paths and its compatibility record.
  The pinned native snapshot does not carry those old payloads. Do not waive a
  hash or metadata check.
- **CE inner-0 is missing:** restore the audited original from the recorded
  Task 3B pilot source at the path in step 2 and verify through the reader.
  Do not fabricate a Task 3C run record.
- **Snapshot commit or file hash differs:** restore the complete pinned result
  snapshot or supplied marker. Do not update the marker to match altered
  files.
- **A lock or fold-membership check fails:** stop and use one consistent frozen
  snapshot, configuration, named reuse root, and set of all 15 locks. The
  reader is read-only and has no bypass option.
- **Output conflict or input/output overlap:** preserve the existing report
  and choose a fresh output directory outside both input roots.
- **Matplotlib is unavailable:** install `requirements-analysis.txt` or add
  `--no-figures` to produce CSV, JSON, and Markdown outputs only.
- **A smoke report has fewer than 15 pairs:** check `complete_matrix` in
  `analysis.json`; run the full command into a fresh output root before
  interpreting seed aggregates.

## Developer notes

`InnerArtifactReader` owns frozen-identity, lock, membership, and hash
validation and loads only aligned inner arrays. `StudyDiagnosticsRunner`
coordinates the 15 seed/fold pairs and uses saved locked inference outputs; it
does not fit an expert, router, or replacement method. The numerical modules
(`complementarity`, `router`, `stability`, `sinkhorn`, and `prior`) are pure
array functions. `aggregation` summarizes pair rows, and `reporting` writes
JSON, CSV, Markdown, and figure artifacts.

To add a diagnostic, define a typed array-only function in the relevant
numerical module (or a new focused module), document input shapes and the
denominator behind each metric, and return JSON-safe values with `null` for
undefined cases such as absent classes or constant profiles. Add a synthetic
unit test for the metric's boundary cases. Wire its rows into the runner,
aggregate at the pair-table layer, and update the report/export only when the
new metric needs a persisted table or figure. Keep `InnerArtifactReader`
read-only and do not modify frozen manifests, locks, source artifacts, or the
scientific selection path for descriptive reporting.

Preserve the evidence labels **Expert OOF**, **Selection CV**, and **Router
fit set**. Aggregate five folds within each seed before calculating the mean
and sample SD across the three seed means. The 15 seed/fold pairs are not
independent observations; do not add independent-pair significance tests or
call fit-set gains generalization. A hard-selection oracle is label-dependent
and is not a soft-mixture bound. A nonzero Sinkhorn marginal residual is a
fit-population constraint discrepancy, not by itself evidence of solver
nonconvergence; use saved convergence and objective diagnostics for that.
