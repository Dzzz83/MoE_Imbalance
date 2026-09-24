# Bug-fix report — session of the routing audit

> Scope: every bug verified in the audit (`docs/` + `records/` knowledge base)
> was fixed, each implementation fix carries a red run (failing before) and a
> green run (passing after), and this file states what the fixes do to the
> reported results.
>
> Related: [`protocol.md`](../protocol.md) · [`results.md`](../results.md) ·
> [`research.md`](../research.md) · [routing-preregistration.md](../../records/routing-preregistration.md)

---

## 1. Headline

**No reported number changes. No retraining. No re-evaluation required.**

Every number in `results.md`, `README.md` and the records is produced by
`scripts/evaluate_experts.py` / `scripts/analyze_subsets.py` →
`scripts/evaluation.py` + `scripts/subsets.py` → `scripts/base_trainer.py`
(group split, BA) and `scripts/evaluation.py` (ECE). **Not one file on that path
was modified** (verified with `git diff` against the list in §4), and the
canonical split artifact is byte-identical (`data/processed/lt_ir100_train_indices.npy`
untouched — regeneration still reproduces it exactly, asserted by
`tests/test_protocol_splits.py`).

**Confirmed empirically, not just by diff.** After the local GPU driver recovered,
`scripts/evaluate_experts.py --seeds 78 88 1034 --device cuda --tta-augs 10` was
re-run on the fixed code (one logged read, logged in `docs/test-access-log.md`).
Every published value reproduces to the same rounding — see §4a. This upgrades
the central claim of this report from "the path is unchanged" to "the numbers
come out identical".

What was wrong was **labels and latent paths**, not measured values: three
documents described the Head/Medium/Tail split as 30/36/34 classes when the
committed split is 35/35/30, and two library functions that no reported number
flows through were computing the wrong thing.

---

## 2. Fixes and their evidence

| # | Bug | Fix | Red evidence | Green evidence |
|:-:|:--|:--|:--|:--|
| A1 | Two divergent Head/Med/Tail definitions: `scripts/utils/data.py` used `> 20` for Medium, so the class with exactly 20 samples fell into Tail | `get_class_groups` now delegates to `scripts/base_trainer.compute_class_groups` (the one frozen definition); `scripts/evaluate_dace.py` delegates to it too | `test_evaluation.py`: ❌ `scripts.utils.data.medium disagrees … 99` | ✅ `one definition: head=30 medium=37 tail=33` (boundary fixture); ✅ `class groups: head=35, medium=35, tail=30 (boundary class 69 has 20 samples)` |
| A2 | `BaseRouter.evaluate()` read the combine-then-argmax sentinel as "expert 0", so Uniform/Probability reported expert 0's accuracy and 100% usage | New `selects_single_expert` flag on `BaseRouter` (False for Uniform/Probability); `evaluate()` always scores `predict_class` and reports usage from the routing weights; `compute_routing_metrics` gained `class_predictions`/`routing_weights` | `test_router_contract.py`: ❌ `UniformRouter.evaluate() reported BA 0.0000, but the rule's own predictions score 1.0000` | ✅ `all 4 registered rules evaluate their own predictions` + ✅ `combine-then-argmax rules declared and report equal usage` (33.3/33.3/33.3) |
| A3 | `scripts/utils/metrics.py::ece` binned `[lo, hi)`, dropping every confidence of exactly 1.0 while still dividing by the full N | Bins are now right-closed `(lo, hi]`, matching the live `scripts/evaluation.py` implementation | `test_evaluation.py`: ❌ `metrics.ece = 0.25, expected 0.75` | ✅ `ECE counts full-confidence samples (0.7500 == 0.7500)` |
| A4 | Dead placeholder `oracle_ba = balanced_accuracy(...)  # placeholder`, overwritten two lines later | Removed | n/a (no behaviour; code inspection) | compiles, suite green |
| A5 | `scripts/utils/data.create_cifar_loader` read the removed `lt_train_indices.npy` / `lt_val_indices.npy`, bypassing `protocol_splits` and resurrecting a val split | Reads the canonical artifact via `load_lt_train_indices`; `'val'` raises `ProtocolError` with the reason | `test_protocol_splits.py`: ❌ `FileNotFoundError: …/processed/lt_train_indices.npy` | ✅ `legacy loader uses the canonical artifact; 'val' rejected` |
| A5b | `load_expert_checkpoint`/`load_all_experts` looked for `{expert}_best.pt`, a name the trainer never writes; a multi-seed dir was silently guessed | New `find_expert_checkpoint` resolves `{expert}_seed{N}_final.pt` and **refuses** an ambiguous directory, matching `ExpertPool` | `test_evaluation.py`: ❌ `load_expert_checkpoint() got an unexpected keyword argument 'checkpoint_dir'` | ✅ `legacy checkpoint loader resolves _final.pt and refuses ambiguity` |
| B1 | Docs stated Head 30 / Med 36 / Tail 34 (copied from a synthetic fixture) | Corrected to 35 / 35 / 30 with the half-open boundary spelled out, in `README.md`, `docs/project-context.md`, `docs/results.md`; the misleading ✅ print in `tests/test_evaluate_dace_synthetic.py` now says the fixture is synthetic | new regression test fails on any drift | ✅ `test_class_group_sizes_match_the_artifact` |
| B2 | `data/protocol_splits.py` docstring claimed `routing_dev` yields "honest correctness labels" — the opposite of the verified finding | Docstring now states that no split of that pool is honest, and that nothing live reads it | n/a (docstring) | doc reviewed against `problem.md` §5 |
| B3 | `test_no_validation_artifact` claimed "no validation artifact may exist" but checked one filename | Now checks all three protocol-owned legacy names **and** scans the live data-layer modules for references to removed artifacts; unrelated leftovers are reported, never silently passed | ❌ `scripts/utils/data.py still references removed artifacts: ['lt_val_indices']` | ✅ + ⚠️ lists `balanced_val_indices.npy`, `val_targets.npy`, `dace_val_cache.npz`, `phase0d_*_val.npz` |
| B4 | `docs/project-context.md` labelled TTA the best routing rule (44.15 < Confidence 44.27) | Corrected, with a note that TTA is best on Tail only | n/a (docs) | checked against `checkpoints/test_evaluation.json` |
| B5 | `scripts/router/__init__.py` docstring still listed the removed `Product` rule and omitted `Probability` | Corrected, with the Amendment-2 reason | n/a (docstring) | doc matches `ROUTERS` |
| C1 | Skipped tests were counted as **passes**: `test_gpu.py` reported "13 passed, 0 failed" while 8 CUDA checks never ran; `test_router_contract._skip_if_absent` did the same | `_SkipTest` is raised and counted separately; both runners print `passed / skipped / failed` and a warning that skipped checks are unverified | ❌ 8 no-ops reported as passes | ✅ `13 tests: 5 passed, 8 skipped, 0 failed` + `NOTE: 8 check(s) did not run` |
| C2 | The only behaviour check for Confidence/TTA was `idx.min() >= 0 and idx.max() < 3` (true for any argmax) | Added exact-decision tests for Confidence and TTA and an exact usage check | — | ✅ `confidence/TTA decisions exact; usage {'A': 50.0, 'B': 25.0, 'C': 25.0}` |
| C3 | Every DACE routing assertion ran on a degenerate fixture (all samples "disagree" → identical prototypes → constant score, always expert 0) | New test forces a threshold that splits the batch and then requires varied scores and decisions | the old fixture printed `agree=0, disagree=16` on every case | ✅ `scores vary (max \|score\| 1.69e-04), experts chosen [0, 1]` |
| C4 | `tests/test_dace_b_synthetic.py` re-implemented mixup with numpy RNG instead of exercising the shipped module | Imports and calls `data.mixup.Mixup` / `Mixup.criterion` | — | ✅ suite green |
| C5 | Unseeded fixtures with probabilistic/ratio assertions (contrastive loss, partition sampler, KL, routing head, and an unseeded `np.random.shuffle` inside the DACE probe) | Seeded at module or fixture level; the sampler tests take an explicit `torch.Generator` | sampler class mix varied run to run (504/499/515 of 1000) | ✅ three consecutive runs identical (`share 0.290`) |
| C6 | `PartitionedSampler(full_ratio=1.0)` test asserted only that indices were in range — a sampler ignoring `full_ratio` passed | Asserts the drawn primary share matches the data share | — | ✅ `full_ratio=1.0 draws from all classes (share 0.290 vs data share 0.279)` |
| D1 | `losses/contrastive_routing_loss.py` B<2 path returned a float in `aux` and a graph-free loss, breaking `.item()` and `.backward()` on 1-sample batches | Returns `embeddings.sum() * 0.0` and a detached tensor in `aux`; docstring formula corrected to the SupCon form actually implemented | ❌ `aux['contrastive_loss'] is float, but the main path returns a tensor` | ✅ `B<2 path returns a zero tensor with a graph` |
| D2–D8 | Retired-line defects: `train_paco_weighted` passed the two-view **list** as `im_q` (TypeError every step) and counted views as batch size (`n_val` collapsed to 0, so the documented 20% val interleave never ran); its `PaCoLoss` was never registered so its buffers stayed on CPU; DACE A/B/C never forwarded `--seed` (every seed run was seed 0); DACE-C built a `MultiStepLR` that was never stepped (400-epoch run actually used the 160/180 schedule); DACE B/C multiplied an already-reduced scalar loss by per-sample weights (a silent global rescale); `train_paco.py` read `best_metric_val` before assigning it (AttributeError on the first improving epoch) and called `_save_checkpoint(is_best=True)` (unsupported kwarg) | Each patched at its site, with comments recording why | static only — these scripts cannot run against the current API (documented) | `py_compile` clean on every touched file; the fixes are mechanical and each comment names the failure it prevents |

Changes are confined to: `scripts/utils/{data,metrics}.py`,
`scripts/router/{base,uniform,probability,__init__}.py`,
`data/protocol_splits.py` (docstring), `losses/contrastive_routing_loss.py`,
`scripts/evaluate_dace.py` (delegation), the retired
`scripts/train_paco*.py` / `train_dace_*.py` / `diagnose_dace.py`, `README.md`,
the `docs/` knowledge base, and `tests/` (10 suites).

---

## 3. What the fixes do to reported results

| Reported quantity | Affected? | Why |
|:--|:--:|:--|
| Per-expert BA / Head / Medium / Tail / ECE (`results.md` §2) | **No** | produced by `evaluate_predictions` → `compute_class_groups` + `evaluation.expected_calibration_error`; untouched |
| Ensembling baselines: Uniform 46.98, Probability 45.95, Confidence 44.27, TTA 44.15 | **No** | `evaluate_experts.py` uses `predict_class`/`predict_proba`; the new `selects_single_expert` flag is read only by `evaluate()`, which that script never calls |
| Tail values (e.g. Uniform 18.76) | **No** | averaged over the same 30 tail classes as before; only the *documented count* was wrong |
| Headroom: all-wrong 39.73, oracle 60.27, per-seed counts, κ | **No** | `HeadroomAnalyzer` untouched |
| Ensemble-size curve 40.04 / 44.19 / 45.97 / 46.98 | **No** | `scripts/subsets.py` untouched |
| Training-run health table (`results.md` §6) | **No** | `base_trainer` training loop and `check_runs.py` untouched |
| Recorded `checkpoints/test_evaluation.json` | **No** | an output of the above; left exactly as produced |
| `docs` statement "Head 30 / Medium 36 / Tail 34" | **Yes, corrected** | the only published claim that changed — it was false; the split is 35 / 35 / 30 |

The one boundary that mattered: a class with **exactly 20** training samples
(class 69, 100 test images) is Medium under the frozen definition
(`20 ≤ n < 100`). Had the drifted `scripts/utils/data.py` definition been the one
in use, Tail would cover 31 classes (3,100 images) instead of 30 (3,000), and
the Med/Tail columns would move by that one class. It never was: that variant is
reachable only through `BaseRouter.evaluate()`, which the reporting path never
called. With A1 fixed, the two definitions cannot diverge again.

---

## 4. What needs re-evaluation or retraining

**Retraining: none.** The training path is unchanged — `scripts/base_trainer.py`,
`scripts/trainers.py`, `scripts/train.py`, `scripts/config.py`,
`data/cifar_lt.py`, `data/lt_datamodule.py`, `data/mixup.py`, `data/tta.py`,
`models/resnet32.py`, `losses/{ce,lal,balanced_softmax}_loss.py` and
`data/processed/lt_ir100_train_indices.npy` all carry no diff
(`git diff --quiet` on each). The 12 checkpoints remain valid and comparable.

**Re-evaluation: not required for anything already published.** No metric's
definition, binarisation, split or aggregation changed on the reporting path.

**Both optional follow-ups have now been done**, by your instruction:

1. **Empirical confirmation — done, and it reproduces exactly.** With the driver
   recovered, `scripts/evaluate_experts.py --seeds 78 88 1034 --device cuda
   --tta-augs 10` was re-run on the fixed code. This added the seventh row to
   `docs/test-access-log.md` (append-only; the read is visible). Full comparison
   in §4a: every published value matches to the same rounding, so the fixes
   demonstrably did not move a single reported number. The run also rewrote
   `checkpoints/test_evaluation.json`, which now records the four pre-registered
   rules (the old file still contained the deleted `Product` rule).
2. **The GPU path is verified.** The local driver was wedged when the fixes were
   made, so `tests/test_gpu.py` reported `5 passed, 8 skipped`; the GPU came back
   and the suite ran clean at `13 tests: 13 passed, 0 skipped, 0 failed` on
   `NVIDIA GeForce RTX 3060 Laptop GPU (sm_86, 6.1 GB)` with CUDA 12.6 —
   including the four-expert GPU training step, the VRAM check (batch 128 peaks
   at 336 MB of 6076 MB), the GPU NaN guard, and CPU/GPU agreement at
   `max |Δ| = 2.75e-04` with 100% argmax match (matching the ≈2.8e-04 that
   `docs/project-context.md` §7 documents with TF32 disabled). A 2-batch
   `scripts/train.py --device cuda` dry run and a full 1-epoch run over all
   10,847 samples (6 s, loss 4.0581, finite gradients, checkpoint written outside
   the repo) both complete.

### 4a. The GPU re-run against the published numbers

| Quantity | Published (`results.md`) | Re-run on GPU (fixed code) | Match |
|:--|:--|:--|:--:|
| CE BA / Head / Tail | 37.69 ±0.73 / 65.15 / 8.67 | 0.3769 ±0.0073 / 0.6515 / 0.0867 | ✅ |
| LAL BA / Head / Tail | 42.43 ±1.53 / 59.54 / 23.49 | 0.4243 ±0.0153 / 0.5954 / 0.2349 | ✅ |
| BalancedSoftmax BA / Head / Tail | 41.34 ±0.78 / 58.88 / 22.60 | 0.4134 ±0.0078 / 0.5888 / 0.2260 | ✅ |
| Mixup BA / Head / Tail | 38.69 ±0.61 / 69.40 / 5.69 | 0.3869 ±0.0061 / 0.6940 / 0.0569 | ✅ |
| Uniform BA / Tail | 46.98 ±0.69 / 18.76 ±0.86 | 0.4698 ±0.0069 / 0.1876 ±0.0086 | ✅ |
| Probability BA / Tail | 45.95 ±0.55 / 18.70 ±0.50 | 0.4595 ±0.0055 / 0.1870 ±0.0050 | ✅ |
| Confidence BA / Tail | 44.27 ±0.46 / 18.69 ±0.36 | 0.4427 ±0.0046 / 0.1869 ±0.0036 | ✅ |
| TTA BA / Tail | 44.15 ±0.68 / 19.38 ±0.97 | 0.4415 ±0.0068 / 0.1938 ±0.0097 | ✅ |
| All-wrong floor, per seed | 38.91 / 40.26 / 40.02 | 0.3891 / 0.4026 / 0.4002 | ✅ |
| Oracle ceiling, per seed | 61.09 / 59.74 / 59.98 | 0.6109 / 0.5974 / 0.5998 | ✅ |
| Decision rule verdicts | all three "no gain" | Probability, Confidence, TTA all "no gain" | ✅ |
| Ensemble size, uniform, k=1..4 | 40.04 ±0.53 / 44.19 ±0.50 / 45.97 ±0.57 / **46.98 ±0.69** | 0.4004 / 0.4419 / 0.4597 / 0.4698 (same stds) | ✅ |
| Ensemble size, best-of-subset (selection-on-test) | 42.77 / 46.44 / 47.59 / 46.98 | 0.4277 / 0.4644 / 0.4759 / 0.4698 | ✅ |
| Best subset per size | LAL · LAL+BS · LAL+BS+Mixup · all four | identical, all three seeds | ✅ |
| Ensemble size, probability averaging | 40.04 / 43.21 / 44.79 / 45.95 | 0.4004 / 0.4321 / 0.4479 / 0.4595 | ✅ |

The ensemble-size curve needed its own entry point, so with your approval
`scripts/analyze_subsets.py --seeds 78 88 1034 --device cuda` was run as a
**second logged read** (eighth row in `docs/test-access-log.md`). It reproduces
the whole table above, including the best subset at every size and the
probability-averaging row — so **every published table is now empirically
re-confirmed**, not just the headline numbers.

**Side effect you should know about: two tracked JSONs were rewritten.**
`scripts/evaluate_experts.py` and `scripts/analyze_subsets.py` write to
`checkpoints/test_evaluation.json` and `checkpoints/subset_analysis.json` by
default, and both are tracked by git.

- `checkpoints/subset_analysis.json` — **zero differences** against the committed
  file (every rule, size, metric, mean/std, best-subset list and per-seed entry).
- `checkpoints/test_evaluation.json` — zero numeric differences, one structural
  change: the committed file still carried a `Product` entry (the rule removed by
  Amendment 2) while the refreshed file records only the four pre-registered
  rules.

Keeping both refreshes is the honest record (they now match the frozen registry);
`git checkout -- checkpoints/` restores the committed snapshots if you prefer them.

**Future evaluations do change behaviour in one way** (correctly): any caller of
`BaseRouter.evaluate()` — the documented API for scoring a rule — previously got
expert 0's metrics for Uniform and Probability, and now gets the rule's own.
Nothing in the reported pipeline used it; a new analysis that calls it will get
the right answer.

**Housekeeping decision for you (not changed by this session):** untracked
retired-split leftovers sit in `data/processed/` (`balanced_val_indices.npy`,
`val_targets.npy`, `dace_val_cache.npz`, `phase0d_dace_val.npz`,
`phase0d_original_val.npz`). Nothing live reads them, and the test suite now
names them instead of passing silently, but deleting or archiving them is your
call — no test deletes data.

---

## 5. Test-suite state

```text
test_contrastive_routing_loss.py   exit 0   7 passed, 0 failed
test_dace_a_synthetic.py           exit 0   4 passed, 0 failed
test_dace_b_synthetic.py           exit 0   6 passed, 0 failed
test_dace_c_synthetic.py           exit 0   5 passed, 0 failed
test_evaluate_dace_real.py         exit 0   10 passed, 0 failed
test_evaluate_dace_synthetic.py    exit 0   6 passed, 0 failed
test_evaluation.py                 exit 0   34 passed, 0 failed
test_gpu.py                        exit 0   13 tests: 13 passed, 0 skipped, 0 failed
test_kl_divergence.py              exit 0   9 passed, 0 failed
test_partitioned_dataset.py        exit 0   8 passed, 0 failed
test_protocol_splits.py            exit 0   24 passed, 0 failed
test_resnet32_routing.py           exit 0   9 passed, 0 failed
test_router_contract.py            exit 0   22 tests: 22 passed, 0 skipped, 0 failed
test_subsets.py                    exit 0   10 passed, 0 failed
test_training_config.py            exit 0   21 passed, 0 failed
test_training_protocol.py          exit 0   21 passed, 0 failed
test_tta.py                        exit 0   14 passed, 0 failed
```

**Executed vs skipped:** 223 checks ran and passed; 0 skipped. (Before the driver
recovered this read `215 passed, 8 skipped` — the 8 CUDA-only checks now actually
execute, and no other suite changed behaviour with a GPU present.)

**Environment caveat (not a repo defect):** while the driver was wedged, one
batch run aborted `test_tta.py` with `exit 134` and no summary; the process died
inside PyTorch's own `c10::ApproximateClock` assertion
(`getCount is non-monotonic`), which also happened once in a standalone snippet.
`test_tta.py` has since passed on every run, including all runs with the GPU
present, so this looks tied to the broken-driver period rather than the
repository. It can make any suite run flaky, so re-run before believing a single
red.

---

## 6. Adversarial review of this delivery

The strongest objections raised against the work above, and their answers:

| Objection | Answer |
|:--|:--|
| "The A2 fix relies on a class attribute a future rule could forget." | Partly true and now closed: accuracy is taken from `predict_class` **unconditionally**, so a forgotten flag cannot corrupt BA; it could only mis-report usage. `test_combine_then_argmax_rules_report_equal_usage` pins the set of combining rules, so any registry change forces a deliberate decision. Verified by disabling the flag on `ProbabilityAverageRouter` in memory: BA stayed correct (1.0000 == 1.0000), which is why the usage check — not the BA check — is the one that guards the flag. |
| "'No reported number changes' is asserted by diff, not by re-running." | **Resolved empirically.** After the driver recovered, the frozen evaluation was re-run once on GPU and every published value reproduced to the same rounding (§4a), including all four routing rules, the per-expert table and the headroom. The diff argument still stands as the reason it *had* to reproduce: no file on the reporting path carries a change. |
| "The retired-script fixes are unverifiable, so they do not count as fixed." | Stated as such: they are `py_compile`-clean and each comment names the failure it prevents, but they cannot be executed against the current API. They change no reported number and no live path. |
| "Some tests were edited, which can hide bugs." | Every test edit is a strengthening or a correction of a false green: skips no longer count as passes, the sampler test now asserts the split it claims, the DACE routing test now requires varied scores, the group-count print no longer states a false fact, and the deleted local mixup copy now exercises the shipped module. No assertion was removed to obtain green. |
| "The doublecheck timeline tooling shows 0 passing runs." | Correct and worth stating plainly: the suite was run through plain `bash`, so `doublecheck_report`'s timeline is empty. The actual red/green outputs are quoted in §2 and reproducible with `python tests/test_<name>.py`. An empty timeline must not be read as evidence either way. |

---

## 7. Known issues deliberately left alone

| Item | Why left |
|:--|:--|
| Retired scripts still reference `lt_train_indices.npy` / `balanced_val_indices.npy` (`train_paco.py`, `train_dace_*.py`, `diagnose_dace*.py`, `evaluate_dace.py`) | They cannot run against the current API (documented); repointing them at the canonical protocol would be rewriting the retired pipeline, which this session's spec excludes |
| `scripts/utils/data.print_data_info(val_counts=…)` still accepts a validation concept | Printing helper only; no split is read or created |
| `scripts/evaluate.py`, `scripts/analyze.py` (legacy entry points) | Their loader is fixed, but they still use legacy single-checkpoint naming/CLI shapes; they are not in the documented dead-code list and should either be added to it or deleted — a decision for you |
| `doublecheck-spec.md` is untracked and not gitignored | Written by this session's tooling; the repo previously dropped that file from git, so it is reported rather than committed or deleted |
| `metrics.compute_routing_metrics` default `class_counts` uses the formula rather than the artifact | Only reached when the caller passes no counts; the live path always passes real counts |
| `scripts/analyze_subsets.py` was not re-run | **No longer true — it was re-run** (second logged read, eighth log row) and the whole curve reproduces; see §4a |

### Audit-trail note (checked, no discrepancy)

`checkpoints/test_evaluation.json` and `checkpoints/subset_analysis.json` carried
mtimes later than the last pre-existing access-log row. That is a timezone
artefact, not an unlogged read: the log records **UTC** while the filesystem
records local time (UTC+7). 16:27 local = 09:27 UTC, matching the 09:26:23 UTC
evaluate row; 16:39 local = 09:39 UTC, matching the 09:38:52 UTC subset row. The
freshly added row for this session's read (`13:14:09 UTC` = 20:14 local) is
consistent with both clocks.
