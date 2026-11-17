# OT methods and reporting handoff

Updated: 2026-11-17.

## Start here

The user will specify the next task in the new chat. No implementation task is
pending. Read the current AGENTS.md instructions and this handoff before acting.
Keep changes focused and explanations simple. Present a brief plan before
implementation, verify changed behavior, and perform one focused self-review
in the same agent. Do not spawn agents unless explicitly requested or required
by applicable instructions.

## Repository state

- Workspace: `/home/dzzz83/Documents/code/expert_method`.
- GitHub repository: `Dzzz83/MoE_Imbalance`.
- Branch: `integration/ot-methods`, tracking `origin/integration/ot-methods`.
- Verified HEAD: `1b6a33f` (`fix: use balanced accuracy for main result charts`).
- At handoff creation there were no tracked changes. Existing untracked items
  were `.agents/`, `docs/ot_reweighting_debug_handoff.md`,
  `docs/prototype_weighting_handoff.md`, and `docs/reweighting_handoff.md`.
  Preserve them. This handoff is newly added and has not been committed or pushed.
- AGENTS.md was previously updated locally and is Git-ignored; do not force-add it.

## Implemented methods

The integration branch merged reweighting from `feat/ot-reweighting` and feature
synthesis from `hieu`. Merge commit: `01840f8`. The paths remain independent;
their losses are not combined. See `docs/ot_methods.md` and `docs/reweighting.md`.

Shared setup: CIFAR-100, plain ResNet-32, imbalance factor 0.01, preparation
seed 42, retained pool of 10,847 images. Training seeds are 0, 1, 2. The balanced
test set has 100 examples per class. Test data never enters fitting or synthesis.

Reweighting classifier continuations include uniform, class-balanced,
cluster-balanced, prototype Sinkhorn, and SAVA-inspired weighting. SAVA assigns
equal total mass to each class and distributes it within the class using its
scores. The reported `sinkhorn` candidate is prototype weighting, not a separate
full `ot_finetuning` result.

### Final synthetic method

1. Train/load a baseline and extract the retained images' 64-dimensional features.
2. Freeze the backbone, including batch-normalization state.
3. For each class, generate `max(0, 500 - real_count)` extra features.
4. Fit a one-versus-rest linear guide on real features. It uses hinge loss and
   L2 regularization, optimized with Adam. It is SVM-style, has no bias term,
   and is separate from the final classifier.
5. Give harder positive examples more transport target mass using a softmax of
   their hinge losses. Sample initial points from the class's coordinate-wise
   Gaussian mean and standard deviation.
6. Optimize transport using geometry cost, a guide score-margin penalty, and
   entropy regularization. Project each update onto the required row/column
   marginals using Sinkhorn-style scaling, with backtracking on the objective.
7. Generate each feature as a normalized weighted average of real features from
   the same class. The result lies inside their convex hull. Gaussian initial
   points influence the geometry cost; they are not directly saved as samples.
8. Train the existing final linear layer on real plus synthetic features with
   cross-entropy. The guide is not needed at inference; ordinary images pass
   through the frozen backbone and updated classifier.

Generation scales features internally and converts them back to raw coordinates
before saving. It generates feature vectors, not images. With this retained pool
and target 500, all classes reach 500 combined vectors. There is no downsampling
of a class already at or above the target.

Code: `pipeline/extract_features.py`, `pipeline/methods/synthesis.py`, and
`pipeline/train_classifier.py`. `pipeline/run_comparison.py` provides controls:

| Candidate | Behavior |
|---|---|
| Baseline | Original trained model |
| Real-only | Continue classifier training on real features |
| Random duplication | Repeat real features to reach the target |
| Uniform OT | Uniform target masses, no guidance penalty |
| Loss-shifted OT | Hardness-based target masses, no guidance penalty |
| Full guided OT | Hardness-based target masses plus guidance penalty |

The latest discussion explained this method and its SVM-style guide. No code
changes were requested during that discussion.

## Training and historical results

Matched configs load the same baseline per seed for both paths. However, the
archived results below predate that matched protocol: reweighting used a
warm-up/multistep baseline, synthesis a cosine baseline. Their classifier
continuation recipes also differ. Do not attribute cross-family differences
solely to the method.

`scripts/kaggle_prototype_comparison.py` supports `EXPERIMENT_PATH` values
`reweighting`, `synthesis`, or `both`, with seeds 0, 1, 2. It pins implementation
commit `01840f8` and publishes to `ot-methods-results`. Its historical filename
does not mean it only runs prototype weighting. See the integration guide for
resume behavior and checkpoint compatibility.

`results/` contains archived metrics/configurations, not new evaluations of the
merged code and not model checkpoints:

- Reweighting: source `prototype-weighting-results` at
  `f39296984fc5b03ca844ac5578a387238efdc5a7`.
- Synthesis: source `hieu` at `30ac14ca302ab0e6698a88fe97ee371bef04cbb1`.
- `results/manifest.json` records provenance for 91 imported files; their exact
  Git blob hashes and source line endings were verified during import.
- Synthesis's main three-seed results are `comparison_5000_seed0/1/2`.
  Here 5000 is the generation iteration limit, not the number of generated
  examples. The older `comparison_seed0` is preserved but excluded from summaries.
- `ot-methods-results` had no saved results available at import time; this is
  historical information, not a fresh check of the remote branch.

## Reporting choices to preserve

Implementation: `scripts/report_results.py`; tests: `tests/test_report_results.py`.
Outputs and detailed interpretation: `results/paper/README.md`.

The user's final preferences are:

- Two tables, one per method family, in Markdown, LaTeX, and CSV.
- Four simple bar charts, PNG only: balanced accuracy and tail accuracy for
  each family. Each chart includes the baseline and five candidates.
- Bars show the mean across seeds 0, 1, 2, with numeric labels, no error bars,
  individual seed markers, or legends. Tables retain mean ± sample SD (ddof=1).
- Main charts read `balanced_accuracy` directly. Overall and balanced accuracy
  happen to be equal on this balanced test set. Tables still label their
  original overall-accuracy column; only charts were changed by that request.
- Tail accuracy means macro recall over classes with fewer than 20 retained
  real examples, averaged across seeds. Synthetic counts do not change groups.
- Shared zero-based axes within each metric: currently 0–45% balanced accuracy
  and 0–20% tail accuracy. No significance claims.
- Do not restore old point plots, PDF/SVG images, analysis JSON, or seed CSV
  exports. Raw source metrics and manifest remain in the archive.

Current PNGs:

- `results/paper/reweighting_balanced_accuracy.png`
- `results/paper/synthesis_balanced_accuracy.png`
- `results/paper/reweighting_tail_accuracy.png`
- `results/paper/synthesis_tail_accuracy.png`

Images are 2160 × 1260 pixels at 300 DPI. Tables are
`results/paper/{reweighting,synthesis}.{md,tex,csv}`. LaTeX snippets require
booktabs and amsmath; compilation was unavailable and has not been verified.

## Environment and verification

Previously used configured Python:

```text
/home/dzzz83/Documents/code/expert_method-local-archive-2026-11-05/.venv/bin/python
```

It had PyTorch, NumPy, Matplotlib, and pytest available. System Python lacked
PyTorch. Recheck availability if the environment changes. Optional reporting
dependency is declared in `pyproject.toml` as `.[reporting]`.

Run commands from the repository root using that interpreter:

```bash
python -m pytest tests/test_report_results.py
python scripts/report_results.py --overwrite
```

The last reporting implementation check passed all 13 focused tests; the four
PNGs were visually inspected, and `git diff --check` passed. Tests cover metric
validation, matched-seed summaries, paired changes, formatting, output safety,
cleanup of obsolete exports, and chart selection/axis limits. These results
predate this documentation-only handoff. Do not claim a fresh test run or a
current full-suite pass. CPU integration checks do not establish research
quality or CUDA training behavior.

No next task has been selected. Wait for the user's instructions.
