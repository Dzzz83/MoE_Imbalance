# Prototype weighting handoff

## User objective and communication

The user wants OT/Sinkhorn to give useful training samples greater influence
and improve CIFAR-100-LT performance over suitable baselines. They prefer short,
plain explanations, one idea at a time, with concrete examples. Avoid lengthy
research summaries unless requested. Explain the method before proposing changes.
The next chat should continue the discussion; no new experiment or code change
has been agreed.

## Current method and what the user understands

All four continuations load the same seed-0 CE checkpoint, freeze its complete
backbone, and train its existing classifier for 40 epochs. Features are cached
once using deterministic preprocessing.

| Variant | Training weights |
|---|---|
| Uniform | Equal per image |
| Class-balanced | Each class gets 1/100, shared equally among its images |
| Cluster-balanced | Two cosine feature clusters per class, each gets half the class mass, shared equally within the cluster |
| Sinkhorn | The same prototypes and equal prototype quotas, with sample weights obtained from a relaxed transport plan's row sums |

Classes fall back to equal within-class weights if two stable clusters cannot
be found or either cluster has fewer than five images. Clustering is seeded and
shared between the cluster-balanced and Sinkhorn variants.

Sinkhorn's cost is cosine distance between normalized features and prototypes.
The prototype column masses are fixed to 1/2 within a class. Sample row masses
are flexible, with a KL penalty toward uniform sample masses. Defaults are
entropy epsilon 0.1, row KL strength 0.2, 200 maximum iterations, tolerance 0.001.
Each class's plan has total mass one; row sums are divided by 100 to become
global training weights.

The user most recently asked what Sinkhorn does here. The explanation given was
that it constructs an image-by-prototype contribution table, favors similar
image/prototype pairs, satisfies the equal column quotas, and discourages large
departures from uniform sample weights. An image's row sum is its training weight.
The weighted loss uses mean(N * global_weight * cross_entropy) on uniformly
shuffled cached features.

Features, prototypes, and weights are fixed during these continuations.
Classifier performance gives no feedback into the weighting calculation.
This exact recipe was proposed in this chat as an experiment. Its performance
was unproven when proposed. Chizat et al. support the generalized Sinkhorn solver,
and subclass contrastive learning provides related motivation, but neither
establishes the effectiveness of this exact frozen-feature weighting recipe.
Do not describe it as a directly reproduced published method.

Baseline pretraining uses weak augmentation: random crop to 32 with padding 4,
random horizontal flip, tensor conversion, normalization. Classifier continuations
use no augmentation.

## Code and Git status

Repository is /home/dzzz83/Documents/code/expert_method.
Remote is https://github.com/Dzzz83/MoE_Imbalance.git.
Branch is feat/ot-reweighting.

Implementation was committed and pushed as
7ff1eda64ac722580785412ab1af5f3208ccecb4.

| Path | Purpose |
|---|---|
| pipeline/methods/prototype_weights.py | Clustering, relaxed Sinkhorn, fixed weights, per-class diagnostics |
| pipeline/methods/finetuning.py | Checkpoint loading and new prototype_finetuning entry point |
| pipeline/methods/reweighting.py | Shared frozen-feature weighted classifier loop |
| pipeline/train.py | Method registration |
| configs/class_balanced_finetuning.json | Class balance control |
| configs/cluster_balanced_finetuning.json | Hard cluster balance control |
| configs/sinkhorn_finetuning.json | Relaxed Sinkhorn variant |
| configs/uniform_finetuning.json | Uniform continuation control |
| tests/test_prototype_weights.py | Weighting, fallback, convergence, checkpoint checks |
| docs/reweighting.md | Implementation settings and execution commands |

The implementation suite passed 206 CPU tests. The system Python lacked torch.
Checks ran using the existing environment at
/home/dzzz83/Documents/code/expert_method-local-archive-2026-11-05/.venv/bin/python.
That environment was used only to run checks; archived code is not imported by
the pipeline.

Later local additions are scripts/kaggle_prototype_comparison.py,
tests/test_kaggle_prototype_comparison.py, and the Kaggle section of
docs/reweighting.md. These are still uncommitted and unpushed as of this handoff.
The two Kaggle orchestration tests passed after review fixes.
Check current Git status before changing anything; .agents/ and the earlier
handoff files are user-owned untracked paths.

## Kaggle workflow

The user's GPU server became unavailable, so they moved training to Kaggle.
The runner pins scientific source commit 7ff1eda, reads Kaggle secret
GITHUB_TOKEN, uses GPU 0, and pushes artifacts to the separate
prototype-weighting-results branch after completed jobs. It prepares the
retained training pool, trains a seed-0 baseline unless an attached compatible
checkpoint is configured, then trains the four continuations and evaluates.

Kaggle paths are /kaggle/working/prototype-code,
/kaggle/working/prototype-results, and /kaggle/working/prototype-data.
The optional CIFAR input path is
/kaggle/input/datasets/dzzz833/cifar100. Downloads are enabled as a fallback.
Only Git subprocesses receive the token; the remote URL and askpass file do not
contain it. Completed checkpoints can be restored across sessions. Interrupted
training jobs restart from their input checkpoint; there is no epoch-level resume.
Review fixed partial optional checkpoint copies and evaluation metrics by
publishing them through a temporary file only after success.

The local runner itself was not executed inside Kaggle by the assistant.
The user subsequently ran training and published results. The assistant verified
the remote result files directly using the GitHub connector.

## Verified Kaggle results

On November 15, 2026, GitHub branch prototype-weighting-results contained all
five final checkpoints, histories, metrics, and comparison.json.
Its final summary commit was 883ca33813e7c8f153e231a25b7f2666295180f4,
with message "save prototype comparison summary".

| Run | Accuracy | Head recall | Medium recall | Tail recall |
|---|---:|---:|---:|---:|
| Baseline | 38.36% | 65.66% | 36.49% | 8.70% |
| Uniform | 38.40% | 65.63% | 36.63% | 8.70% |
| Class-balanced | 38.74% | 65.51% | 37.11% | 9.40% |
| Cluster-balanced | 38.73% | 65.51% | 37.09% | 9.40% |
| Sinkhorn | 38.73% | 65.51% | 37.09% | 9.40% |

This is one seed. Sinkhorn added no observed gain over class-balanced weighting
under these settings. Its gain over the Kaggle baseline was 0.37 percentage
points, while class-balanced weighting gained 0.38 points. Class-balanced
weighting had one more correct prediction than Sinkhorn. Equal aggregate
metrics do not establish identical predictions.

These are new Kaggle results. Keep them distinct from the earlier server
experiment in ot_reweighting_debug_handoff.md, which had a 38.41% baseline
and tested a different OT implementation.

- Results https://github.com/Dzzz83/MoE_Imbalance/tree/prototype-weighting-results
- Summary https://github.com/Dzzz83/MoE_Imbalance/blob/prototype-weighting-results/comparison.json

## Open investigation

The user is trying to understand why this use of Sinkhorn would help and whether
it actually contributes. The metric table alone does not establish the reason
it failed to add value.

If asked to diagnose further, inspect remote histories and prototype diagnostics.
Check how many classes actually formed two valid clusters, within-class weight
variation, Sinkhorn convergence, and classifier_parameter_delta_l2. Compare
these with the controls before proposing changes to the classifier schedule,
features, or weighting objective. Those diagnostic contents have not yet been
inspected in this chat.

Do not tune against the test scores or promise that stronger augmentation,
more seeds, or different Sinkhorn parameters will improve accuracy.

## Suggested next-chat prompt

Read docs/prototype_weighting_handoff.md. Continue helping me understand the
prototype Sinkhorn weighting method and its Kaggle results. Keep explanations
short and simple. Explain one idea at a time and use the saved diagnostics before
claiming a cause or suggesting another experiment.
