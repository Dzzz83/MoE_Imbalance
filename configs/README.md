# Configuration layout

`experts/` contains the four human-edited training recipes (`ce`,
`logit_adjusted`, `balanced_softmax`, and `mixup`). These are the only source of
truth for expert training parameters.

`studies/` contains scientific study definitions: protocol, folds, seeds,
search grids, metrics, and references to the expert recipes. Scientific fields
contribute to the frozen study identity.

`profiles/` contains machine-specific runtime settings such as dataset and
artifact paths, reuse mounts, device, shard selection, and per-session job
limits. `profiles/kaggle.yaml` is a committed template, not a file to edit for
an individual Kaggle run. Copy it outside the checkout (for example to
`/kaggle/working/rs3-profile.yaml`), edit that copy, and pass it with
`--profile "$PROFILE"`; this keeps the checkout clean for `study freeze` and
the frozen commit requirement.

Runtime profiles are excluded from the frozen study identity except for the
sorted logical names in `reuse_roots`. Choose whether reuse is enabled before
the first freeze and keep those names unchanged afterward. The mounted paths
for those names, `data_root`, `run_root`, device, shard settings, `max_jobs`,
and bundle input/output paths may vary across sessions without changing that
identity, provided the selected reuse roots still pass the recorded audit.
