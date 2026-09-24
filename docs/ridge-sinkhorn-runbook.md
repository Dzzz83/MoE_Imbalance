# Ridge/Sinkhorn staged study: seed 78, outer fold 0

Development analysis was run locally from the validated Task 3C aligned OOF
artifact. The study used inner folds 1–3 (6,507 rows) only for supervised
targets, Ridge fitting, feature scaling, class weights, and dual prices.
Inner fold 0 (2,170 rows) supplied the prespecified development selection
metrics. Task 3C had previously inspected that selection fold descriptively.
No reserved outer rows or balanced CIFAR-100 test images were read.

## Development result

| Selection-fold method | BA | Tail |
|:--|--:|--:|
| Uniform original logits | 37.2941% | 10.8333% |
| Frozen `fixed_006` | 37.9474% | 13.0556% |
| Frozen `fixed_007` | 37.2401% | 14.7222% |
| Highlighted contribution Ridge, no OT | 38.0458% | 10.8333% |
| Highlighted Ridge, frozen OT price, rho 10 | 37.9514% | 14.7222% |
| Highlighted Ridge, prior-only global bias | 38.4867% | 14.7222% |
| **Locked residual Ridge, no OT** | **38.0374%** | **13.0556%** |

None of the prespecified frozen-price strengths improved both BA and Tail
over the same highlighted Ridge kernel, so the OT gate failed. The
strength-1 residual+OT result is diagnostic only. The locked candidate uses
confidence-only features, hinge-oracle penalty 1, Ridge alpha 0.1, class
weight gamma 0, residual scale 1, anchor `fixed_007`, Euclidean simplex
projection, and weighted original logits. Its selection BA is 0.0901
percentage points above `fixed_006` with equal Tail. The prior-only bias
control exceeds this candidate on both metrics, so these development results
do not establish a benefit from adaptive routing beyond a global composition
change.

The immutable artifacts are under
`artifacts/oof/ridge_sinkhorn_v3/{allocation_v1,residual_v1,selection_v1,selected_diagnostics_v1,locked_outer_v1}`.
Earlier local `v1` and `v2` directories are retained as implementation
verification records. Their selection arrays and selected metrics match `v3`;
`v3` adds the complete source and lock hashes and is the only outer-evaluation
input.
The lock was written after the selection decision and after refitting only
the selected model on all four inner OOF folds. These local numbers are
exploratory and do not establish improvement on the reserved outer fold.

## Kaggle: four full outer expert jobs

Run each command in a checkout containing the canonical training index data,
using the same `artifacts/oof` root and experiment ID. Omit `--inner-fold`;
that omission selects the reserved outer evaluation role. Each config has the
frozen 200-epoch recipe.

```bash
python scripts/run_oof.py --config configs/ce.yaml --expert ce --seed 78 --outer-fold 0 --experiment-id ridge_sinkhorn_outer_s78_o0 --device cuda --execute-full
python scripts/run_oof.py --config configs/lal.yaml --expert logit_adjusted --seed 78 --outer-fold 0 --experiment-id ridge_sinkhorn_outer_s78_o0 --device cuda --execute-full
python scripts/run_oof.py --config configs/balanced_softmax.yaml --expert balanced_softmax --seed 78 --outer-fold 0 --experiment-id ridge_sinkhorn_outer_s78_o0 --device cuda --execute-full
python scripts/run_oof.py --config configs/mixup.yaml --expert mixup --seed 78 --outer-fold 0 --experiment-id ridge_sinkhorn_outer_s78_o0 --device cuda --execute-full
```

Return all four run directories, including their final checkpoints,
prediction JSON, metadata, and resolved configs. With those artifacts in
`artifacts/oof/ridge_sinkhorn_outer_s78_o0`, the locked evaluation command is:

```bash
.venv/bin/python scripts/run_ridge_sinkhorn_outer.py
```

It validates every completed expert run, evaluates the one locked classifier
against uniform and the frozen references, and writes paired metrics and
10,000-replicate class/sample hierarchical bootstrap intervals. The outer
point-estimate gate controls whether full five-fold, three-seed nested OOF
training is justified. The original balanced test set remains excluded.
