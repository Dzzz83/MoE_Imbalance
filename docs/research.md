# Research constraints and open questions

Use the [README](../README.md) for the current headline and selected
references. Full-data measurements, Task 3C–3F findings, and the completed
three-seed matrix remain in their respective
[result records](README.md#historical-evidence-and-records). Data roles and
metric rules are in [protocol.md](protocol.md).

## Measured constraints

- **Expert complementarity does not establish a router.** Fixed mixtures
  improved both OOF development metrics over uniform logits. The
  label-dependent feasibility diagnostic measures possible mixtures, not
  inference-time choices.
- **Contribution prediction is not classification.** The Ridge target is a
  local change in true-class log probability. Lower target error and
  retrospective signal associations did not produce a selected adaptive
  advantage. Detailed measurements are in the
  [Task 3F record](archive/oof-task-history.md).
- **Tail evidence is sparse.** The seed-78 development partition had 183 Tail
  rows across 30 classes; 14 classes had no expert correct at top-1. These
  observations do not mean those classes are unlearnable.
- **Earlier expansion and v1 performance gates failed.** The locked seed-78
  outer-fold-0 result failed its fixed-reference gate, and all five v1
  methods failed the all-seed gate. These retrospective studies do not
  establish how a router transfers to a new population.

## Open questions

1. Can inference-time signals predict useful expert contributions on an
   unconsumed population?
2. Do confidence, disagreement, predicted class group, contribution, or margin
   signals survive independent evaluation?
3. Can another target improve classification rather than only target error?
4. Can an adaptive method pass the all-seed gate and avoid domination by fixed
   mixtures on both BA and Tail?
5. How do results transfer to a new population or unseen training seeds and
   folds? The completed study spans three seeds and five folds but reuses the
   same canonical population.
6. Which other transport formulations, if any, help allocation and
   classification? The two completed studies tested specific frozen variants,
   not every transport method.

Any future study needs a new declared plan and separated fitting, selection,
and evaluation roles under the safeguards in [protocol.md](protocol.md). Do
not present an untested hypothesis as a validated improvement.

## Further reading

The [README references](../README.md#key-references) cover OOF stacking, Ridge,
Sinkhorn, RIDE, logit adjustment, BalancedSoftmax, and Mixup. These additional
works broaden the related-literature map:

- **Expert diversity:** ACE, Cai et al. (2021),
  [arXiv:2108.02385](https://arxiv.org/abs/2108.02385); SADE, Zhou et al.
  (2022), [arXiv:2107.09249](https://arxiv.org/abs/2107.09249); BalPoE, Aimar
  et al. (2023), [arXiv:2206.05260](https://arxiv.org/abs/2206.05260); MDCS,
  Zhao et al. (2023), [arXiv:2308.09917](https://arxiv.org/abs/2308.09917).
- **Routing:** Routers in Vision MoE, Liu and Blondel (2024),
  [OpenReview](https://openreview.net/forum?id=5vSXd8cogo); Divide, Weight,
  and Route, Wei et al. (2025),
  [arXiv:2508.19630](https://arxiv.org/abs/2508.19630); RICASSO, Zhang et al.
  (2024), [arXiv:2410.10548](https://arxiv.org/abs/2410.10548).
- **Other long-tail losses:** LDAM, Cao et al. (2019),
  [arXiv:1906.07413](https://arxiv.org/abs/1906.07413); LOB Mixup, Zhang et
  al. (2021), [arXiv:2110.04964](https://arxiv.org/abs/2110.04964); MAMix,
  Cheng et al. (2023), [arXiv:2308.15457](https://arxiv.org/abs/2308.15457);
  Remix, Chou et al. (2020), ECCV Workshops.
- **Calibration and transport:** Guo et al. (2017), [PMLR calibration
  study](https://proceedings.mlr.press/v70/guo17a.html); Peyré and Cuturi
  (2019), [Computational Optimal Transport](https://optimaltransport.github.io/book/).
