# Research Constraints, Questions, and Literature

This document combines measured limitations with the literature that shaped
the project. Detailed numbers are owned by [results.md](results.md) for the
full-data track and [oof-results.md](oof-results.md) for OOF development and
the locked outer evaluation. Canonical data and evaluation rules are in
[protocol.md](protocol.md). Retired Ridge/Sinkhorn proposals remain in
[archive/superseded-research-proposals.md](archive/superseded-research-proposals.md).

## Measured constraints

### Complementarity does not establish predictable routing

The four experts make distinct correct predictions, including exclusive
correct predictions in the OOF development data. Several fixed convex logit
mixtures improve both BA and Tail over uniform. A label-dependent soft
feasibility oracle also finds correcting mixtures when all expert top-1
predictions are wrong. These findings show complementarity and possible
headroom; neither shows that useful per-image weights can be predicted from
inference-time information. See [OOF results](oof-results.md).

### The tested Ridge models favor Mixup on Tail rows

The highlighted confidence-only Ridge gives Mixup the highest weight on
99.45% of Tail development rows. The saved predictor ranks Mixup higher on
102 of the 103 Tail rows where LAL's measured contribution target is larger.
Retrospective confidence and disagreement signals identify some cases where
LAL or BalancedSoftmax is correct while Mixup is wrong, but the saved weights
do not reduce Mixup on those patterns. The diagnostic subsets are small and
were not selected as new features.

### Contribution prediction error and classification are different outcomes

Across 15 matched comparisons, the full 13-feature Ridge reduces contribution
MSE relative to confidence-only Ridge in every setting, including Tail. Its
predefined primary classification result is lower on both BA and Tail. The
Task 3F-E contribution target is a local true-class log-probability change;
positive target values rarely correct an incorrect uniform top-1 result at
the tested finite perturbations. Neither that target nor the retrospective
margin quantity is an approved routing target.

### Tail evidence is sparse

The development partition contains 183 Tail images over 30 classes, and 14
classes have no correct top-1 prediction from any expert in that partition.
Small changes to a few predictions can change macro Tail accuracy materially.
Zero observed coverage is not evidence that a class is intrinsically
unlearnable, and the current data do not identify whether limited Tail data,
target choice, or feature choice is the main cause of Ridge behavior.

### Fixed references and evaluation limits bind current conclusions

The locked no-OT residual Ridge improved over outer uniform by point estimate,
but its paired 95% intervals include zero and `fixed_007` and `fixed_010`
each exceed it on both BA and Tail. Frozen-price Sinkhorn did not pass its
development gate. The prespecified expansion gate therefore failed; the
full five-fold, three-seed experiment was not run.

The OOF evidence uses one seed and one outer fold. Inner fold 0 had earlier
descriptive exposure, and the experts producing fit and selection rows have
overlapping training populations. The original balanced test set has been
used for historical full-data work and is unavailable for ordinary method
development. These constraints limit generalization; they do not prove that
all future routing methods will fail.

### Historical approaches have protocol-specific failures

- On the old split, confidence ranked the correct expert below another
  expert in 83.9% of 596 samples considered savable by that rule. The old
  split is superseded and its metrics are not comparable to the current
  protocol.
- The retired DACE cascade produced an anti-predictive correctness score
  (AUROC 0.327), expert scores on incompatible scales, and a three-expert
  uniform result below its corresponding original pool. This diagnoses that
  implementation and objective, not every cascade or learned router.
- A gate fitted on the old split collapsed onto one expert. The tested
  Correctness, Pairwise, Cluster, and Selective routers did not retain their
  old-split gains under the canonical protocol.
- The historical TTA BA/Tail row is invalid because the earlier augmentation
  path had incorrect normalized-tensor padding and batch-shared crop
  semantics. Historical routing calibration fields for Uniform, Confidence,
  and TTA are invalid because they used a class distribution inconsistent
  with the classifier. No replacement values are reported.
- Product-of-experts is not a distinct classifier here:
  `prod_e softmax(z_e)` has the same argmax as summed or averaged logits.

The full mechanism catalogue, with protocol labels and historical metrics,
is in [routing_mechanism.md](../records/routing_mechanism.md). Historical
audit provenance is in [archive/bugfix-report.md](archive/bugfix-report.md).

## Open questions and research boundaries

1. **Can useful contributions be predicted?** A future method must separate
   router fitting and selection, use inference-time features only, and be
   assessed on an unconsumed evaluation population.
2. **Which signals survive independent evaluation?** Confidence,
   disagreement, predicted class group, local log-probability contribution,
   and margin diagnostics are retrospective findings. None is a selected
   feature set.
3. **Does an improved target help classification?** Lower target MSE did not
   improve the primary BA–Tail result in the tested feature comparison. A
   different target needs a frozen definition and direct classifier-level
   controls.
4. **Can a router beat fixed composition?** Compare any adaptive method with
   uniform logits and the frozen fixed-weight references, including their
   BA–Tail trade-off. A point estimate over uniform alone is insufficient.
5. **Do results transfer across folds, seeds, and expert populations?** The
   completed outer result does not answer this. Fold 0 is consumed, so a
   replacement method needs a separately declared development and evaluation
   plan.
6. **What remains of Sinkhorn?** The tested frozen-price allocation did not
   pass its gate. This rules out the tested scores, priors, and settings; it
   does not establish that every optimal-transport formulation is ineffective.

Any future study must follow the data-role, leakage, artifact, and test-access
rules in [protocol.md](protocol.md). Do not describe a hypothesis as a
validated improvement before its prespecified evaluation is complete.

## Literature and project decisions

### Long-tail experts and diversity

RIDE (Wang et al., 2021) combines distribution-aware experts with a learned
router; SADE/TADE (Zhou et al., 2022) use diverse objectives and a
self-supervised aggregation task. ACE (Cai et al., 2021) adds explicit
complementarity, BalPoE/BPCE (Aimar et al., 2023) combines calibrated expert
scores, and MDCS (Zhao et al., 2023) motivates controlled diversity. These
works motivate expert mixtures, but use architectures or training signals
that this project has not reproduced. The measured agreement between experts
alone does not establish predictable specialization.

### Routing and mixtures of experts

Vision MoE router studies and methods such as Divide, Weight, and Route and
RICASSO use held-out supervision or other learned routing signals. The
original full-data track has no validation split, motivating the separate
OOF pipeline before fitted routers are considered. Sparsely-Gated MoE and
load-balancing work inform allocation constraints; they do not replace an
honest data-role design.

### Losses, Mixup, and ensembling

The pool uses CE, logit adjustment, BalancedSoftmax, and vanilla Mixup.
LDAM-DRW informed the shared training schedule. Focal, supervised
contrastive, KCL, TSC, and PaCo are reviewed alternatives, not current
experts. Long-tail Mixup variants such as LOB Mixup, MAMix, and Remix
motivate concern that vanilla Mixup can suppress minority labels; measured
Mixup is useful for Head accuracy and calibration but weak on Tail, so it is
included for complementarity rather than treated as a long-tail solution.

Long-tail ensemble work motivates reporting logit and probability averaging
separately. They differ on this imbalanced pool. The product-of-experts
identity removes one redundant baseline. Ridge regression motivates the
tested supervised contribution model, while Sinkhorn and optimal-transport
work motivate allocation experiments; the current evidence does not validate
an adaptive routing gain.

## References

1. Zhang, H. et al. (2018). *mixup: Beyond Empirical Risk Minimization*. ICLR.
   [code](https://github.com/facebookresearch/mixup-cifar10)
2. Cai, J., Wang, Y., Hwang, J.-N. (2021). *ACE: Ally Complementary Experts*.
   ICCV. [arXiv:2108.02385](https://arxiv.org/abs/2108.02385)
3. Zhou, Z. et al. (2022). *SADE: Self-Supervised Aggregation of Diverse
   Experts*. NeurIPS. [arXiv:2107.09249](https://arxiv.org/abs/2107.09249)
4. Aimar, E. S. et al. (2023). *Balanced Product of Calibrated Experts*.
   CVPR. [arXiv:2206.05260](https://arxiv.org/abs/2206.05260)
5. Zhao, Q. et al. (2023). *MDCS: More Diverse Experts with Consistency
   Self-Distillation*. ICCV. [arXiv:2308.09917](https://arxiv.org/abs/2308.09917)
6. Liu, Z., Blondel, M. (2024). *Routers in Vision Mixture of Experts*.
   TMLR. [OpenReview](https://openreview.net/forum?id=5vSXd8cogo)
7. Ghosh, A. et al. (2021). *ELF: An Early-Exiting Framework for Long-Tailed
   Classification*. ICASSP. [arXiv:2006.11979](https://arxiv.org/abs/2006.11979)
8. Menon, A. K. et al. (2021). *Long-tail learning via logit adjustment*.
   ICLR. [arXiv:2007.07314](https://arxiv.org/abs/2007.07314) ·
   [code](https://github.com/google-research/google-research/tree/master/logit_adjustment)
9. Ren, J. et al. (2020). *Balanced Meta-Softmax*. NeurIPS.
   [arXiv:2007.10740](https://arxiv.org/abs/2007.10740) ·
   [code](https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification)
10. Cao, K. et al. (2019). *LDAM: Label-Distribution-Aware Margin Loss*.
    NeurIPS. [arXiv:1906.07413](https://arxiv.org/abs/1906.07413) ·
    [code](https://github.com/kaidic/LDAM-DRW)
11. Kang, B. et al. (2020). *Decoupling Representation and Classifier*.
    ICLR. [arXiv:1910.09217](https://arxiv.org/abs/1910.09217)
12. Wei, X. et al. (2025). *Divide, Weight, and Route*.
    [arXiv:2508.19630](https://arxiv.org/abs/2508.19630)
13. Zhang, X. et al. (2024). *RICASSO*.
    [arXiv:2410.10548](https://arxiv.org/abs/2410.10548)
14. Khosla, P. et al. (2020). *Supervised Contrastive Learning*. NeurIPS.
    [arXiv:2004.11362](https://arxiv.org/abs/2004.11362)
15. Shazeer, N. et al. (2017). *Outrageously Large Neural Networks*. ICLR.
    [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)
16. Li, T. et al. (2022). *Targeted Supervised Contrastive Learning*. CVPR.
    [arXiv:2111.13998](https://arxiv.org/abs/2111.13998)
17. Kang, B. et al. (2021). *K-positive Contrastive Learning*. ICLR.
    [arXiv:2102.10078](https://arxiv.org/abs/2102.10078)
18. Cui, J. et al. (2021). *Parametric Contrastive Learning*. ICCV.
    [arXiv:2107.12028](https://arxiv.org/abs/2107.12028)
19. Lin, T.-Y. et al. (2017). *Focal Loss*. ICCV.
    [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
20. He, K. et al. (2020). *MoCo v2*. CVPR.
    [arXiv:2003.04297](https://arxiv.org/abs/2003.04297)
21. Zhang, S., Chen, C., Zhang, X., Peng, S. (2021).
    *Label-Occurrence-Balanced Mixup*.
    [arXiv:2110.04964](https://arxiv.org/abs/2110.04964)
22. Cheng, W.-C., Mai, T.-H., Lin, H.-T. et al. (2023). *From SMOTE to Mixup
    for Deep Imbalanced Classification* (MAMix). TAAI.
    [arXiv:2308.15457](https://arxiv.org/abs/2308.15457) ·
    [code](https://github.com/ntuclab/imbalanced-DL)
23. Chou, H.-P. et al. (2020). *Remix: Rebalanced Mixup*. ECCV Workshops.
24. Verma, V. et al. (2019). *Manifold Mixup*. ICML.
    [arXiv:1806.05236](https://arxiv.org/abs/1806.05236)
25. Buchanan, E. K., Pleiss, G., Wang, Y., Cunningham, J. P. (2023).
    *The Effects of Ensembling on Long-Tailed Data*. NeurIPS Heavy Tails
    Workshop. [code](https://github.com/ekellbuch/longtail_ensembles)
26. Tassi, C. R., Gawlikowski, J. *The impact of averaging logits over
    probabilities on ensembles of neural networks*.
    [CEUR-WS Vol-3215](http://sunsite.informatik.rwth-aachen.de/Publications/CEUR-WS/Vol-3215/19.pdf)
27. Wang, X. et al. (2021). *RIDE: Long-Tailed Recognition by Routing Diverse
    Distribution-Aware Experts*. ICLR.
    [arXiv:2010.01809](https://arxiv.org/abs/2010.01809)
28. Sinkhorn, R., Knopp, P. (1967). *Concerning Nonnegative Matrices and
    Doubly Stochastic Matrices*. Pacific Journal of Mathematics 21(2):343–348.
    [DOI](https://doi.org/10.2140/pjm.1967.21.343)
29. Cuturi, M. (2013). *Sinkhorn Distances: Lightspeed Computation of Optimal
    Transport*. NeurIPS.
    [proceedings](https://proceedings.neurips.cc/paper_files/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html)
30. Peyré, G., Cuturi, M. (2019). *Computational Optimal Transport*.
    Foundations and Trends in Machine Learning.
    [book](https://optimaltransport.github.io/book/)
31. Hoerl, A. E., Kennard, R. W. (1970). *Ridge Regression: Biased
    Estimation for Nonorthogonal Problems*. Technometrics 12(1):55–67.
    [DOI](https://doi.org/10.1080/00401706.1970.10488634)
32. Lewis, M. et al. (2021). *BASE Layers: Simplifying Training of Large,
    Sparse Models*. ICML. [PMLR](https://proceedings.mlr.press/v139/lewis21a.html)
33. Chizat, L., Peyré, G., Schmitzer, B., Vialard, F.-X. (2018). *Scaling
    Algorithms for Unbalanced Optimal Transport Problems*. Mathematics of
    Computation. [arXiv:1607.05816](https://arxiv.org/abs/1607.05816)
34. Chapel, L., Flamary, R., Wu, H., Févotte, C., Gasso, G. (2021).
    *Unbalanced Optimal Transport through Non-negative Penalized Linear
    Regression*. NeurIPS.
    [proceedings](https://proceedings.neurips.cc/paper_files/paper/2021/hash/c3c617a9b80b3ae1ebd868b0017cc349-Abstract.html)
35. Blondel, M., Seguy, V., Rolet, V. (2018). *Smooth and Sparse Optimal
    Transport*. AISTATS. [PMLR](https://proceedings.mlr.press/v84/blondel18a.html)
36. Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). *On Calibration
    of Modern Neural Networks*. ICML.
    [PMLR](https://proceedings.mlr.press/v70/guo17a.html)
37. Li, Z., Li, Z., Zhou, T. (2025). *R2-T2: Re-Routing in Test-Time for
    Multimodal Mixture-of-Experts*. ICML.
    [PMLR](https://proceedings.mlr.press/v267/li25bc.html)
38. Nguyen, D. A. et al. (2026). *Selective Sinkhorn Routing for Improved
    Sparse Mixture of Experts*. ICML 2026 AdaptFM Workshop.
    [OpenReview](https://openreview.net/pdf?id=qRQU6W1vJ4) ·
    [arXiv:2511.08972](https://arxiv.org/abs/2511.08972)
39. Tian, X. et al. (2026). *Region-Graph Optimal Transport Routing for
    Mixture-of-Experts Whole-Slide Image Classification (ROAM)*. preprint.
    [arXiv:2604.07298](https://arxiv.org/abs/2604.07298)
