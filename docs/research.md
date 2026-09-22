# Research — Literature and Current Direction

This document records the literature that shaped the project and how it relates
to the measured evidence. Superseded Ridge/Sinkhorn implementation proposals
are preserved separately in
[archive/superseded-research-proposals.md](archive/superseded-research-proposals.md).

Related project evidence:
[results.md](results.md), [oof-results.md](oof-results.md),
[problem.md](problem.md), and
[records/routing_mechanism.md](../records/routing_mechanism.md).

## Current research-status summary

- Legitimate held-out OOF predictions have been collected for the four existing
  experts on one outer fold and one training seed.
- Fixed-weight feasibility has been investigated on the permitted OOF
  router-fitting partition.
- The convex-logit soft-mixture oracle has been implemented and evaluated.
- The oracle demonstrates additional theoretical headroom beyond hard
  selection, but it does not predict weights.
- Predictability of useful expert weighting has not been established.
- Ridge and Sinkhorn remain unimplemented.

The current scientific question is:

> Can a model predict useful expert contributions from inference-time
> information using legitimate OOF supervision?

Ridge is a candidate simple predictive model. Its target, features,
regularization, baseline comparisons and internal validation procedure have not
been frozen. Sinkhorn is an allocation mechanism, not a source of routing
signal, and should be investigated separately only after a useful suitability
signal is established.

## 1. Multi-expert architectures for long-tailed recognition

### RIDE — Long-Tailed Recognition by Routing Diverse Distribution-Aware Experts

Wang et al., ICLR 2021
[arXiv:2010.01809](https://arxiv.org/abs/2010.01809), use shared-backbone
experts with expert-specific batch normalization and distribution-aware
sampling. A learned router assigns samples to experts. RIDE is the reference
for the claim that routing diverse experts can improve long-tail recognition;
its sampling diversity and shared-backbone design differ from this repository's
separate loss-trained experts.

### SADE / TADE — Self-Supervised Aggregation of Diverse Experts

Zhou et al., NeurIPS 2022
[arXiv:2107.09249](https://arxiv.org/abs/2107.09249), combine experts trained
with different losses and use a self-supervised task to weight them. Their
reported comparison suggests that controlled sampling diversity can add more
than simply adding related losses. This informed the concern that loss
diversity alone may not create predictable routing diversity.

### ACE, BalPoE, MDCS and ELF

ACE (Cai et al., ICCV 2021,
[arXiv:2108.02385](https://arxiv.org/abs/2108.02385)) trains complementary
heads with an explicit diversity objective and combines them by averaging.
BalPoE/BPCE (Aimar et al., CVPR 2023,
[arXiv:2206.05260](https://arxiv.org/abs/2206.05260)) provides a long-tail
precedent for combining calibrated expert scores in log space. MDCS (Zhao et
al., ICCV 2023, [arXiv:2308.09917](https://arxiv.org/abs/2308.09917)) supports
controlled rather than unconstrained diversity. ELF (Ghosh et al., ICASSP 2021,
[arXiv:2006.11979](https://arxiv.org/abs/2006.11979)) motivates confidence
routing as a baseline.

The project used these ideas as motivation, not as claims that a specific
architecture was reproduced.

## 2. Routers and mixture-of-experts

Routers in Vision MoE (Liu and Blondel, TMLR 2024,
[OpenReview](https://openreview.net/forum?id=5vSXd8cogo)) reports that learned
routers can benefit from balanced validation supervision. That supervision is
not available for the original full-data protocol, which is why the OOF
pipeline is required before fitting a router.

Divide, Weight, and Route (Wei et al., 2025,
[arXiv:2508.19630](https://arxiv.org/abs/2508.19630)) uses a difficulty signal
to route between loss-specialized experts. RICASSO (Zhang et al., 2024,
[arXiv:2410.10548](https://arxiv.org/abs/2410.10548)) uses an RL router with
balanced validation supervision. Sparsely-Gated MoE (Shazeer et al., ICLR 2017,
[arXiv:1701.06538](https://arxiv.org/abs/1701.06538)) provides load-balancing
precedent. None is implemented here, and none removes the requirement for an
honest development protocol.

## 3. Long-tail loss functions and expert choice

The expert pool uses CE, LAL, BalancedSoftmax and Mixup:

| Loss | Literature | Role here |
|:--|:--|:--|
| CE | ERM baseline | Used |
| LAL | Menon et al., ICLR 2021, [arXiv:2007.07314](https://arxiv.org/abs/2007.07314) | Used, tau = 1 |
| BalancedSoftmax | Ren et al., NeurIPS 2020, [arXiv:2007.10740](https://arxiv.org/abs/2007.10740) | Used |
| Mixup | Zhang et al., ICLR 2018 | Used, alpha = 1 |

LDAM (Cao et al., NeurIPS 2019,
[arXiv:1906.07413](https://arxiv.org/abs/1906.07413)) supplied the shared
training schedule. Focal Loss (Lin et al., ICCV 2017,
[arXiv:1708.02002](https://arxiv.org/abs/1708.02002)), SupCon (Khosla et al.,
NeurIPS 2020, [arXiv:2004.11362](https://arxiv.org/abs/2004.11362)), KCL
(Kang et al., ICLR 2021,
[arXiv:2102.10078](https://arxiv.org/abs/2102.10078)), TSC (Li et al., CVPR
2022, [arXiv:2111.13998](https://arxiv.org/abs/2111.13998)) and PaCo (Cui et
al., ICCV 2021, [arXiv:2107.12028](https://arxiv.org/abs/2107.12028)) informed
possible alternatives but are not current experts.

Earlier literature suggested that contrastive or sampling diversity might
produce lower agreement. In this project, measured full-data agreement did not
separate same-objective from different-objective pairs clearly, and OOF
diagnostics nevertheless found complementary correctness. The appropriate
conclusion is that agreement alone is not a sufficient routing criterion.

## 4. The Mixup family

Vanilla Mixup (Zhang et al., ICLR 2018) improves calibration and generalization
but can suppress minority labels under random pairing. Label-Occurrence-Balanced
Mixup (LOB Mixup, Zhang et al. 2021,
[arXiv:2110.04964](https://arxiv.org/abs/2110.04964)) explains this mechanism
and proposes balanced samplers. MAMix (Cheng, Mai and Lin, TAAI 2023,
[arXiv:2308.15457](https://arxiv.org/abs/2308.15457)) modifies the label factor
using class-frequency information. Remix (Chou et al., ECCV Workshops 2020) and
Manifold Mixup (Verma et al., ICML 2019,
[arXiv:1806.05236](https://arxiv.org/abs/1806.05236)) are related alternatives.

The measured full-data Mixup expert is well calibrated and strong on Head but
weak on Tail. This makes it useful for studying complementarity, not evidence
that vanilla Mixup is a complete long-tail solution. Rebalanced Mixup variants
remain unimplemented and are not the current next step.

## 5. Ensembling

Buchanan et al., The Effects of Ensembling on Long-Tailed Data (NeurIPS 2023
Heavy Tails Workshop,
[code](https://github.com/ekellbuch/longtail_ensembles)), motivates reporting
both logit and probability averaging because they can differ on imbalanced data.
Tassi and Gawlikowski study the same distinction in
[this survey/report](http://sunsite.informatik.rwth-aachen.de/Publications/CEUR-WS/Vol-3215/19.pdf).

The full-data track finds logit averaging stronger than probability averaging.
The OOF track also reports both baselines. A product-of-experts rule was
removed because its argmax is mathematically identical to uniform logit
averaging.

## 6. What was predicted versus measured

| Idea | Literature expectation | Measured or current interpretation |
|:--|:--|:--|
| More diverse experts should route better | RIDE, SADE, TSC | Full-data agreement alone did not establish predictable specialization |
| A learned gate should help under imbalance | Vision-MoE literature | Requires held-out supervision; OOF is now available only for development |
| Mixup improves the pool | Calibration and ensemble literature | Calibration/head complementarity improved, Tail remained weak |
| Logit and probability averaging are interchangeable | Balanced-data intuition | They differ on this imbalanced pool |
| Product-of-experts is a new baseline | Generic ensemble intuition | It is the same classifier as uniform logit averaging |
| Soft adaptive weighting may help | Convex-mixture feasibility | The OOF oracle shows existence headroom, not predictability |

## 7. Current research boundary

Before any fitted router is implemented, the project must freeze a supervised
target, inference-time features, calibration choices, regularization grid,
uniform/fixed/no-OT baselines, and the fit-versus-selection procedure. Fit and
selection must use the OOF roles in the
[nested protocol](nested-oof-protocol.md), while the outer evaluation and
original test set remain untouched.

Ridge should first answer whether useful expert suitability is predictable.
Sinkhorn should then be tested as a separate global-allocation control rather
than assumed to create signal. The old ranked variants and staged sequence are
historical hypotheses preserved in
[archive/superseded-research-proposals.md](archive/superseded-research-proposals.md);
they are not approved implementation instructions.

## References

1. Zhang, H. et al. (2018). *mixup: Beyond Empirical Risk Minimization*. ICLR. [code](https://github.com/facebookresearch/mixup-cifar10)
2. Cai, J., Wang, Y., Hwang, J.-N. (2021). *ACE: Ally Complementary Experts*. ICCV. [arXiv:2108.02385](https://arxiv.org/abs/2108.02385)
3. Zhou, Z. et al. (2022). *SADE: Self-Supervised Aggregation of Diverse Experts*. NeurIPS. [arXiv:2107.09249](https://arxiv.org/abs/2107.09249)
4. Aimar, E. S. et al. (2023). *Balanced Product of Calibrated Experts*. CVPR. [arXiv:2206.05260](https://arxiv.org/abs/2206.05260)
5. Zhao, Q. et al. (2023). *MDCS: More Diverse Experts with Consistency Self-Distillation*. ICCV. [arXiv:2308.09917](https://arxiv.org/abs/2308.09917)
6. Liu, Z., Blondel, M. (2024). *Routers in Vision Mixture of Experts*. TMLR. [OpenReview](https://openreview.net/forum?id=5vSXd8cogo)
7. Ghosh, A. et al. (2021). *ELF: An Early-Exiting Framework for Long-Tailed Classification*. ICASSP. [arXiv:2006.11979](https://arxiv.org/abs/2006.11979)
8. Menon, A. K. et al. (2021). *Long-tail learning via logit adjustment*. ICLR. [arXiv:2007.07314](https://arxiv.org/abs/2007.07314) · [code](https://github.com/google-research/google-research/tree/master/logit_adjustment)
9. Ren, J. et al. (2020). *Balanced Meta-Softmax*. NeurIPS. [arXiv:2007.10740](https://arxiv.org/abs/2007.10740) · [code](https://github.com/jiawei-ren/BalancedMetaSoftmax-Classification)
10. Cao, K. et al. (2019). *LDAM: Label-Distribution-Aware Margin Loss*. NeurIPS. [arXiv:1906.07413](https://arxiv.org/abs/1906.07413) · [code](https://github.com/kaidic/LDAM-DRW)
11. Kang, B. et al. (2020). *Decoupling Representation and Classifier*. ICLR. [arXiv:1910.09217](https://arxiv.org/abs/1910.09217)
12. Wei, X. et al. (2025). *Divide, Weight, and Route*. [arXiv:2508.19630](https://arxiv.org/abs/2508.19630)
13. Zhang, X. et al. (2024). *RICASSO*. [arXiv:2410.10548](https://arxiv.org/abs/2410.10548)
14. Khosla, P. et al. (2020). *Supervised Contrastive Learning*. NeurIPS. [arXiv:2004.11362](https://arxiv.org/abs/2004.11362)
15. Shazeer, N. et al. (2017). *Outrageously Large Neural Networks*. ICLR. [arXiv:1701.06538](https://arxiv.org/abs/1701.06538)
16. Li, T. et al. (2022). *Targeted Supervised Contrastive Learning*. CVPR. [arXiv:2111.13998](https://arxiv.org/abs/2111.13998)
17. Kang, B. et al. (2021). *K-positive Contrastive Learning*. ICLR. [arXiv:2102.10078](https://arxiv.org/abs/2102.10078)
18. Cui, J. et al. (2021). *Parametric Contrastive Learning*. ICCV. [arXiv:2107.12028](https://arxiv.org/abs/2107.12028)
19. Lin, T.-Y. et al. (2017). *Focal Loss*. ICCV. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
20. He, K. et al. (2020). *MoCo v2*. CVPR. [arXiv:2003.04297](https://arxiv.org/abs/2003.04297)
21. Zhang, S., Chen, C., Zhang, X., Peng, S. (2021). *Label-Occurrence-Balanced Mixup*. [arXiv:2110.04964](https://arxiv.org/abs/2110.04964)
22. Cheng, W.-C., Mai, T.-H., Lin, H.-T. et al. (2023). *From SMOTE to Mixup for Deep Imbalanced Classification* (MAMix). TAAI. [arXiv:2308.15457](https://arxiv.org/abs/2308.15457) · [code](https://github.com/ntuclab/imbalanced-DL)
23. Chou, H.-P. et al. (2020). *Remix: Rebalanced Mixup*. ECCV Workshops.
24. Verma, V. et al. (2019). *Manifold Mixup*. ICML. [arXiv:1806.05236](https://arxiv.org/abs/1806.05236)
25. Buchanan, E. K., Pleiss, G., Wang, Y., Cunningham, J. P. (2023). *The Effects of Ensembling on Long-Tailed Data*. NeurIPS Heavy Tails Workshop. [code](https://github.com/ekellbuch/longtail_ensembles)
26. Tassi, C. R., Gawlikowski, J. *The impact of averaging logits over probabilities on ensembles of neural networks*. [CEUR-WS Vol-3215](http://sunsite.informatik.rwth-aachen.de/Publications/CEUR-WS/Vol-3215/19.pdf)
27. Wang, X. et al. (2021). *RIDE: Long-Tailed Recognition by Routing Diverse Distribution-Aware Experts*. ICLR. [arXiv:2010.01809](https://arxiv.org/abs/2010.01809)
28. Sinkhorn, R., Knopp, P. (1967). *Concerning Nonnegative Matrices and Doubly Stochastic Matrices*. Pacific Journal of Mathematics 21(2):343–348. [DOI](https://doi.org/10.2140/pjm.1967.21.343)
29. Cuturi, M. (2013). *Sinkhorn Distances: Lightspeed Computation of Optimal Transport*. NeurIPS. [proceedings](https://proceedings.neurips.cc/paper_files/paper/2013/hash/af21d0c97db2e27e13572cbf59eb343d-Abstract.html)
30. Peyré, G., Cuturi, M. (2019). *Computational Optimal Transport*. Foundations and Trends in Machine Learning. [book](https://optimaltransport.github.io/book/)
31. Hoerl, A. E., Kennard, R. W. (1970). *Ridge Regression: Biased Estimation for Nonorthogonal Problems*. Technometrics 12(1):55–67. [DOI](https://doi.org/10.1080/00401706.1970.10488634)
32. Lewis, M. et al. (2021). *BASE Layers: Simplifying Training of Large, Sparse Models*. ICML. [PMLR](https://proceedings.mlr.press/v139/lewis21a.html)
33. Chizat, L., Peyré, G., Schmitzer, B., Vialard, F.-X. (2018). *Scaling Algorithms for Unbalanced Optimal Transport Problems*. Mathematics of Computation. [arXiv:1607.05816](https://arxiv.org/abs/1607.05816)
34. Chapel, L., Flamary, R., Wu, H., Févotte, C., Gasso, G. (2021). *Unbalanced Optimal Transport through Non-negative Penalized Linear Regression*. NeurIPS. [proceedings](https://proceedings.neurips.cc/paper_files/paper/2021/hash/c3c617a9b80b3ae1ebd868b0017cc349-Abstract.html)
35. Blondel, M., Seguy, V., Rolet, A. (2018). *Smooth and Sparse Optimal Transport*. AISTATS. [PMLR](https://proceedings.mlr.press/v84/blondel18a.html)
36. Guo, C., Pleiss, G., Sun, Y., Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks*. ICML. [PMLR](https://proceedings.mlr.press/v70/guo17a.html)
37. Li, Z., Li, Z., Zhou, T. (2025). *R2-T2: Re-Routing in Test-Time for Multimodal Mixture-of-Experts*. ICML. [PMLR](https://proceedings.mlr.press/v267/li25bc.html)
38. Nguyen, D. A. et al. (2026). *Selective Sinkhorn Routing for Improved Sparse Mixture of Experts*. ICML 2026 AdaptFM Workshop. [OpenReview](https://openreview.net/pdf?id=qRQU6W1vJ4) · [arXiv:2511.08972](https://arxiv.org/abs/2511.08972)
39. Tian, X. et al. (2026). *Region-Graph Optimal Transport Routing for Mixture-of-Experts Whole-Slide Image Classification (ROAM)*. preprint. [arXiv:2604.07298](https://arxiv.org/abs/2604.07298)
