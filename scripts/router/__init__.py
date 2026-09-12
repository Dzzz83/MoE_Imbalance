"""
Router framework for CIFAR-100-LT — **parameter-free mechanisms only**.

Why only four
-------------
This project has no validation split: experts train on the full 10,847-sample
long-tailed training set, so no honest held-out labels exist anywhere. Any
mechanism that fitted parameters would be fitting on the test set. The five
fitted routers (correctness trust meters, pairwise comparators, feature
clustering, learned gates, selective thresholds) were therefore removed; their
measured results are preserved in `records/routing_mechanism.md`.

Every router inherits from ``BaseRouter`` and implements:
  - predict(logits, features) → np.ndarray (expert index per sample)

There is deliberately **no** ``train()``/``fit()``/``calibrate()`` method.

Surviving registry — none of these fit anything:

    Uniform     : mean of expert logits, then argmax
    Product     : geometric mean of expert probabilities
    Confidence  : argmax of raw max-softmax confidence
    TTA         : a parameter-free router over TTA-averaged logits

The candidate set is frozen in `records/routing-preregistration.md` before any
test-set evaluation, so choosing among these rules cannot become
selection-on-test.
"""

from __future__ import annotations

from collections import OrderedDict

from scripts.router.base import BaseRouter
from scripts.router.uniform import UniformRouter
from scripts.router.probability import ProbabilityAverageRouter
from scripts.router.confidence import ConfidenceRouter
from scripts.router.tta import TTARouter

#: The complete, frozen set of parameter-free routers, in reporting order.
#: `Uniform` averages logits; `Probability` averages softmax probabilities.
#: Both are reported because logit and probability ensembling are known to
#: differ on imbalanced data (Buchanan et al., NeurIPS 2023 Heavy Tails).
#: `ProductRouter` was removed: `prod_e softmax(z_e)` is proportional to
#: `exp(sum_e z_e)` whose normaliser is class-independent, so its argmax is
#: exactly the argmax of the mean logits — the same classifier as `Uniform`.
ROUTERS: 'OrderedDict[str, type[BaseRouter]]' = OrderedDict([
    ('Uniform', UniformRouter),
    ('Probability', ProbabilityAverageRouter),
    ('Confidence', ConfidenceRouter),
    ('TTA', TTARouter),
])

__all__ = [
    'BaseRouter',
    'UniformRouter',
    'ProbabilityAverageRouter',
    'ConfidenceRouter',
    'TTARouter',
    'ROUTERS',
]
