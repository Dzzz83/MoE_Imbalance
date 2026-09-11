"""
Router framework for CIFAR-100-LT — **parameter-free mechanisms only**.

Why only four
-------------
This project has no validation split: experts train on the full 10,847-sample
long-tailed training set, so no honest held-out labels exist anywhere. Any
mechanism that fitted parameters would be fitting on the test set. The five
fitted routers (correctness trust meters, pairwise comparators, feature
clustering, learned gates, selective thresholds) were therefore removed; their
measured results are preserved in `docs/routing-results-record.md`.

Every router inherits from ``BaseRouter`` and implements:
  - predict(logits, features) → np.ndarray (expert index per sample)

There is deliberately **no** ``train()``/``fit()``/``calibrate()`` method.

Surviving registry — none of these fit anything:

    Uniform     : mean of expert logits, then argmax
    Product     : geometric mean of expert probabilities
    Confidence  : argmax of raw max-softmax confidence
    TTA         : a parameter-free router over TTA-averaged logits

The candidate set is frozen in `docs/routing-preregistration.md` before any
test-set evaluation, so choosing among these rules cannot become
selection-on-test.
"""

from __future__ import annotations

from collections import OrderedDict

from scripts.router.base import BaseRouter
from scripts.router.uniform import UniformRouter
from scripts.router.product import ProductRouter
from scripts.router.confidence import ConfidenceRouter
from scripts.router.tta import TTARouter

#: The complete, frozen set of parameter-free routers, in reporting order.
ROUTERS: 'OrderedDict[str, type[BaseRouter]]' = OrderedDict([
    ('Uniform', UniformRouter),
    ('Product', ProductRouter),
    ('Confidence', ConfidenceRouter),
    ('TTA', TTARouter),
])

__all__ = [
    'BaseRouter',
    'UniformRouter',
    'ProductRouter',
    'ConfidenceRouter',
    'TTARouter',
    'ROUTERS',
]
