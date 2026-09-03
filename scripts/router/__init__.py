"""
Router framework for expert routing on CIFAR-100-LT.

Every router inherits from ``BaseRouter`` and implements:
  - train(val_logits, val_labels, val_features) → Self
  - predict(logits, features) → np.ndarray (expert index per sample)

Available routers:
  - UniformRouter     : average logits across experts
  - ConfidenceRouter  : pick highest max-softmax confidence
  - ProductRouter     : multiply softmax probabilities
  - CorrectnessRouter : train trust meters on 89-d/92-d features
  - PairwiseRouter    : pairwise tournament with learned comparators
  - ClusterRouter     : cluster features + per-cluster optimal weights
  - GateRouter        : learned MLP gate for expert weighting
  - TTARouter         : test-time augmentation + routing on TTA features
  - SelectiveRouter   : abstain on low-confidence samples
"""

from scripts.router.base import BaseRouter

# Register all routers
from scripts.router.uniform import UniformRouter
from scripts.router.confidence import ConfidenceRouter
from scripts.router.product import ProductRouter
from scripts.router.correctness import CorrectnessRouter
from scripts.router.pairwise import PairwiseRouter
from scripts.router.cluster import ClusterRouter
from scripts.router.gate import GateRouter
from scripts.router.tta import TTARouter
from scripts.router.selective import SelectiveRouter

__all__ = [
    'BaseRouter',
    'UniformRouter',
    'ConfidenceRouter',
    'ProductRouter',
    'CorrectnessRouter',
    'PairwiseRouter',
    'ClusterRouter',
    'GateRouter',
    'TTARouter',
    'SelectiveRouter',
]
