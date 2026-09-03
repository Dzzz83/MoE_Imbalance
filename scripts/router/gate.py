"""
Learned gate routing — train an MLP to predict expert weights from features.

The gate network takes 89-d/92-d features as input and outputs a 3-d
weight vector (softmax-normalized) for combining expert predictions.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from sklearn.neural_network import MLPClassifier

from scripts.router.base import BaseRouter
from scripts.utils.features import (
    softmax, compute_89d_features, compute_92d_features,
)

EPS = 1e-12


class GateRouter(BaseRouter):
    """Route via learned MLP gate on 89-d/92-d features.

    The gate predicts a 3-d weight vector (via softmax) for combining
    expert predictions. Trained to minimize cross-entropy of the
    weighted ensemble.
    """

    def __init__(
        self,
        expert_names: list[str] | None = None,
        use_92d: bool = True,
        hidden_layer_sizes: tuple[int, ...] = (128, 64, 32),
        max_iter: int = 1000,
        alpha_tune_steps: int = 101,
    ):
        super().__init__(expert_names)
        self.use_92d = use_92d
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.alpha_tune_steps = alpha_tune_steps
        self.gate: MLPClassifier | None = None
        self.best_alpha: float = 1.0

    def train(
        self,
        val_logits: np.ndarray,
        val_labels: np.ndarray,
        val_features: dict | None = None,
    ) -> Self:
        """Train gate network on validation features.

        Two-stage training:
          1. Train MLP to predict per-expert correctness probabilities.
          2. Tune softmax temperature α on validation split.
        """
        N = val_logits.shape[0]

        if val_features is None:
            val_features = self._build_features(val_logits, val_labels)

        # Compute feature matrix
        if self.use_92d:
            pairwise_scores = val_features.get('pairwise_scores', None)
            F = compute_92d_features(
                val_features, self.expert_names, pairwise_scores=pairwise_scores
            )
        else:
            F = compute_89d_features(val_features, self.expert_names)

        # Target: per-expert correctness
        expert_preds = val_logits.argmax(axis=2)
        correct = (expert_preds == val_labels[:, None]).astype(float)

        # Train gate to predict correctness (multi-output)
        self.gate = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
        )
        self.gate.fit(F, correct)

        # Tune alpha on validation (using the gate's correctness predictions
        # as routing weights with softmax temperature)
        gate_probs = self.gate.predict_proba(F)
        if isinstance(gate_probs, list):
            # Multi-output MLP returns list of (N, 2) arrays
            gate_weights = np.column_stack([p[:, 1] for p in gate_probs])
        else:
            gate_weights = gate_probs

        probs = softmax(val_logits)
        self.best_alpha = self._tune_alpha(
            gate_weights, probs, val_labels,
            np.arange(N), np.arange(N),  # use all for tuning (no separate val)
        )

        self._is_trained = True
        return self

    def predict(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Return expert with highest gate-assigned weight."""
        weights = self.predict_proba(logits, features)
        return weights.argmax(axis=1)

    def predict_class(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Weighted combination via gate, then argmax."""
        weights = self.predict_proba(logits, features)
        probs = softmax(logits)
        combined = (weights[:, :, None] * probs).sum(axis=1)
        return combined.argmax(axis=1)

    def predict_proba(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Return gate-predicted expert weights."""
        if features is None:
            features = self._build_features(logits, np.zeros(logits.shape[0]))

        if self.use_92d:
            pairwise_scores = features.get('pairwise_scores', None)
            F = compute_92d_features(
                features, self.expert_names, pairwise_scores=pairwise_scores
            )
        else:
            F = compute_89d_features(features, self.expert_names)

        gate_probs = self.gate.predict_proba(F)
        if isinstance(gate_probs, list):
            gate_weights = np.column_stack([p[:, 1] for p in gate_probs])
        else:
            gate_weights = gate_probs

        # Apply tuned softmax temperature
        return softmax(self.best_alpha * gate_weights)

    def _tune_alpha(
        self,
        gate_weights: np.ndarray,
        probs: np.ndarray,
        labels: np.ndarray,
        tr_idx: np.ndarray,
        ev_idx: np.ndarray,
    ) -> float:
        """Tune softmax temperature α on validation split."""
        from scripts.utils.metrics import balanced_accuracy

        best_ba = -1.0
        best_alpha = 1.0

        for alpha in np.linspace(0, 10, self.alpha_tune_steps):
            w = softmax(alpha * gate_weights[tr_idx])
            combined = (w[:, :, None] * probs[tr_idx]).sum(axis=1)
            ba = balanced_accuracy(labels[tr_idx], combined.argmax(1))
            if ba > best_ba:
                best_ba = ba
                best_alpha = alpha

        return best_alpha

    @staticmethod
    def _build_features(logits: np.ndarray, labels: np.ndarray) -> dict:
        """Build a minimal features dict for compatibility."""
        N = logits.shape[0]
        num_experts = logits.shape[1]
        expert_names = [f"Expert_{i}" for i in range(num_experts)]

        features = {'targets': labels}
        for e in range(num_experts):
            logits_e = logits[:, e]
            probs_e = softmax(logits_e)
            features[expert_names[e]] = {
                'logits': logits_e,
                'probs': probs_e,
                'preds': logits_e.argmax(axis=1),
                'feats': np.random.randn(N, 64),
            }
        return features
