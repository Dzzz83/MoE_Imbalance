"""
Correctness-prediction routing — train trust meters to predict which
expert will be correct for each sample.

This is the core routing approach behind the 89-d and 92-d methods.
Trust meters are MLPs trained on feature vectors to predict per-expert
correctness probabilities.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from scripts.router.base import BaseRouter
from scripts.utils.features import (
    softmax, compute_89d_features, compute_92d_features,
)

EPS = 1e-12


class CorrectnessRouter(BaseRouter):
    """Route by predicting which expert will be correct.

    Trains one trust meter (MLP classifier) per expert on features
    extracted from the validation set. At test time, each trust meter
    predicts P(expert_e is correct | features), and the sample is
    routed to the expert with the highest predicted correctness.

    Supports 89-d and 92-d feature variants.
    """

    def __init__(
        self,
        expert_names: list[str] | None = None,
        use_92d: bool = True,
        hidden_layer_sizes: tuple[int, ...] = (128, 64),
        max_iter: int = 500,
    ):
        super().__init__(expert_names)
        self.use_92d = use_92d
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.trust_meters: list[MLPClassifier | LogisticRegression] = []
        self.feature_dim: int | None = None

    def train(
        self,
        val_logits: np.ndarray,
        val_labels: np.ndarray,
        val_features: dict | None = None,
    ) -> Self:
        """Train one trust meter per expert on validation features.

        Args:
            val_logits: (N_val, num_experts, 100).
            val_labels: (N_val,).
            val_features: Must contain extracted dict with 'probs', 'preds',
                          'feats', 'logits' for each expert, plus 'targets'.
                          If None, will be constructed from val_logits.
        """
        # Build features if not provided
        if val_features is None:
            val_features = self._build_features(val_logits, val_labels)

        # Compute correctness for each expert
        expert_preds = val_logits.argmax(axis=2)  # (N, num_experts)
        correct = (expert_preds == val_labels[:, None]).astype(float)

        # Compute feature matrix
        F = self._compute_feature_matrix(val_features)

        # Train one trust meter per expert
        self.trust_meters = []
        for e in range(self.num_experts):
            clf = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                max_iter=self.max_iter,
                random_state=42 + e,
                early_stopping=True,
                validation_fraction=0.2,
            )
            clf.fit(F, correct[:, e])
            self.trust_meters.append(clf)

        self._is_trained = True
        return self

    def predict(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Route to expert with highest predicted correctness."""
        if features is None:
            features = self._build_features(logits, np.zeros(logits.shape[0]))

        F = self._compute_feature_matrix(features)
        correctness_probs = np.column_stack([
            clf.predict_proba(F)[:, 1] for clf in self.trust_meters
        ])
        return correctness_probs.argmax(axis=1)

    def predict_proba(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Return predicted correctness probabilities as routing weights."""
        if features is None:
            features = self._build_features(logits, np.zeros(logits.shape[0]))

        F = self._compute_feature_matrix(features)
        correctness_probs = np.column_stack([
            clf.predict_proba(F)[:, 1] for clf in self.trust_meters
        ])
        # Softmax over correctness probs for valid routing weights
        return softmax(correctness_probs * 5.0)

    def _compute_feature_matrix(self, features: dict) -> np.ndarray:
        """Compute 89-d or 92-d feature matrix from extracted dict."""
        if self.use_92d:
            # Try to get pairwise_scores from features dict
            pairwise_scores = features.get('pairwise_scores', None)
            return compute_92d_features(
                features, self.expert_names, pairwise_scores=pairwise_scores
            )
        else:
            return compute_89d_features(features, self.expert_names)

    @staticmethod
    def _build_features(
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> dict:
        """Build a minimal features dict from logits for compatibility."""
        N = logits.shape[0]
        num_experts = logits.shape[1]
        num_classes = logits.shape[2]

        # Generate plausible expert names if not set
        expert_names = [f"Expert_{i}" for i in range(num_experts)]

        features = {'targets': labels}
        for e in range(num_experts):
            logits_e = logits[:, e]
            probs_e = softmax(logits_e)
            features[expert_names[e]] = {
                'logits': logits_e,
                'probs': probs_e,
                'preds': logits_e.argmax(axis=1),
                'feats': np.random.randn(N, 64),  # placeholder — real features needed
            }
        return features
