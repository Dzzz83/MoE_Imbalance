"""
Pairwise routing — train pairwise comparators and rank experts via tournament.

Trains one binary classifier per expert pair to predict which expert will
be more reliable. At test time, uses Copeland's tournament method to
rank experts: the expert with the most pairwise wins is selected.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score

from scripts.router.base import BaseRouter
from scripts.utils.features import (
    softmax, compute_89d_features,
)

EPS = 1e-12

# Expert pair indices: (0,1) = LAL vs PaCo, (0,2) = LAL vs Mixup, (1,2) = PaCo vs Mixup
PAIRS = [(0, 1), (0, 2), (1, 2)]
PAIR_NAMES = ['LAL_vs_PaCo', 'LAL_vs_Mixup', 'PaCo_vs_Mixup']


class PairwiseRouter(BaseRouter):
    """Route via pairwise tournament with learned comparators.

    For each expert pair (i, j), trains a classifier on 89-d features
    to predict P(expert_i is correct XOR expert_j is correct).
    At test time, uses Copeland's method: each expert gets one "win"
    per pairwise score > 0.5; the expert with the most wins is selected.
    """

    def __init__(
        self,
        expert_names: list[str] | None = None,
        hidden_layer_sizes: tuple[int, ...] = (64, 32),
        max_iter: int = 500,
    ):
        super().__init__(expert_names)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.comparators: list[MLPClassifier] = []
        self.pair_aurocs: list[float] = []

    def train(
        self,
        val_logits: np.ndarray,
        val_labels: np.ndarray,
        val_features: dict | None = None,
    ) -> Self:
        """Train one pairwise comparator per expert pair.

        For each pair (i, j), trains on samples where exactly one of
        the two experts is correct (the "clean" XOR samples). The
        classifier predicts P(expert_i is correct | expert_i XOR expert_j).
        """
        # Build features if not provided
        if val_features is None:
            val_features = self._build_features(val_logits, val_labels)

        F = compute_89d_features(val_features, self.expert_names)
        N = F.shape[0]

        # Per-expert correctness
        expert_preds = val_logits.argmax(axis=2)
        correct = (expert_preds == val_labels[:, None]).astype(float)

        self.comparators = []
        self.pair_aurocs = []

        for p_idx, (i, j) in enumerate(PAIRS):
            # XOR mask: samples where exactly one expert is correct
            xor_mask = (correct[:, i].astype(int) != correct[:, j].astype(int))

            if xor_mask.sum() < 10:
                # Too few clean samples — fall back to 0.5
                self.comparators.append(None)
                self.pair_aurocs.append(0.0)
                continue

            # Labels: 1 if expert_i is correct and expert_j is wrong
            y_pair = correct[:, i].astype(int)

            clf = MLPClassifier(
                hidden_layer_sizes=self.hidden_layer_sizes,
                max_iter=self.max_iter,
                random_state=42 + p_idx,
                early_stopping=True,
                validation_fraction=0.2,
            )
            clf.fit(F[xor_mask], y_pair[xor_mask])

            # AUROC on clean samples
            if xor_mask.sum() > 1 and len(np.unique(y_pair[xor_mask])) > 1:
                try:
                    au = roc_auc_score(
                        y_pair[xor_mask],
                        clf.predict_proba(F[xor_mask])[:, 1]
                    )
                except Exception:
                    au = 0.0
            else:
                au = 0.0

            self.comparators.append(clf)
            self.pair_aurocs.append(au)

        self._is_trained = True
        return self

    def predict(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Run tournament ranking and return winning expert per sample."""
        pairwise_scores = self._get_pairwise_scores(logits, features)
        best_expert, _ = self._tournament_ranking(pairwise_scores)
        return best_expert

    def predict_proba(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Return soft routing weights based on aggregated pairwise scores."""
        pairwise_scores = self._get_pairwise_scores(logits, features)
        _, wins = self._tournament_ranking(pairwise_scores)

        # Convert win counts to soft weights via softmax
        weights = softmax(wins * 3.0)  # temperature for sharpness
        # Handle ties: if all wins are 0, use uniform
        uniform_mask = (wins.sum(axis=1) == 0)
        if uniform_mask.any():
            weights[uniform_mask] = 1.0 / self.num_experts
        return weights

    def _get_pairwise_scores(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Compute pairwise preference scores for all pairs.

        Returns:
            Array shape (N, 3) with P(i > j) for each pair.
        """
        if features is None:
            features = self._build_features(logits, np.zeros(logits.shape[0]))

        F = compute_89d_features(features, self.expert_names)
        pairwise_scores = np.zeros((F.shape[0], 3))

        for p_idx, (i, j) in enumerate(PAIRS):
            clf = self.comparators[p_idx]
            if clf is not None:
                pairwise_scores[:, p_idx] = clf.predict_proba(F)[:, 1]
            else:
                pairwise_scores[:, p_idx] = 0.5

        return pairwise_scores

    @staticmethod
    def _tournament_ranking(
        pairwise_scores: np.ndarray,
        threshold: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert 3 pairwise scores to expert ranking via Copeland's method.

        Args:
            pairwise_scores: (N, 3) array of P(i > j) for each pair.
            threshold: Win threshold (default 0.5).
        Returns:
            best_expert: (N,) array of winning expert indices.
            wins: (N, 3) array of win counts per expert.
        """
        n_samp = pairwise_scores.shape[0]
        wins = np.zeros((n_samp, 3))

        # Pair 0: (0, 1) — LAL vs PaCo
        wins[:, 0] += (pairwise_scores[:, 0] > threshold).astype(float)
        wins[:, 1] += (pairwise_scores[:, 0] < (1 - threshold)).astype(float)

        # Pair 1: (0, 2) — LAL vs Mixup
        wins[:, 0] += (pairwise_scores[:, 1] > threshold).astype(float)
        wins[:, 2] += (pairwise_scores[:, 1] < (1 - threshold)).astype(float)

        # Pair 2: (1, 2) — PaCo vs Mixup
        wins[:, 1] += (pairwise_scores[:, 2] > threshold).astype(float)
        wins[:, 2] += (pairwise_scores[:, 2] < (1 - threshold)).astype(float)

        best_expert = wins.argmax(axis=1)
        return best_expert, wins

    @staticmethod
    def _build_features(
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> dict:
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
