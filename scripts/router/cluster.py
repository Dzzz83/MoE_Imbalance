"""
Cluster-based routing — cluster the feature space and learn per-cluster
optimal expert weights.

Uses KMeans on 192-d concatenated backbone features. For each cluster,
finds the optimal fixed weight combination (via grid search) on training
samples. At test time, assigns each sample to its nearest cluster and
uses that cluster's weights.
"""

from __future__ import annotations

from itertools import product
from typing import Self

import numpy as np
from sklearn.cluster import KMeans

from scripts.router.base import BaseRouter
from scripts.utils.features import softmax
from scripts.utils.metrics import balanced_accuracy

EPS = 1e-12


class ClusterRouter(BaseRouter):
    """Route via feature clustering + per-cluster optimal weights.

    Three variants:
      - hard: each sample assigned to one cluster, uses that cluster's weights.
      - soft: distance-weighted membership across all clusters.
      - agreement: group by correctness agreement pattern instead of features.
    """

    def __init__(
        self,
        expert_names: list[str] | None = None,
        n_clusters: int = 8,
        variant: str = "hard",
        weight_grid_step: float = 0.05,
    ):
        super().__init__(expert_names)
        self.n_clusters = n_clusters
        self.variant = variant
        self.weight_grid_step = weight_grid_step
        self.kmeans: KMeans | None = None
        self.cluster_weights: np.ndarray | None = None  # (n_clusters, num_experts)

    def train(
        self,
        val_logits: np.ndarray,
        val_labels: np.ndarray,
        val_features: dict | None = None,
    ) -> Self:
        """Cluster training features and find per-cluster optimal weights."""
        N = val_logits.shape[0]

        # Build feature matrix
        if val_features is not None and 'feats' in val_features.get(self.expert_names[0], {}):
            feat_mat = np.concatenate(
                [val_features[n]['feats'] for n in self.expert_names], axis=1
            )
        else:
            feat_mat = val_logits.reshape(N, -1)

        if self.variant == "agreement":
            # Group by correctness agreement pattern
            expert_preds = val_logits.argmax(axis=2)
            correct = (expert_preds == val_labels[:, None]).astype(float)
            # Each sample gets a 3-bit pattern: e.g., [1, 0, 1]
            # Convert to integer labels 0..7
            cluster_labels = (correct * np.array([4, 2, 1])).sum(axis=1).astype(np.int64)
            n_clusters_actual = len(np.unique(cluster_labels))
            self.cluster_weights = np.zeros((n_clusters_actual, self.num_experts))
            probs = softmax(val_logits)

            for ci in range(n_clusters_actual):
                mask = cluster_labels == ci
                if mask.sum() >= 5:
                    self.cluster_weights[ci] = self._find_optimal_weights(
                        probs[mask], val_labels[mask]
                    )
                else:
                    self.cluster_weights[ci] = np.ones(self.num_experts) / self.num_experts

            # Store cluster mapping for prediction
            self._cluster_map = {int(k): v for k, v in enumerate(np.unique(cluster_labels))}
            self._reverse_map = {v: k for k, v in self._cluster_map.items()}
            self._cluster_labels = cluster_labels
        else:
            # KMeans clustering on features
            self.kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=42,
                n_init=10,
            )
            self.kmeans.fit(feat_mat)

            # Find optimal weights per cluster
            probs = softmax(val_logits)
            cluster_labels = self.kmeans.predict(feat_mat)
            self.cluster_weights = np.zeros((self.n_clusters, self.num_experts))

            for ci in range(self.n_clusters):
                mask = cluster_labels == ci
                if mask.sum() >= 5:
                    self.cluster_weights[ci] = self._find_optimal_weights(
                        probs[mask], val_labels[mask]
                    )
                else:
                    self.cluster_weights[ci] = np.ones(self.num_experts) / self.num_experts

        self._is_trained = True
        return self

    def predict(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Return expert with highest cluster-assigned weight."""
        weights = self.predict_proba(logits, features)
        return weights.argmax(axis=1)

    def predict_class(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Weighted combination via cluster weights, then argmax."""
        weights = self.predict_proba(logits, features)
        probs = softmax(logits)
        combined = (weights[:, :, None] * probs).sum(axis=1)
        return combined.argmax(axis=1)

    def predict_proba(
        self,
        logits: np.ndarray,
        features: dict | None = None,
    ) -> np.ndarray:
        """Return routing weights per sample based on cluster assignment."""
        N = logits.shape[0]

        if self.variant == "agreement":
            # Recompute agreement pattern
            expert_preds = logits.argmax(axis=2)
            # We don't have labels at test time, so fall back to uniform
            return np.ones((N, self.num_experts)) / self.num_experts

        # Build feature matrix, matching the dimension used during training
        n_feats_train = self.kmeans.n_features_in_ if self.kmeans is not None else 192
        if features is not None and 'feats' in features.get(self.expert_names[0], {}):
            feat_mat = np.concatenate(
                [features[n]['feats'] for n in self.expert_names], axis=1
            )
        else:
            # Use logits reshaped to match training dimension
            feat_mat = logits.reshape(N, -1)
            # If dimension doesn't match, take first n_feats_train
            if feat_mat.shape[1] != n_feats_train:
                feat_mat = feat_mat[:, :n_feats_train]

        if self.variant == "soft":
            # Soft clustering: distance-weighted membership
            from scipy.spatial.distance import cdist
            dists = cdist(feat_mat, self.kmeans.cluster_centers_)
            temp = dists.mean()
            soft_weights = np.exp(-dists / max(temp, EPS))
            soft_weights /= soft_weights.sum(axis=1, keepdims=True)
            # Weighted combination of per-cluster weights
            weights = soft_weights @ self.cluster_weights  # (N, num_experts)
            return weights
        else:
            # Hard clustering
            cluster_labels = self.kmeans.predict(feat_mat)
            weights = self.cluster_weights[cluster_labels]  # (N, num_experts)
            return weights

    def _find_optimal_weights(
        self,
        probs: np.ndarray,     # (N, num_experts, 100)
        labels: np.ndarray,    # (N,)
    ) -> np.ndarray:
        """Grid search for optimal fixed weights on a subset."""
        best_ba = -1.0
        best_weights = np.ones(self.num_experts) / self.num_experts

        step = self.weight_grid_step
        n_steps = int(1.0 / step) + 1

        # 3-expert grid search (constrained to sum=1)
        for w0 in np.linspace(0, 1, n_steps):
            for w1 in np.linspace(0, 1 - w0, n_steps):
                w2 = 1.0 - w0 - w1
                if w2 < 0:
                    continue
                weights = np.array([w0, w1, w2])
                combined = (weights[None, :, None] * probs).sum(axis=1)
                preds = combined.argmax(axis=1)
                ba = balanced_accuracy(labels, preds)
                if ba > best_ba:
                    best_ba = ba
                    best_weights = weights

        return best_weights
