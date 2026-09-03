"""
Feature extraction utilities for expert routing.

Provides:
  - softmax: Stable NumPy softmax
  - extract_logits: Run a single model on a DataLoader
  - extract_all_experts: Run all models on a DataLoader
  - compute_24d_features: 24-d per-expert output-level features
  - compute_89d_features: 89-d feature vector (24d + 21d pairwise + 9d KL + 3d cal + 32d PCA)
  - compute_92d_features: 92-d = 89-d + 3-d pairwise scores
  - compute_energy: Energy score from logits
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.decomposition import PCA

from scripts.utils.data import load_all_experts

EPS = 1e-12


# ── Softmax ──────────────────────────────────────────────────────────────


def softmax(logits: np.ndarray) -> np.ndarray:
    """Stable softmax over the last axis.

    Args:
        logits: Any shape; softmax computed over the last dimension.
    Returns:
        Probabilities with same shape, summing to 1 over the last axis.
    """
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=-1, keepdims=True)


# ── Logit Extraction ─────────────────────────────────────────────────────


def extract_logits(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str = "cpu",
    expert_name: str = "",
) -> np.ndarray:
    """Run a model on a DataLoader and return all logits.

    Handles PaCo (encoder_q + linear_q) vs standard (backbone + fc) models.
    """
    all_logits: list[np.ndarray] = []
    with torch.no_grad():
        for imgs, _tgts in loader:
            imgs = imgs.to(device, non_blocking=True)
            if expert_name.upper() == "PACO":
                feats = model.encoder_q[0](imgs)
                feats = feats.view(feats.size(0), -1)
                logits = model.linear_q(feats)
            else:
                feats = model.backbone(imgs)
                logits = model.fc(feats)
            all_logits.append(logits.cpu().numpy())
    return np.concatenate(all_logits)


def extract_all_experts(
    models: dict[str, torch.nn.Module],
    loader: torch.utils.data.DataLoader,
    device: str = "cpu",
    return_features: bool = True,
) -> dict[str, dict]:
    """Run all expert models on a DataLoader and collect outputs.

    Args:
        models: Dict mapping expert name → loaded model.
        loader: DataLoader yielding (images, targets).
        device: Computation device.
        return_features: If True, also extract backbone features (64-d).
    Returns:
        Dict like::

            {
              'LAL': {
                  'logits': np.ndarray (N, 100),
                  'probs':  np.ndarray (N, 100),
                  'preds':  np.ndarray (N,),
                  'feats':  np.ndarray (N, 64)   # if return_features
              },
              ...
            }
    """
    results: dict[str, dict] = {name: {} for name in models}
    all_logits: dict[str, list[np.ndarray]] = {name: [] for name in models}
    all_feats: dict[str, list[np.ndarray]] = {name: [] for name in models} if return_features else {}
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for imgs, tgts in loader:
            imgs = imgs.to(device, non_blocking=True)
            all_targets.append(tgts.numpy())

            for name, model in models.items():
                if name.upper() == "PACO":
                    feats = model.encoder_q[0](imgs)
                    feats = feats.view(feats.size(0), -1)
                    logits = model.linear_q(feats)
                else:
                    feats = model.backbone(imgs)
                    logits = model.fc(feats)

                all_logits[name].append(logits.cpu().numpy())
                if return_features:
                    all_feats[name].append(feats.cpu().numpy())

    for name in models:
        logits = np.concatenate(all_logits[name])
        results[name]["logits"] = logits
        results[name]["probs"] = softmax(logits)
        results[name]["preds"] = logits.argmax(axis=1)
        if return_features:
            results[name]["feats"] = np.concatenate(all_feats[name])

    results["targets"] = np.concatenate(all_targets)
    return results


# ── 24-d Features ────────────────────────────────────────────────────────


def compute_24d_features(probs_list: list[np.ndarray]) -> np.ndarray:
    """Compute 24-d output-level features from expert probabilities.

    For each expert (3 experts × 7 features = 21-d) + 3 global = 24-d.

    Per-expert features (7-d):
      0. entropy
      1. max confidence
      2. margin (top1 - top2)
      3. top-2 mass
      4. tail residual (1 - top2_mass)
      5. cosine similarity to mean prediction
      6. KL divergence from mean prediction

    Global features (3-d):
      7. mean entropy across experts
      8. class-wise prediction variance (mean over classes)
      9. variance of confidences across experts
    """
    N = probs_list[0].shape[0]
    p_mean = np.mean(probs_list, axis=0)

    per_expert_features = []
    for p_e in probs_list:
        entropy = -np.sum(p_e * np.log(p_e + EPS), axis=1)
        max_conf = p_e.max(axis=1)
        sorted_p = np.sort(p_e, axis=1)[:, ::-1]
        margin = sorted_p[:, 0] - sorted_p[:, 1]
        topk_mass = sorted_p[:, :2].sum(axis=1)
        tail_residual = 1.0 - topk_mass
        cos_sim = (
            (p_e / (np.linalg.norm(p_e, axis=1, keepdims=True) + EPS))
            * (p_mean / (np.linalg.norm(p_mean, axis=1, keepdims=True) + EPS))
        ).sum(axis=1)
        kl = np.sum(p_e * np.log((p_e + EPS) / (p_mean + EPS)), axis=1)
        per_expert_features.append(
            np.stack([entropy, max_conf, margin, topk_mass, tail_residual, cos_sim, kl], axis=1)
        )

    # Global features
    stacked = np.stack(probs_list, axis=0)
    class_var = np.var(stacked, axis=0).mean(axis=1)
    confs = np.stack([p.max(axis=1) for p in probs_list], axis=0)
    global_feats = np.stack(
        [
            np.mean([f[:, 0] for f in per_expert_features], axis=0),
            class_var,
            np.var(confs, axis=0),
        ],
        axis=1,
    )

    return np.concatenate([*per_expert_features, global_feats], axis=1).astype(np.float32)


# ── Energy Score ─────────────────────────────────────────────────────────


def compute_energy(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Energy score: ``-temperature * log(sum(exp(logits / temperature)))``.

    Lower energy → higher confidence. Useful for OOD detection and
    cross-expert comparison.
    """
    return -temperature * np.log(np.sum(np.exp(logits / temperature), axis=1))


# ── 89-d and 92-d Features ──────────────────────────────────────────────


def compute_89d_features(
    extracted_dict: dict,
    expert_names: list[str] | None = None,
    n_pca_components: int = 32,
) -> np.ndarray:
    """Compute 89-d feature vector for routing.

    Components:
      - 24-d standard features (from compute_24d_features)
      - 21-d pairwise comparison features (entropy diff, conf diff, margin diff,
        energy diff, top2 diff, KL divergences)
      - 9-d KL consensus features (KL, L2, cosine to mean per expert)
      - 3-d calibration features (confidence per expert)
      - 32-d PCA of concatenated backbone features (192-d → 32-d)

    Returns:
        Array shape (N, 89).
    """
    if expert_names is None:
        expert_names = ["LAL", "Mixup", "PaCo"]

    probs_dict = {n: extracted_dict[n]["probs"] for n in expert_names}
    N = probs_dict[expert_names[0]].shape[0]
    p_mean = np.mean([probs_dict[n] for n in expert_names], axis=0)

    # ── 24-d standard ──
    probs_list = [probs_dict[n] for n in expert_names]
    F24 = compute_24d_features(probs_list)

    # ── 21-d pairwise ──
    fpw = []
    for i, n1 in enumerate(expert_names):
        for j, n2 in enumerate(expert_names):
            if j <= i:
                continue
            p1, p2 = probs_dict[n1], probs_dict[n2]
            e1 = -np.sum(p1 * np.log(p1 + EPS), axis=1)
            e2 = -np.sum(p2 * np.log(p2 + EPS), axis=1)
            fpw.append(e1 - e2)                                   # entropy diff
            fpw.append(p1.max(1) - p2.max(1))                     # confidence diff
            s1 = np.sort(p1, axis=1)[:, ::-1]
            s2 = np.sort(p2, axis=1)[:, ::-1]
            fpw.append((s1[:, 0] - s1[:, 1]) - (s2[:, 0] - s2[:, 1]))  # margin diff
            fpw.append(
                compute_energy(extracted_dict[n1]["logits"])
                - compute_energy(extracted_dict[n2]["logits"])
            )                                                      # energy diff
            fpw.append(s1[:, :2].sum(1) - s2[:, :2].sum(1))       # top2 mass diff
            fpw.append(np.sum(p1 * np.log((p1 + EPS) / (p2 + EPS)), axis=1))  # KL(p1||p2)
            fpw.append(np.sum(p2 * np.log((p2 + EPS) / (p1 + EPS)), axis=1))  # KL(p2||p1)
    Fpw = np.stack(fpw, axis=1).astype(np.float32)

    # ── 9-d KL consensus ──
    fkl = []
    for n in expert_names:
        p = probs_dict[n]
        kl = np.sum(p * np.log((p + EPS) / (p_mean + EPS)), axis=1)
        l2 = np.sqrt(np.sum((p - p_mean) ** 2, axis=1))
        cs_kl = (
            (p / (np.linalg.norm(p, axis=1, keepdims=True) + EPS))
            * (p_mean / (np.linalg.norm(p_mean, axis=1, keepdims=True) + EPS))
        ).sum(1)
        fkl.extend([kl, l2, cs_kl])
    Fkl = np.stack(fkl, axis=1).astype(np.float32)

    # ── 3-d calibration (confidence per expert) ──
    Fcal = np.stack(
        [
            extracted_dict[n]["probs"][np.arange(N), extracted_dict[n]["preds"]]
            for n in expert_names
        ],
        axis=1,
    ).astype(np.float32)

    # ── 32-d PCA of backbone features ──
    F192 = np.concatenate(
        [extracted_dict[n]["feats"] for n in expert_names], axis=1
    )
    pca = PCA(n_components=n_pca_components, random_state=42)
    Fpca = pca.fit_transform(F192).astype(np.float32)

    return np.concatenate([F24, Fpw, Fkl, Fcal, Fpca], axis=1)


def compute_92d_features(
    extracted_dict: dict,
    expert_names: list[str] | None = None,
    pairwise_scores: np.ndarray | None = None,
    n_pca_components: int = 32,
) -> np.ndarray:
    """Compute 92-d features = 89-d + optionally 3-d pairwise scores.

    The 3 additional dimensions are pairwise routing scores
    (e.g., from a pairwise tournament or learned preferences).

    Args:
        extracted_dict: Output from extract_all_experts.
        expert_names: List of expert names.
        pairwise_scores: Optional (N, 3) array of pairwise scores.
        n_pca_components: PCA components for backbone features.
    Returns:
        Array shape (N, 89) if no pairwise_scores, else (N, 92).
    """
    F_89d = compute_89d_features(extracted_dict, expert_names, n_pca_components)
    if pairwise_scores is not None:
        return np.concatenate([F_89d, pairwise_scores], axis=1)
    return F_89d
