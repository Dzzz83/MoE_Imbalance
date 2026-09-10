"""
KL divergence computation between two experts' probability distributions.

Provides both symmetric and asymmetric KL divergence, plus JS divergence
as a symmetric alternative.  All functions operate on logits or probabilities
and are numerically stable.

Usage:
    kl_val = kl_divergence(logits_a, logits_b)         # KL(P_A || P_B)
    js_val = js_divergence(logits_a, logits_b)          # Symmetric JS
    kl_agree_label = compute_agreement_label(logits_a, logits_b, threshold=0.5)
"""

import torch
import torch.nn.functional as F


def kl_divergence(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    log_input: bool = False,
    reduction: str = "batch_mean",
) -> torch.Tensor:
    """
    Compute KL(P_A || P_B) where P = softmax(logits).

    Args:
        logits_a: (B, C) tensor — logits of expert A.
        logits_b: (B, C) tensor — logits of expert B.
        log_input: If True, treat inputs as log-probabilities instead of logits.
        reduction: 'batch_mean' returns scalar mean over batch.
                   'none' returns per-sample KL (B,).

    Returns:
        Tensor of shape () if batch_mean, or (B,) if none.
    """
    if log_input:
        log_p_a = logits_a
        log_p_b = logits_b
    else:
        log_p_a = F.log_softmax(logits_a, dim=1)
        log_p_b = F.log_softmax(logits_b, dim=1)

    # KL(P_A || P_B) = sum(P_A * (log P_A - log P_B))
    p_a = torch.exp(log_p_a)
    kl_per_sample = (p_a * (log_p_a - log_p_b)).sum(dim=1)  # (B,)

    if reduction == "batch_mean":
        return kl_per_sample.mean()
    elif reduction == "sum":
        return kl_per_sample.sum()
    else:
        return kl_per_sample  # (B,)


def symmetric_kl_divergence(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    reduction: str = "batch_mean",
) -> torch.Tensor:
    """
    Symmetric KL: KL(P_A || P_B) + KL(P_B || P_A).
    """
    kl_ab = kl_divergence(logits_a, logits_b, reduction="none")
    kl_ba = kl_divergence(logits_b, logits_a, reduction="none")
    sym_kl = kl_ab + kl_ba  # (B,)

    if reduction == "batch_mean":
        return sym_kl.mean()
    elif reduction == "sum":
        return sym_kl.sum()
    else:
        return sym_kl


def js_divergence(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    reduction: str = "batch_mean",
) -> torch.Tensor:
    """
    Jensen-Shannon Divergence (symmetric, bounded [0, log(2)]).

    JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = (P + Q) / 2.
    """
    log_p_a = F.log_softmax(logits_a, dim=1)
    log_p_b = F.log_softmax(logits_b, dim=1)
    p_a = torch.exp(log_p_a)
    p_b = torch.exp(log_p_b)

    m = 0.5 * (p_a + p_b)
    log_m = torch.log(m + 1e-12)

    # KL(P_A || M)
    kl_am = (p_a * (log_p_a - log_m)).sum(dim=1)
    # KL(P_B || M)
    kl_bm = (p_b * (log_p_b - log_m)).sum(dim=1)

    js = 0.5 * (kl_am + kl_bm)

    if reduction == "batch_mean":
        return js.mean()
    elif reduction == "sum":
        return js.sum()
    else:
        return js


def compute_agreement_label(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    threshold: float = 0.5,
    metric: str = "kl",
) -> torch.Tensor:
    """
    Binarize the divergence between two experts into agree/disagree labels.

    Args:
        logits_a: (B, C) tensor — logits of expert A.
        logits_b: (B, C) tensor — logits of expert B.
        threshold: Divergence threshold. Samples with divergence > threshold
                   are labeled 1 (disagree), others 0 (agree).
        metric: 'kl' for asymmetric KL, 'sym_kl' for symmetric KL, 'js' for JS.

    Returns:
        labels: (B,) long tensor — 0 = agree, 1 = disagree.
    """
    if metric == "kl":
        div = kl_divergence(logits_a, logits_b, reduction="none")
    elif metric == "sym_kl":
        div = symmetric_kl_divergence(logits_a, logits_b, reduction="none")
    elif metric == "js":
        div = js_divergence(logits_a, logits_b, reduction="none")
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return (div > threshold).long()
