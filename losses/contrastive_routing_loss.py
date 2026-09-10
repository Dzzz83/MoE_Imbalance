"""
Supervised contrastive loss for routing embeddings (DACE).

The loss operates on binary agreement labels (0 = agree, 1 = disagree) instead
of class labels.  Embeddings with the SAME agreement label are pulled together;
embeddings with DIFFERENT agreement labels are pushed apart.

Loss formulation (standard supervised NT-Xent / InfoNCE):

    L_i = -log( Σ_{j: label_j = label_i} exp(sim(e_i, e_j)/τ_c)
              / Σ_{k ≠ i} exp(sim(e_i, e_k)/τ_c) )

Where sim(e_i, e_j) = cosine_similarity(e_i, e_j) and τ_c is the temperature.

Reference:  Chen et al., "A Simple Framework for Contrastive Learning
of Visual Representations" (SimCLR, ICML 2020).
            Khosla et al., "Supervised Contrastive Learning" (NeurIPS 2020).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveRoutingLoss(nn.Module):
    """
    Supervised contrastive loss for routing embeddings with binary labels.

    Args:
        temperature: Contrastive temperature τ_c (default 0.5).
        base_temperature: Reference temperature for scaling (default 0.5).
    """

    def __init__(self, temperature: float = 0.5, base_temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(
        self,
        embeddings: torch.Tensor,   # (B, D) routing embeddings
        labels: torch.Tensor,       # (B,) binary agreement labels: 0=agree, 1=disagree
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            embeddings: L2-normalized (or unnormalized) routing embeddings.
            labels:     Binary labels (0 = agree, 1 = disagree).

        Returns:
            loss: scalar contrastive loss.
            aux: dict with 'contrastive_loss' scalar for logging.
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=device), {"contrastive_loss": 0.0}

        # L2-normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)  # (B, D)

        # Compute cosine similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T)  # (B, B)

        # Create mask: 1 if labels[i] == labels[j] (same agreement class)
        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float().to(device)  # (B, B)

        # Remove self-contrast (diagonal)
        logits_mask = torch.ones_like(pos_mask) - torch.eye(batch_size, device=device)
        pos_mask = pos_mask * logits_mask  # zero out diagonal

        # Temperature-scaled similarities
        logits = sim_matrix / self.temperature  # (B, B)

        # Numerical stability: subtract max per row
        logits_max, _ = torch.max(logits * logits_mask, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Compute log-probabilities
        exp_logits = torch.exp(logits) * logits_mask  # zero out self
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # Positive mask: only same-label pairs (excluding self)
        pos_count = pos_mask.sum(1)
        # Avoid division by zero for samples with no positive pair
        pos_count = pos_count.clamp(min=1)

        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_count

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss, {"contrastive_loss": loss.detach()}
