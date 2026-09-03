"""
PaCo Loss — Cui et al., ICCV 2021.

Matches the official implementation:
    https://github.com/dvlab-research/Parametric-Contrastive-Learning

The loss concatenates supervised logits (adjusted by class frequencies) with
contrastive similarities into one tensor, then computes InfoNCE over the
combined set.  Positives are:
  - (supervised part) the query's own class centre (classifier weight),
    weighted by β
  - (contrastive part) other samples in the batch from the same class,
    weighted by α

Negatives include all other class centres, all other-class samples, and
all queued keys (MoCo memory bank).

Reference: https://arxiv.org/abs/2107.12028
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PaCoLoss(nn.Module):
    """
    Args:
        alpha:       Weight for same-class sample-sample positives (default 0.5).
        beta:        Weight for class-centre supervised positive (default 1.0).
        gamma:       Weight for negative-mask logits (default 1.0).
        supt:        Temperature for supervised logits (default 1.0).
        temperature: Temperature for contrastive similarities (default 0.07).
        K:           Queue size (default 4096; K=65536 in the paper).
        num_classes: Number of classes.
    """

    def __init__(
        self,
        alpha: float = 0.01,          # Official PaCo CIFAR-100-LT IR=0.01
        beta: float = 1.0,
        gamma: float = 1.0,
        supt: float = 1.0,
        temperature: float = 0.05,    # Official: --moco-t 0.05
        K: int = 1024,                # Official: --moco-k 1024
        num_classes: int = 100,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.supt = supt
        self.temperature = temperature
        self.K = K
        self.num_classes = num_classes
        self.base_temperature = temperature

        # class-frequency weight for logit adjustment (set via set_class_weight)
        self.register_buffer('weight', torch.ones(1, num_classes) / num_classes)

    def set_class_weight(self, cls_num_list: list | torch.Tensor):
        """
        Set class-frequency weights for the logit-adjustment term.

        Args:
            cls_num_list: list or tensor of per-class sample counts.
        """
        if not isinstance(cls_num_list, torch.Tensor):
            cls_num_list = torch.tensor(cls_num_list, dtype=torch.float)
        if cls_num_list.dim() == 1:
            cls_num_list = cls_num_list.unsqueeze(0)  # (1, C)
        weight = cls_num_list / cls_num_list.sum()
        self.weight.data = weight.to(self.weight.device)

    def forward(
        self,
        features: torch.Tensor,     # (2*B + K, dim)  query, key, queue embeds
        labels: torch.Tensor,        # (2*B + K,)
        sup_logits: torch.Tensor,    # (B, num_classes)  classifier logits for query
        epoch: int | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns:
            total_loss: scalar.
            aux: dict with 'contrastive_loss' and 'ce_loss' for logging.
        """
        device = features.device
        batch_size = (features.shape[0] - self.K) // 2

        labels = labels.contiguous().view(-1, 1)
        # mask: 1 if feature shares the same class as the query
        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)  # (B, 2B+K)

        # ── contrastive logits: similarity of query to all features ──
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature,
        )  # (B, 2B+K)

        # ── supervised logits (adjusted by class frequency) ──
        sup_logits_adj = (sup_logits + torch.log(self.weight + 1e-9)) / self.supt

        # ── combined logits: sup part first, then contrastive part ──
        # shape: (B, num_classes + 2B + K)
        combined_logits = torch.cat([sup_logits_adj, anchor_dot_contrast], dim=1)

        # ── numerical stability ──
        logits_max, _ = torch.max(combined_logits, dim=1, keepdim=True)
        logits = combined_logits - logits_max.detach()

        # ── logits mask: exclude self-contrast (query vs query) ──
        logits_mask = torch.ones_like(mask)  # (B, 2B+K)
        logits_mask = torch.scatter(
            logits_mask,
            1,
            torch.arange(batch_size, device=device).view(-1, 1),
            0,
        )
        mask = mask * logits_mask  # remove self-contrast

        # one-hot for supervised part
        one_hot_label = F.one_hot(labels[:batch_size].view(-1),
                                  num_classes=self.num_classes).float()

        # Combine masks:
        #   supervised columns:  one_hot_label * beta      (positive = own class)
        #   contrastive columns: mask * alpha               (positive = same-class samples)
        combined_mask = torch.cat(
            [one_hot_label * self.beta, mask * self.alpha], dim=1
        )

        # Full logits mask (covers both sup and contrastive columns)
        sup_logits_mask = torch.ones(batch_size, self.num_classes, device=device)
        full_logits_mask = torch.cat(
            [sup_logits_mask, self.gamma * logits_mask], dim=1
        )

        # ── InfoNCE ──
        exp_logits = torch.exp(logits) * full_logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (combined_mask * log_prob).sum(1) / combined_mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        # Compute contrastive-only loss for logging (InfoNCE on anchor_dot_contrast only)
        # Note: anchor_dot_contrast is already divided by self.temperature (see line 96)
        with torch.no_grad():
            # Numerical stability on contrastive logits only
            logits_max_c, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits_c = anchor_dot_contrast - logits_max_c.detach()
            exp_logits_c = torch.exp(logits_c) * (self.gamma * logits_mask)
            log_prob_c = logits_c - torch.log(exp_logits_c.sum(1, keepdim=True) + 1e-12)
            # Contrastive positives: same-class samples (mask * alpha)
            contrastive_pos_mask = mask * self.alpha
            mean_log_prob_pos_c = (contrastive_pos_mask * log_prob_c).sum(1) / contrastive_pos_mask.sum(1).clamp(min=1.0)
            contrastive_loss_val = - (self.temperature / self.base_temperature) * mean_log_prob_pos_c
            contrastive_loss_val = contrastive_loss_val.mean()

        aux = {
            'total_loss': loss.detach(),
            'contrastive_loss': contrastive_loss_val.detach(),
            'ce_loss': F.cross_entropy(sup_logits, labels[:batch_size].view(-1)).detach(),
        }
        return loss, aux
