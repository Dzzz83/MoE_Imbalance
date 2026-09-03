"""
Balanced Softmax Loss — Ren et al., ECCV 2020.

Adjusts the logits by adding the log of per-class sample counts before
the softmax operation:

    logits'_j = logits_j + log(n_j)

where n_j is the number of training samples in class j.

This is equivalent to LAL with τ=1 when class priors are used instead
of raw counts (the constant offset log(N) cancels in softmax).

Reference:
    Ren, J., et al. "Balanced Meta-Softmax for Long-Tailed Visual
    Recognition." ECCV 2020.
    https://arxiv.org/abs/2007.10740
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedSoftmaxLoss(nn.Module):
    """
    Balanced Softmax Loss.

    Args:
        class_counts: Tensor of shape (C,) with the number of training
                      samples in each class (unnormalized counts).
    """

    def __init__(self, class_counts: torch.Tensor):
        super().__init__()
        # Register as buffer so it moves with the model to the correct device
        self.register_buffer('log_counts', torch.log(class_counts.float() + 1e-12))

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Balanced Softmax loss.

        Args:
            logits:  (B, C) pre-softmax logits.
            targets: (B,) ground-truth class indices.
        Returns:
            Scalar loss.
        """
        adjusted = logits + self.log_counts.unsqueeze(0)  # (B, C)
        return F.cross_entropy(adjusted, targets)
