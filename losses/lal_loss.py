"""
Logit-Adjusted Loss (LAL)  — Menon et al., ICLR 2021.

Adds a label-dependent offset to the logits before softmax:

    f(x)_y' = f(x)_y + τ · log(π_y)

where π_y = n_y / N is the class prior (frequency in the training set).

Lower values of τ reduce the adjustment; τ=0 recovers standard CE.
We use τ=1.0 as recommended by the paper for CIFAR-100-LT IR=100.
"""

import torch
import torch.nn as nn


class LALLoss(nn.Module):
    """
    Logit-Adjusted Loss.

    Args:
        class_priors: Tensor of shape (C,) with π_y = n_y / N.
        tau:          Temperature scaling the logit offset (default 1.0).
    """

    def __init__(self, class_priors: torch.Tensor, tau: float = 1.0):
        super().__init__()
        self.tau = tau
        # register as buffer so it moves with the model to the correct device
        self.register_buffer('log_prior', torch.log(class_priors + 1e-12))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        adjusted = logits + self.tau * self.log_prior.unsqueeze(0)  # (B, C)
        return nn.functional.cross_entropy(adjusted, targets)
