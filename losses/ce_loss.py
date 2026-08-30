"""Standard Cross-Entropy loss."""

import torch.nn as nn


class CELoss(nn.Module):
    """Standard cross-entropy loss for multi-class classification."""

    def __init__(self):
        super().__init__()
        self._loss = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        return self._loss(logits, targets)
