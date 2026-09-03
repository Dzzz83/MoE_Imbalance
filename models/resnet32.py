"""
ResNet-32 for CIFAR-100 — standard CIFAR variant (He et al., 2016).

Uses Option-A shortcuts (zero-padded slicing) as in the original CIFAR-10 paper
and the official PaCo implementation (https://github.com/dvlab-research/PaCo).

Also provides PaCoResNet32 which extends the backbone with:
  - An MLP projection head (for contrastive embeddings)
  - A classifier head whose weights act as parametric class centres
  - A momentum-updated key encoder + queue (MoCo-style memory bank)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


# ---------------------------------------------------------------------------
# BasicBlock with Option-A shortcut
# ---------------------------------------------------------------------------

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                # Zero-padded slicing (no learnable params)
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant", 0
                    )
                )
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, planes, kernel_size=1,
                              stride=stride, bias=False),
                    nn.BatchNorm2d(planes),
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


# ---------------------------------------------------------------------------
# ResNet-32 backbone (no classifier)
# ---------------------------------------------------------------------------

class ResNet32Backbone(nn.Module):
    """Returns 64‑d feature vector after global average pooling."""

    def __init__(self, num_blocks=None):
        super().__init__()
        if num_blocks is None:
            num_blocks = [5, 5, 5]
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(64, num_blocks[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s, option='A'))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avgpool(out)
        return out.view(out.size(0), -1)  # (B, 64)


# ---------------------------------------------------------------------------
# Full ResNet-32  (backbone + linear classifier)
# ---------------------------------------------------------------------------

class ResNet32(nn.Module):
    """Standard ResNet-32 for CIFAR-100.  Returns logits."""

    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.backbone = ResNet32Backbone()
        self.fc = nn.Linear(64, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone(x)
        return self.fc(features)


# ---------------------------------------------------------------------------
# MoCo-based PaCo model  (Cui et al., ICCV 2021)
# ---------------------------------------------------------------------------

@torch.no_grad()
def concat_all_gather(tensor):
    """Performs all-gather operation (single-GPU: no-op)."""
    return tensor


class PaCoResNet32(nn.Module):
    """
    PaCo model adapted for single-GPU training.

    Architecture (matches official PaCo implementation):
      - encoder_q: backbone + MLP projection head → embeddings
      - encoder_k: momentum-updated copy of encoder_q
      - linear_q:  backbone features → classifier logits  (weights = parametric centres)
      - linear_k:  momentum-updated copy of linear_q
      - queue:     memory bank of past key embeddings  (MoCo-style)

    Forward returns:
        features: [2*B + K, dim]  (query emb, key emb, queue)
        labels:   [2*B + K]       (corresponding labels)
        logits:   [B, num_classes] (classifier logits for query, used as parametric-centre similarities)
    """

    def __init__(
        self,
        num_classes: int = 100,
        dim: int = 32,
        K: int = 1024,
        m: float = 0.999,
        mlp: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.K = K
        self.m = m
        self.num_classes = num_classes

        # ── shared backbone ──
        feat_dim = 64  # ResNet-32 output after avgpool

        # ── query encoder (backbone + projection head) ──
        self.encoder_q = nn.Sequential(
            ResNet32Backbone(),
        )
        # MLP projection head
        if mlp:
            self.encoder_q.add_module('proj_head', nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, dim),
            ))
        else:
            self.encoder_q.add_module('proj_head', nn.Linear(feat_dim, dim))

        # ── key encoder (momentum-updated copy) ──
        self.encoder_k = nn.Sequential(
            ResNet32Backbone(),
        )
        if mlp:
            self.encoder_k.add_module('proj_head', nn.Sequential(
                nn.Linear(feat_dim, feat_dim),
                nn.BatchNorm1d(feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(feat_dim, dim),
            ))
        else:
            self.encoder_k.add_module('proj_head', nn.Linear(feat_dim, dim))

        # copy init
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                     self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # ── classifier heads ──
        self.linear_q = nn.Linear(feat_dim, num_classes)
        self.linear_k = nn.Linear(feat_dim, num_classes)
        for param_q, param_k in zip(self.linear_q.parameters(),
                                     self.linear_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # ── queue (memory bank) ──
        self.register_buffer('queue', torch.randn(K, dim))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer('queue_label', torch.randint(0, num_classes, (K,)))
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        # ── hook to grab features before the projection head for classifier ──
        self.feat_q = None
        self.feat_k = None

        def hook_q(_, __, output):
            self.feat_q = output.view(output.size(0), -1)

        def hook_k(_, __, output):
            self.feat_k = output.view(output.size(0), -1)

        # Register hook on the backbone's avgpool layer directly (robust to
        # architecture changes in the Sequential wrapper)
        self.encoder_q[0].avgpool.register_forward_hook(hook_q)
        self.encoder_k[0].avgpool.register_forward_hook(hook_k)

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                     self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)
        for param_q, param_k in zip(self.linear_q.parameters(),
                                     self.linear_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        # queue must hold an integer number of batches
        if ptr + batch_size > self.K:
            # wrap around to start if not enough space
            remaining = self.K - ptr
            self.queue[ptr:self.K] = keys[:remaining]
            self.queue_label[ptr:self.K] = labels[:remaining]
            self.queue[:batch_size - remaining] = keys[remaining:]
            self.queue_label[:batch_size - remaining] = labels[remaining:]
            self.queue_ptr[0] = (ptr + batch_size) % self.K
        else:
            self.queue[ptr:ptr + batch_size] = keys
            self.queue_label[ptr:ptr + batch_size] = labels
            self.queue_ptr[0] = (ptr + batch_size) % self.K

    def forward(self, im_q, im_k=None, labels=None):
        if not self.training:
            return self._inference(im_q)

        # ── query ──
        q = self.encoder_q(im_q)                     # (B, dim)
        q = F.normalize(q, dim=1)
        logits_q = self.linear_q(self.feat_q)         # (B, C) classifier logits

        # ── key (no grad) ──
        with torch.no_grad():
            self._momentum_update_key_encoder()
            k = self.encoder_k(im_k)                  # (B, dim)
            k = F.normalize(k, dim=1)

        # ── assemble features ──
        features = torch.cat([q, k, self.queue.clone().detach()], dim=0)  # (2B+K, dim)
        all_labels = torch.cat([labels, labels, self.queue_label], dim=0)  # (2B+K,)

        # ── enqueue ──
        self._dequeue_and_enqueue(k, labels)

        return features, all_labels, logits_q

    def _inference(self, x):
        features = self.encoder_q[0](x)   # backbone only
        features = features.view(features.size(0), -1)
        logits = self.linear_q(features)
        return logits
