"""Plain CIFAR ResNet-32, adapted from the archived model implementation."""

import torch.nn.functional as functional
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        self.channel_padding = (out_channels - in_channels) // 2

    def forward(self, images):
        residual = images
        if self.stride != 1 or self.channel_padding:
            # Option A downsamples spatially and pads channels without parameters.
            residual = residual[:, :, ::self.stride, ::self.stride]
            residual = functional.pad(
                residual, (0, 0, 0, 0, self.channel_padding, self.channel_padding)
            )
        features = functional.relu(self.bn1(self.conv1(images)))
        features = self.bn2(self.conv2(features))
        return functional.relu(features + residual)


class ResNet32Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_stage(16, 16, stride=1)
        self.layer2 = self._make_stage(16, 32, stride=2)
        self.layer3 = self._make_stage(32, 64, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _make_stage(in_channels, out_channels, stride):
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride),
            *(BasicBlock(out_channels, out_channels) for _ in range(4)),
        )

    def forward(self, images):
        features = functional.relu(self.bn1(self.conv1(images)))
        features = self.layer3(self.layer2(self.layer1(features)))
        return self.avgpool(features).flatten(1)


class ResNet32(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet32Backbone()
        self.fc = nn.Linear(64, 100)
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, images):
        return self.fc(self.backbone(images))


def create_model() -> ResNet32:
    return ResNet32()
