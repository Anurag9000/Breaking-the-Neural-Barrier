"""
ResNet v1 (post-activation) for CIFAR-10/100 — single-model supervised.

Implements the canonical CIFAR ResNet from He et al. (2015):
- BasicBlock with Conv-BN-ReLU order and ReLU after the residual add (v1, NOT pre-activation).
- Depth options follow 6n+2 formula: 20, 32, 44, 56, 110.
- Stem: 3x3 conv(16), then 3 stages with widths [16, 32, 64].
- Downsampling by stride=2 at the start of stage 2 and 3.
- Head: global average pooling -> linear(num_classes).

Example: depth=20 -> n=3 blocks per stage (3*3*2 + 2 = 20 layers with convs counted).
This file mirrors your clean, modular CNN style with a factory function.
"""
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn

__all__ = ["ResNetCIFAR", "make_resnet_cifar", "BasicBlockV1"]

class BasicBlockV1(nn.Module):
    expansion = 1
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        # He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNetCIFAR(nn.Module):
    def __init__(self, depth: int = 20, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        assert (depth - 2) % 6 == 0, "CIFAR ResNet depth should be 6n+2 (e.g., 20, 32, 44, 56, 110)."
        n = (depth - 2) // 6
        widths = [16, 32, 64]

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)
        self.relu  = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(16,  widths[0], n, stride=1)
        self.layer2 = self._make_layer(widths[0], widths[1], n, stride=2)
        self.layer3 = self._make_layer(widths[1], widths[2], n, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(widths[2], num_classes)

        # He init for head
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)

    def _make_layer(self, in_planes: int, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers: List[nn.Module] = []
        layers.append(BasicBlockV1(in_planes, planes, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlockV1(planes, planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    @staticmethod
    def param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())


def make_resnet_cifar(depth: int = 20, num_classes: int = 10, in_channels: int = 3) -> ResNetCIFAR:
    return ResNetCIFAR(depth=depth, num_classes=num_classes, in_channels=in_channels)
