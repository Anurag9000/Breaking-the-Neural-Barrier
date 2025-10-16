"""
Wide-ResNet (WRN) for CIFAR-10/100 — single-model supervised.

Canonical WRN from "Wide Residual Networks" (Zagoruyko & Komodakis, 2016):
- Pre-activation BasicBlock with optional dropout between the two 3x3 convs.
- Depth must satisfy depth = 6*N + 4 (e.g., 16, 22, 28, 40). Commonly WRN-28-10 (N=4, widen_factor=10).
- Widths: [16, 16*k, 32*k, 64*k]. Downsample at the start of stage 2 and 3.
- Head: BN→ReLU→GAP→FC.

This mirrors your modular style with a factory and param_count helper.
"""
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn

__all__ = ["WideResNet", "make_wrn_cifar", "WRNBasicBlock"]

class WRNBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes: int, planes: int, stride: int = 1, p_drop: float = 0.0):
        super().__init__()
        self.bn1  = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2  = nn.BatchNorm2d(planes)
        self.drop = nn.Dropout(p=p_drop) if p_drop > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = None
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(x))
        identity = x if self.shortcut is None else self.shortcut(out)
        out = self.conv1(out)
        out = self.drop(self.relu(self.bn2(out)))
        out = self.conv2(out)
        out += identity
        return out

class _BlockGroup(nn.Module):
    def __init__(self, in_planes: int, planes: int, num_blocks: int, stride: int, p_drop: float):
        super().__init__()
        layers: List[nn.Module] = []
        layers.append(WRNBasicBlock(in_planes, planes, stride=stride, p_drop=p_drop))
        for _ in range(1, num_blocks):
            layers.append(WRNBasicBlock(planes, planes, stride=1, p_drop=p_drop))
        self.group = nn.Sequential(*layers)

    def forward(self, x):
        return self.group(x)

class WideResNet(nn.Module):
    def __init__(self, depth: int = 28, widen_factor: int = 10, num_classes: int = 10, in_channels: int = 3, p_drop: float = 0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0, "WRN depth should be 6*N+4 (e.g., 16, 22, 28, 40)."
        N = (depth - 4) // 6
        widths = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]

        self.conv1 = nn.Conv2d(in_channels, widths[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.group1 = _BlockGroup(widths[0], widths[1], N, stride=1, p_drop=p_drop)
        self.group2 = _BlockGroup(widths[1], widths[2], N, stride=2, p_drop=p_drop)
        self.group3 = _BlockGroup(widths[2], widths[3], N, stride=2, p_drop=p_drop)

        self.bn = nn.BatchNorm2d(widths[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(widths[3], num_classes)

        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.group1(x)
        x = self.group2(x)
        x = self.group3(x)
        x = self.relu(self.bn(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    @staticmethod
    def param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())


def make_wrn_cifar(depth: int = 28, widen_factor: int = 10, num_classes: int = 10, in_channels: int = 3, p_drop: float = 0.0) -> WideResNet:
    return WideResNet(depth=depth, widen_factor=widen_factor, num_classes=num_classes, in_channels=in_channels, p_drop=p_drop)

if __name__ == "__main__":
    m = make_wrn_cifar(depth=28, widen_factor=10, num_classes=10, p_drop=0.3)
    y = m(torch.randn(2,3,32,32))
    print(y.shape)  # (2,10)
