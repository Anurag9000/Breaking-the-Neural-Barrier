"""
DenseNet-BC for CIFAR-10/100 — single-model supervised.

Canonical DenseNet-BC (Huang et al., 2017) adapted for 32×32 inputs:
- Growth rate k (e.g., 12, 24, 32)
- Bottleneck: 1×1 (4k) → 3×3 (k)
- Compression θ in transitions (default 0.5)
- Stem: 3×3 conv with 2k channels
- 3 DenseBlocks with n layers each (default n=16 → L=6n+4 conv layers; e.g., L=100 for n=16)
- Transitions between blocks: BN→ReLU→1×1 conv (θ·C) → AvgPool2d(2)
- Head: BN→ReLU→GAP→Linear(num_classes)

Matches your modular style: clear blocks, factory, and param_count helper.
"""
from __future__ import annotations
from typing import List
import math
import torch
import torch.nn as nn

__all__ = ["DenseNetCIFAR", "make_densenet_cifar", "DenseLayer", "DenseBlock", "Transition"]

class _ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k, s=1, p=0, bias=False):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

class DenseLayer(nn.Module):
    def __init__(self, in_ch: int, growth_rate: int, drop_rate: float = 0.0):
        super().__init__()
        inter = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, inter, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter)
        self.conv2 = nn.Conv2d(inter, growth_rate, kernel_size=3, padding=1, bias=False)
        self.drop = nn.Dropout(p=drop_rate) if drop_rate > 0 else nn.Identity()
        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(x))
        out = self.conv1(out)
        out = self.relu(self.bn2(out))
        out = self.conv2(out)
        out = self.drop(out)
        return torch.cat([x, out], dim=1)

class DenseBlock(nn.Module):
    def __init__(self, num_layers: int, in_ch: int, growth_rate: int, drop_rate: float):
        super().__init__()
        layers: List[nn.Module] = []
        ch = in_ch
        for _ in range(num_layers):
            layers.append(DenseLayer(ch, growth_rate, drop_rate))
            ch += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_ch = ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class Transition(nn.Module):
    def __init__(self, in_ch: int, theta: float = 0.5):
        super().__init__()
        out_ch = int(math.floor(in_ch * theta))
        self.bn = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2)
        self.out_ch = out_ch
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn(x))
        x = self.conv(x)
        x = self.pool(x)
        return x

class DenseNetCIFAR(nn.Module):
    def __init__(self, growth_rate: int = 12, num_layers_per_block: int = 16, compression: float = 0.5,
                 drop_rate: float = 0.0, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        k = growth_rate
        n = num_layers_per_block
        theta = compression

        # Stem: 3x3 conv with 2k channels
        num_features = 2 * k
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=3, stride=1, padding=1, bias=False)

        # Block 1
        self.db1 = DenseBlock(n, num_features, k, drop_rate)
        num_features = self.db1.out_ch
        self.tr1 = Transition(num_features, theta)
        num_features = self.tr1.out_ch

        # Block 2
        self.db2 = DenseBlock(n, num_features, k, drop_rate)
        num_features = self.db2.out_ch
        self.tr2 = Transition(num_features, theta)
        num_features = self.tr2.out_ch

        # Block 3 (no transition after)
        self.db3 = DenseBlock(n, num_features, k, drop_rate)
        num_features = self.db3.out_ch

        # Head
        self.bn = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(num_features, num_classes)

        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.db1(x)
        x = self.tr1(x)
        x = self.db2(x)
        x = self.tr2(x)
        x = self.db3(x)
        x = self.relu(self.bn(x))
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    @staticmethod
    def param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())


def make_densenet_cifar(growth_rate: int = 12, layers_per_block: int = 16, compression: float = 0.5,
                        drop_rate: float = 0.0, num_classes: int = 10, in_channels: int = 3) -> DenseNetCIFAR:
    return DenseNetCIFAR(growth_rate=growth_rate, num_layers_per_block=layers_per_block, compression=compression,
                         drop_rate=drop_rate, num_classes=num_classes, in_channels=in_channels)

if __name__ == "__main__":
    # Example: DenseNet-BC L=100, k=12 → n=16 per block
    m = make_densenet_cifar(growth_rate=12, layers_per_block=16, compression=0.5, drop_rate=0.0, num_classes=10)
    y = m(torch.randn(2,3,32,32))
    print(y.shape)  # (2,10)
