"""
ResNeXt for CIFAR-10/100 — single-model supervised.

Canonical ResNeXt bottleneck with cardinality (groups) and width-per-group, adapted to 32x32 inputs.
Common CIFAR configs: ResNeXt-29 (8x64d) and ResNeXt-29 (16x64d).

Design:
  • Stem: 3×3 conv(64)
  • Stages (3): widths [64, 128, 256] as bottleneck planes; expansion=4 for output channels
  • Downsample with stride=2 at the start of stage 2 and 3
  • Head: GAP → FC

Block:
  1×1 reduce → 3×3 grouped conv (groups=cardinality, width_per_group=base_width) → 1×1 expand (×4)
  BN+ReLU after each conv (post-activation style, like ResNet v1 bottleneck)
  Shortcut projection when spatial/width changes

Depth options: we expose a CIFAR-friendly layer config (3,3,3) which corresponds to popular "29-layer" families.
You can change the repeats per stage via the factory.
"""
from __future__ import annotations
from typing import List, Type
import torch
import torch.nn as nn

__all__ = [
    "ResNeXtCIFAR",
    "make_resnext_cifar",
    "BottleneckX",
]

class BottleneckX(nn.Module):
    expansion = 4
    def __init__(self, in_planes: int, planes: int, stride: int, cardinality: int, base_width: int, downsample: nn.Module | None = None):
        super().__init__()
        D = int(planes * (base_width / 64.0))  # per group width
        C = cardinality
        width = D * C
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=C, bias=False)
        self.bn2   = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNeXtCIFAR(nn.Module):
    def __init__(self, layers: List[int] = [3,3,3], num_classes: int = 10, in_channels: int = 3,
                 cardinality: int = 8, base_width: int = 64):
        super().__init__()
        self.in_planes = 64
        self.cardinality = cardinality
        self.base_width = base_width

        # Stem
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)

        # Stages (planes are bottleneck inner planes)
        self.layer1 = self._make_layer(planes=64,  blocks=layers[0], stride=1)
        self.layer2 = self._make_layer(planes=128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(planes=256, blocks=layers[2], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256 * BottleneckX.expansion, num_classes)
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        out_planes = planes * BottleneckX.expansion
        if stride != 1 or self.in_planes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )
        layers: List[nn.Module] = []
        layers.append(BottleneckX(self.in_planes, planes, stride, self.cardinality, self.base_width, downsample))
        self.in_planes = out_planes
        for _ in range(1, blocks):
            layers.append(BottleneckX(self.in_planes, planes, 1, self.cardinality, self.base_width, None))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x)
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

# ---------------- Factory -------------------
def make_resnext_cifar(depth: int = 29, cardinality: int = 8, base_width: int = 64, num_classes: int = 10, in_channels: int = 3) -> ResNeXtCIFAR:
    """Creates a CIFAR ResNeXt. For depth=29, we use layers=[3,3,3]."""
    if depth == 29:
        layers = [3,3,3]
    elif depth == 47:
        layers = [5,5,5]
    else:
        # Fallback: approximate mapping depth ≈ 9n+2 -> n=(depth-2)/9 -> layers=[n,n,n]
        n = max(1, int(round((depth - 2) / 9)))
        layers = [n, n, n]
    return ResNeXtCIFAR(layers=layers, num_classes=num_classes, in_channels=in_channels, cardinality=cardinality, base_width=base_width)

if __name__ == "__main__":
    for cfg in [(29,8,64),(29,16,64)]:
        m = make_resnext_cifar(depth=cfg[0], cardinality=cfg[1], base_width=cfg[2], num_classes=10)
        y = m(torch.randn(2,3,32,32))
        print(cfg, y.shape)
