"""
ResNet-D tweaks (CIFAR) — single-model supervised.

Implements the ResNet-D modifications on top of standard ResNet for small images:
  • Downsampling blocks move stride from the first conv to the *middle* conv (3x3) in each block.
  • Shortcut uses AvgPool(stride=2) → 1x1 Conv(stride=1) instead of a strided 1x1 Conv.

We provide both BasicBlock (for depths 18/34) and Bottleneck (for depth 50). Widths follow [64,128,256,512].
Stem is a single 3×3 conv suitable for CIFAR (no ImageNet 7×7). Head is GAP→Linear.

Depth options:
  • 18:  [2,2,2,2]  (BasicBlockD)
  • 34:  [3,4,6,3]  (BasicBlockD)
  • 50:  [3,4,6,3]  (BottleneckD)

This matches your modular coding style and factory pattern.
"""
from __future__ import annotations
from typing import List, Type
import torch
import torch.nn as nn

__all__ = [
    "ResNetD_CIFAR",
    "make_resnetd_cifar",
    "BasicBlockD",
    "BottleneckD",
]

# ---------------- Blocks -----------------
class BasicBlockD(nn.Module):
    expansion = 1
    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        # ResNet-D: stride in 2nd conv (3x3)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.downsample = downsample

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

class BottleneckD(nn.Module):
    expansion = 4
    def __init__(self, in_planes: int, planes: int, stride: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        # ResNet-D: stride in 3x3 conv
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
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
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# -------------- Model ---------------------
class ResNetD_CIFAR(nn.Module):
    def __init__(self, block: Type[nn.Module], layers: List[int], num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        # Stages with downsample at 2,3,4
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * (block.expansion if hasattr(block, 'expansion') else 1), num_classes)

        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)

    def _downsample_resnetd(self, in_planes: int, out_planes: int, stride: int) -> nn.Module:
        # ResNet-D shortcut: AvgPool(stride) -> 1x1 Conv(stride=1) -> BN
        return nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=stride, ceil_mode=True, count_include_pad=False) if stride > 1 else nn.Identity(),
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_planes),
        )

    def _make_layer(self, block: Type[nn.Module], planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        out_planes = planes * (block.expansion if hasattr(block, 'expansion') else 1)
        if stride != 1 or self.in_planes != out_planes:
            downsample = self._downsample_resnetd(self.in_planes, out_planes, stride)
        layers: List[nn.Module] = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = out_planes
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, 1, None))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    @staticmethod
    def param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

# -------------- Factory -------------------
def make_resnetd_cifar(depth: int = 18, num_classes: int = 10, in_channels: int = 3) -> ResNetD_CIFAR:
    if depth == 18:
        return ResNetD_CIFAR(BasicBlockD, [2,2,2,2], num_classes=num_classes, in_channels=in_channels)
    elif depth == 34:
        return ResNetD_CIFAR(BasicBlockD, [3,4,6,3], num_classes=num_classes, in_channels=in_channels)
    elif depth == 50:
        return ResNetD_CIFAR(BottleneckD, [3,4,6,3], num_classes=num_classes, in_channels=in_channels)
    else:
        raise ValueError("Supported depths for ResNet-D: 18, 34, 50")

if __name__ == "__main__":
    for d in [18,34,50]:
        m = make_resnetd_cifar(depth=d, num_classes=10)
        y = m(torch.randn(2,3,32,32))
        print(d, y.shape)
