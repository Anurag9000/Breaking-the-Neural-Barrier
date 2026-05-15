"""
SE-ResNet (CIFAR) — single-model supervised.

Squeeze-and-Excitation (SE) added to ResNet blocks (Hu et al., 2018). CIFAR-friendly stem and stages:
  • Stem: 3×3 conv(64)
  • Stages: widths [64,128,256,512] with downsampling at the start of stages 2–4
  • Blocks: Basic (for 18/34) or Bottleneck (for 50) with channel attention via SE
  • Head: GAP → FC

SE module: global avg pool → 2-layer MLP (reduction r, default 16) → sigmoid → channel-wise reweight.
This mirrors your modular style and factory API.
"""
from __future__ import annotations
from typing import List, Type
import torch
import torch.nn as nn

__all__ = [
    "SEResNetCIFAR",
    "make_se_resnet_cifar",
    "SELayer",
    "SEBasicBlock",
    "SEBottleneck",
]

class SELayer(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.sig = nn.Sigmoid()

        # Init last conv to near-zero so early training is stable
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.fc2.weight)
        if self.fc1.bias is not None: nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None: nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.relu(self.fc1(w))
        w = self.sig(self.fc2(w))
        return x * w

class SEBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes: int, planes: int, stride: int = 1, reduction: int = 16, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.se    = SELayer(planes, reduction)
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
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class SEBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes: int, planes: int, stride: int = 1, reduction: int = 16, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.se    = SELayer(planes * self.expansion, reduction)
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
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class SEResNetCIFAR(nn.Module):
    def __init__(self, block: Type[nn.Module], layers: List[int], num_classes: int = 10, in_channels: int = 3, reduction: int = 16):
        super().__init__()
        self.in_planes = 64
        self.reduction = reduction

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * (block.expansion if hasattr(block, 'expansion') else 1), num_classes)

        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)

    def _make_layer(self, block: Type[nn.Module], planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        out_planes = planes * (block.expansion if hasattr(block, 'expansion') else 1)
        if stride != 1 or self.in_planes != out_planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes),
            )
        layers: List[nn.Module] = []
        layers.append(block(self.in_planes, planes, stride=stride, reduction=self.reduction, downsample=downsample))
        self.in_planes = out_planes
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1, reduction=self.reduction, downsample=None))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x)
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

# ---------------- Factory -------------------
def make_se_resnet_cifar(depth: int = 18, num_classes: int = 10, in_channels: int = 3, reduction: int = 16) -> SEResNetCIFAR:
    if depth == 18:
        return SEResNetCIFAR(SEBasicBlock, [2,2,2,2], num_classes=num_classes, in_channels=in_channels, reduction=reduction)
    elif depth == 34:
        return SEResNetCIFAR(SEBasicBlock, [3,4,6,3], num_classes=num_classes, in_channels=in_channels, reduction=reduction)
    elif depth == 50:
        return SEResNetCIFAR(SEBottleneck, [3,4,6,3], num_classes=num_classes, in_channels=in_channels, reduction=reduction)
    else:
        raise ValueError("Supported depths for SE-ResNet: 18, 34, 50")
