"""
ECA-ResNet (CIFAR) — single-model supervised.

Efficient Channel Attention (ECA; Wang et al., 2020) added to ResNet blocks.
ECA replaces the SE MLP with a parameter-free dimensionality reduction:
  GAP → 1D conv on channel descriptor with small odd kernel k → sigmoid → channel-wise reweight.
Default kernel is computed as k = odd(| (log2(C) / gamma) + b |) with gamma=2, b=1.

CIFAR-friendly architecture:
  • Stem: 3×3 conv(64)
  • Stages: widths [64,128,256,512] with downsampling at stages 2–4 starts
  • Blocks: Basic (18/34) or Bottleneck (50) + ECA
  • Head: GAP → FC

Style mirrors your modules + factory pattern.
"""
from __future__ import annotations
from typing import List, Type
import math
import torch
import torch.nn as nn

__all__ = [
    "ECAResNetCIFAR",
    "make_eca_resnet_cifar",
    "ECALayer",
    "ECABasicBlock",
    "ECABottleneck",
]

class ECALayer(nn.Module):
    def __init__(self, channels: int, k_size: int | None = None, gamma: float = 2.0, b: float = 1.0):
        super().__init__()
        if k_size is None:
            t = int(abs((math.log2(channels) / gamma) + b))
            k_size = t if t % 2 == 1 else t + 1
            k_size = max(1, k_size)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k_size, padding=k_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,C,H,W)
        y = self.avg_pool(x)     # (N,C,1,1)
        y = y.squeeze(-1).transpose(-1, -2)  # (N,1,C)
        y = self.conv1d(y)
        y = y.transpose(-1, -2).unsqueeze(-1)  # (N,C,1,1)
        w = self.sigmoid(y)
        return x * w

class ECABasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes: int, planes: int, stride: int = 1, k_size: int | None = None,
                 gamma: float = 2.0, b: float = 1.0, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.eca   = ECALayer(planes, k_size, gamma, b)
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
        out = self.eca(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ECABottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes: int, planes: int, stride: int = 1, k_size: int | None = None,
                 gamma: float = 2.0, b: float = 1.0, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.eca   = ECALayer(planes * self.expansion, k_size, gamma, b)
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
        out = self.eca(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ECAResNetCIFAR(nn.Module):
    def __init__(self, block: Type[nn.Module], layers: List[int], num_classes: int = 10, in_channels: int = 3,
                 k_size: int | None = None, gamma: float = 2.0, b: float = 1.0):
        super().__init__()
        self.in_planes = 64
        self.k_size = k_size
        self.gamma = gamma
        self.b = b

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        exp = block.expansion if hasattr(block, 'expansion') else 1
        self.fc = nn.Linear(512 * exp, num_classes)
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
        layers.append(block(self.in_planes, planes, stride=stride, k_size=self.k_size, gamma=self.gamma, b=self.b, downsample=downsample))
        self.in_planes = out_planes
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1, k_size=self.k_size, gamma=self.gamma, b=self.b, downsample=None))
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
def make_eca_resnet_cifar(depth: int = 18, num_classes: int = 10, in_channels: int = 3,
                          k_size: int | None = None, gamma: float = 2.0, b: float = 1.0) -> ECAResNetCIFAR:
    if depth == 18:
        return ECAResNetCIFAR(ECABasicBlock, [2,2,2,2], num_classes=num_classes, in_channels=in_channels, k_size=k_size, gamma=gamma, b=b)
    elif depth == 34:
        return ECAResNetCIFAR(ECABasicBlock, [3,4,6,3], num_classes=num_classes, in_channels=in_channels, k_size=k_size, gamma=gamma, b=b)
    elif depth == 50:
        return ECAResNetCIFAR(ECABottleneck, [3,4,6,3], num_classes=num_classes, in_channels=in_channels, k_size=k_size, gamma=gamma, b=b)
    else:
        raise ValueError("Supported depths for ECA-ResNet: 18, 34, 50")

if __name__ == "__main__":
    for d in [18,34,50]:
        m = make_eca_resnet_cifar(depth=d, num_classes=10)
        y = m(torch.randn(2,3,32,32))
        print(d, y.shape)
