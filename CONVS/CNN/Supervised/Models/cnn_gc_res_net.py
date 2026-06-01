"""
GC-ResNet (CIFAR) — single-model supervised.

Global Context (GC) block per GCNet (Cao et al., 2019):
  • Attention pooling: generate spatial attention map with 1×1 conv → softmax over H×W →
    aggregate a context vector (N,C,1,1).
  • Transform: 1×1 → ReLU → 1×1 (reduction r) to produce a channel-wise modulation.
  • Fusion: add-only, mul-only (sigmoid), or add+mul (default add-only as in GCNet ablations).

Integrated into CIFAR-style ResNet blocks (Basic 18/34, Bottleneck 50). GC is applied right before
residual addition (on the last conv output), similar placement to SE/CBAM variants.

Head: GAP → FC. Matches your modular + factory style.
"""
from __future__ import annotations
from typing import List, Type
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "GCResNetCIFAR",
    "make_gc_resnet_cifar",
    "GCBlock",
    "GCBasicBlock",
    "GCBottleneck",
]

class GCBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, fusion: str = "add"):
        """
        Args:
            channels: input/output channels
            reduction: bottleneck reduction ratio for transform (min 8)
            fusion: one of {"add", "mul", "add_mul"}
        """
        super().__init__()
        self.channels = channels
        self.fusion = fusion
        hidden = max(channels // reduction, 8)
        # Attention pooling to compute context vector
        self.att = nn.Conv2d(channels, 1, kernel_size=1)
        # Transform for additive path
        self.t_add = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        # Transform for multiplicative path
        self.t_mul = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        # Init last convs of add path to zero to start as identity
        for m in self.t_add.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        # Zero-init the last conv of add so residual starts near-identity
        nn.init.zeros_(self.t_add[-1].weight)
        for m in self.t_mul.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        # Attention map over spatial locations
        att_logits = self.att(x).view(n, 1, -1)
        att = F.softmax(att_logits, dim=-1).view(n, 1, h, w)  # (N,1,H,W)
        # Context vector
        context = (x * att).sum(dim=(2,3), keepdim=True)  # (N,C,1,1)

        if self.fusion == "add":
            y = x + self.t_add(context)
        elif self.fusion == "mul":
            y = x * self.t_mul(context)
        elif self.fusion == "add_mul":
            y = x + self.t_add(context)
            y = y * self.t_mul(context)
        else:
            raise ValueError("fusion must be one of {'add','mul','add_mul'}")
        return y

class GCBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes: int, planes: int, stride: int = 1, reduction: int = 16,
                 fusion: str = "add", downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.gc    = GCBlock(planes, reduction=reduction, fusion=fusion)
        self.downsample = downsample

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        out = self.gc(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class GCBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes: int, planes: int, stride: int = 1, reduction: int = 16,
                 fusion: str = "add", downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.gc    = GCBlock(planes * self.expansion, reduction=reduction, fusion=fusion)
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
        out = self.gc(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class GCResNetCIFAR(nn.Module):
    def __init__(self, block: Type[nn.Module], layers: List[int], num_classes: int = 10, in_channels: int = 3,
                 reduction: int = 16, fusion: str = "add"):
        super().__init__()
        self.in_planes = 64
        self.reduction = reduction
        self.fusion = fusion

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
        layers.append(block(self.in_planes, planes, stride=stride, reduction=self.reduction, fusion=self.fusion, downsample=downsample))
        self.in_planes = out_planes
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1, reduction=self.reduction, fusion=self.fusion, downsample=None))
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
def make_gc_resnet_cifar(depth: int = 18, num_classes: int = 10, in_channels: int = 3,
                         reduction: int = 16, fusion: str = "add") -> GCResNetCIFAR:
    if depth == 18:
        return GCResNetCIFAR(GCBasicBlock, [2,2,2,2], num_classes=num_classes, in_channels=in_channels, reduction=reduction, fusion=fusion)
    elif depth == 34:
        return GCResNetCIFAR(GCBasicBlock, [3,4,6,3], num_classes=num_classes, in_channels=in_channels, reduction=reduction, fusion=fusion)
    elif depth == 50:
        return GCResNetCIFAR(GCBottleneck, [3,4,6,3], num_classes=num_classes, in_channels=in_channels, reduction=reduction, fusion=fusion)
    else:
        raise ValueError("Supported depths for GC-ResNet: 18, 34, 50")
