"""
BAM-ResNet (CIFAR) — single-model supervised.

Bottleneck Attention Module (BAM; Park et al., 2018) added inside ResNet blocks.
BAM computes a gate from **parallel channel and spatial branches**:
  • Channel: GAP → 2-layer MLP (reduction r) → logits (C)
  • Spatial: 1×1 reduce → stack of dilated 3×3 convs → 1×1 → logits (H×W)
The two logits are broadcast-summed and passed through sigmoid. Following the paper, we use a
**residual gate**: y = x * (1 + σ(M_c(x)+M_s(x))).

CIFAR-friendly stack:
  • Stem: 3×3 conv(64)
  • Stages: widths [64,128,256,512] with downsampling at stages 2–4 starts
  • Blocks: Basic (18/34) or Bottleneck (50) with BAM right before the residual add
  • Head: GAP → FC

Matches your modular + factory pattern.
"""
from __future__ import annotations
from typing import List, Type
import torch
import torch.nn as nn

__all__ = [
    "BAMResNetCIFAR",
    "make_bam_resnet_cifar",
    "BAM",
    "BAMBasicBlock",
    "BAMBottleneck",
]

class BAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_dilations: List[int] | None = None):
        super().__init__()
        if spatial_dilations is None:
            spatial_dilations = [1, 2, 4]
        # Channel branch
        hidden = max(channels // reduction, 8)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        # Spatial branch
        red = max(channels // reduction, 8)
        layers: List[nn.Module] = [nn.Conv2d(channels, red, kernel_size=1, bias=False), nn.ReLU(inplace=True)]
        in_ch = red
        for d in spatial_dilations:
            layers += [
                nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True),
            ]
        layers += [nn.Conv2d(in_ch, 1, kernel_size=1, bias=False)]
        self.sa = nn.Sequential(*layers)
        self.sig = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mc = self.ca(x)                 # (N,C,1,1)
        ms = self.sa(x)                 # (N,1,H,W)
        # Broadcast-sum logits then sigmoid; residual gate (1 + a)
        att = self.sig(mc + ms)
        return x * (1 + att)

class BAMBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes: int, planes: int, stride: int = 1, reduction: int = 16,
                 spatial_dilations: List[int] | None = None, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.bam   = BAM(planes, reduction=reduction, spatial_dilations=spatial_dilations)
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
        out = self.bam(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class BAMBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes: int, planes: int, stride: int = 1, reduction: int = 16,
                 spatial_dilations: List[int] | None = None, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.bam   = BAM(planes * self.expansion, reduction=reduction, spatial_dilations=spatial_dilations)
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
        out = self.bam(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class BAMResNetCIFAR(nn.Module):
    def __init__(self, block: Type[nn.Module], layers: List[int], num_classes: int = 10, in_channels: int = 3,
                 reduction: int = 16, spatial_dilations: List[int] | None = None):
        super().__init__()
        self.in_planes = 64
        self.reduction = reduction
        self.spatial_dilations = spatial_dilations

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
        layers.append(block(self.in_planes, planes, stride=stride, reduction=self.reduction,
                            spatial_dilations=self.spatial_dilations, downsample=downsample))
        self.in_planes = out_planes
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1, reduction=self.reduction,
                                spatial_dilations=self.spatial_dilations, downsample=None))
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
def make_bam_resnet_cifar(depth: int = 18, num_classes: int = 10, in_channels: int = 3,
                          reduction: int = 16, spatial_dilations: List[int] | None = None) -> BAMResNetCIFAR:
    if depth == 18:
        return BAMResNetCIFAR(BAMBasicBlock, [2,2,2,2], num_classes=num_classes, in_channels=in_channels, reduction=reduction, spatial_dilations=spatial_dilations)
    elif depth == 34:
        return BAMResNetCIFAR(BAMBasicBlock, [3,4,6,3], num_classes=num_classes, in_channels=in_channels, reduction=reduction, spatial_dilations=spatial_dilations)
    elif depth == 50:
        return BAMResNetCIFAR(BAMBottleneck, [3,4,6,3], num_classes=num_classes, in_channels=in_channels, reduction=reduction, spatial_dilations=spatial_dilations)
    else:
        raise ValueError("Supported depths for BAM-ResNet: 18, 34, 50")
