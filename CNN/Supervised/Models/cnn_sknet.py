"""
SKNet-ResNet (CIFAR) — single-model supervised.

Selective Kernel Networks (SKNet; Li et al., 2019) integrate an adaptive receptive-field selection
module (SK-Conv) inside residual blocks. Each SK-Conv builds two parallel branches with different
receptive fields (we use 3×3 and 3×3 with dilation=2 to emulate ~5×5), aggregates them with a
soft attention over branches derived from global pooled features, and fuses the weighted sum.

CIFAR-friendly stack:
  • Stem: 3×3 conv(64)
  • Stages: widths [64,128,256,512] with downsampling at stages 2–4 starts
  • Blocks: Basic (18/34) or Bottleneck (50) with SK-Conv replacing the second 3×3 (Basic) or the 3×3 middle (Bottleneck)
  • Head: GAP → FC

Matches your modular + factory style.
"""
from __future__ import annotations
from typing import List, Type
import torch
import torch.nn as nn

__all__ = [
    "SKResNetCIFAR",
    "make_sknet_cifar",
    "SKConv",
    "SKBasicBlock",
    "SKBottleneck",
]

class SKConv(nn.Module):
    """Selective Kernel convolution with 2 branches.

    Args:
        channels: input/output channels (branches keep the same width)
        groups: grouped convolution groups (default 1)
        reduction: squeeze ratio for attention bottleneck
        stride: stride applied to both branches
    """
    def __init__(self, channels: int, groups: int = 1, reduction: int = 16, stride: int = 1):
        super().__init__()
        self.channels = channels
        hidden = max(channels // reduction, 8)

        # Two receptive-field branches
        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        # Dilation=2 with 3×3 approx. a 5×5 RF (keeps params modest)
        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=2, dilation=2, groups=groups, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # Fuse → attention over branches (softmax along branch dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        # Produce attention logits for 2 branches, then softmax
        self.fc2 = nn.Conv2d(hidden, 2 * channels, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=1)  # across branch dimension after view

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        U = x1 + x2
        s = self.pool(U)
        z = self.relu(self.fc1(s))
        a_b = self.fc2(z)            # (N,2C,1,1)
        a_b = a_b.view(x.size(0), 2, self.channels, 1, 1)
        att = self.softmax(a_b)      # along dim=1 (two branches)
        a = att[:, 0]
        b = att[:, 1]
        out = a * x1 + b * x2
        return out

class SKBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes: int, planes: int, stride: int = 1, reduction: int = 16,
                 groups: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        # Replace second 3×3 by SK-Conv at the same width
        self.sk    = SKConv(planes, groups=groups, reduction=reduction, stride=1)
        self.bn2   = nn.BatchNorm2d(planes)
        self.downsample = downsample

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.sk(out);  out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class SKBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes: int, planes: int, stride: int = 1, reduction: int = 16,
                 groups: int = 1, downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNum2d = nn.BatchNorm2d(planes)
        self.sk    = SKConv(planes, groups=groups, reduction=reduction, stride=stride)
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
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.sk(out);  out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class SKResNetCIFAR(nn.Module):
    def __init__(self, block: Type[nn.Module], layers: List[int], num_classes: int = 10, in_channels: int = 3,
                 reduction: int = 16, groups: int = 1):
        super().__init__()
        self.in_planes = 64
        self.reduction = reduction
        self.groups = groups

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
        layers.append(block(self.in_planes, planes, stride=stride, reduction=self.reduction, groups=self.groups, downsample=downsample))
        self.in_planes = out_planes
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1, reduction=self.reduction, groups=self.groups, downsample=None))
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
def make_sknet_cifar(depth: int = 18, num_classes: int = 10, in_channels: int = 3, reduction: int = 16, groups: int = 1) -> SKResNetCIFAR:
    if depth == 18:
        return SKResNetCIFAR(SKBasicBlock, [2,2,2,2], num_classes=num_classes, in_channels=in_channels, reduction=reduction, groups=groups)
    elif depth == 34:
        return SKResNetCIFAR(SKBasicBlock, [3,4,6,3], num_classes=num_classes, in_channels=in_channels, reduction=reduction, groups=groups)
    elif depth == 50:
        return SKResNetCIFAR(SKBottleneck, [3,4,6,3], num_classes=num_classes, in_channels=in_channels, reduction=reduction, groups=groups)
    else:
        raise ValueError("Supported depths for SKNet-ResNet: 18, 34, 50")

if __name__ == "__main__":
    for d in [18,34,50]:
        m = make_sknet_cifar(depth=d, num_classes=10)
        y = m(torch.randn(2,3,32,32))
        print(d, y.shape)
