"""
CBAM-ResNet (CIFAR) — single-model supervised.

Convolutional Block Attention Module (CBAM; Woo et al., 2018) added to ResNet blocks.
CIFAR-friendly architecture:
  • Stem: 3×3 conv(64)
  • Stages: widths [64,128,256,512] with downsampling at stages 2–4 starts
  • Blocks: Basic (18/34) or Bottleneck (50) + CBAM (Channel + Spatial attention)
  • Head: GAP → FC

CBAM = Channel attention (global avg/max → 2-layer MLP → sigmoid) followed by Spatial attention
(avg+max across channel → 7×7 conv → sigmoid) applied sequentially.
"""
from __future__ import annotations
from typing import List, Type
import torch
import torch.nn as nn

__all__ = [
    "CBAMResNetCIFAR",
    "make_cbam_resnet_cifar",
    "CBAM",
    "CBAMBasicBlock",
    "CBAMBottleneck",
]

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        w = self.sigmoid(avg_out + max_out)
        return x * w

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3,7)
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.cat([avg_map, max_map], dim=1)
        a = self.conv(a)
        w = self.sigmoid(a)
        return x * w

class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x

class CBAMBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes: int, planes: int, stride: int = 1, reduction: int = 16, spatial_kernel: int = 7,
                 downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.cbam  = CBAM(planes, reduction=reduction, spatial_kernel=spatial_kernel)
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
        out = self.cbam(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class CBAMBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes: int, planes: int, stride: int = 1, reduction: int = 16, spatial_kernel: int = 7,
                 downsample: nn.Module | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.cbam  = CBAM(planes * self.expansion, reduction=reduction, spatial_kernel=spatial_kernel)
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
        out = self.cbam(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class CBAMResNetCIFAR(nn.Module):
    def __init__(self, block: Type[nn.Module], layers: List[int], num_classes: int = 10, in_channels: int = 3,
                 reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.in_planes = 64
        self.reduction = reduction
        self.spatial_kernel = spatial_kernel

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
        layers.append(block(self.in_planes, planes, stride=stride, reduction=self.reduction, spatial_kernel=self.spatial_kernel, downsample=downsample))
        self.in_planes = out_planes
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, stride=1, reduction=self.reduction, spatial_kernel=self.spatial_kernel, downsample=None))
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
def make_cbam_resnet_cifar(depth: int = 18, num_classes: int = 10, in_channels: int = 3, reduction: int = 16, spatial_kernel: int = 7) -> CBAMResNetCIFAR:
    if depth == 18:
        return CBAMResNetCIFAR(CBAMBasicBlock, [2,2,2,2], num_classes=num_classes, in_channels=in_channels, reduction=reduction, spatial_kernel=spatial_kernel)
    elif depth == 34:
        return CBAMResNetCIFAR(CBAMBasicBlock, [3,4,6,3], num_classes=num_classes, in_channels=in_channels, reduction=reduction, spatial_kernel=spatial_kernel)
    elif depth == 50:
        return CBAMResNetCIFAR(CBAMBottleneck, [3,4,6,3], num_classes=num_classes, in_channels=in_channels, reduction=reduction, spatial_kernel=spatial_kernel)
    else:
        raise ValueError("Supported depths for CBAM-ResNet: 18, 34, 50")

if __name__ == "__main__":
    for d in [18,34,50]:
        m = make_cbam_resnet_cifar(depth=d, num_classes=10)
        y = m(torch.randn(2,3,32,32))
        print(d, y.shape)
