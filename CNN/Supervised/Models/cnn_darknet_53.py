"""
Darknet-53 (CIFAR variant) — single-model supervised.

Canonical Darknet-53 from YOLOv3 backbone, adapted to 32×32 inputs.
Blocks: Residual units with 1×1 reduce then 3×3 expand (post-activation BN+LeakyReLU).
Stage repeats follow [1, 2, 8, 8, 4]. Downsampling by stride=2 at each stage start.
Head: global average pooling → linear(num_classes).

Differences vs ImageNet version: CIFAR stem uses 3×3 conv stride=1; channel widths preserved.
"""
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn

__all__ = ["Darknet53", "make_darknet53_cifar"]

class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch, out_ch, k, s=1, p=0, act: str = "lrelu"):
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        if act == "lrelu":
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        elif act == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif act == "silu":
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)

class ResidualUnit(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = ConvBNAct(ch, ch // 2, 1)
        self.conv2 = ConvBNAct(ch // 2, ch, 3, p=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))

class Darknet53(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        # Stem
        self.stem = ConvBNAct(in_channels, 32, 3, s=1, p=1)
        # Stages: (out_channels, repeats)
        cfg = [
            (64,  1),
            (128, 2),
            (256, 8),
            (512, 8),
            (1024,4),
        ]
        layers: List[nn.Module] = []
        in_ch = 32
        for out_ch, n in cfg:
            # Downsample conv (stride=2)
            layers.append(ConvBNAct(in_ch, out_ch, 3, s=2, p=1))
            in_ch = out_ch
            # Residual units
            blocks = [ResidualUnit(in_ch) for _ in range(n)]
            layers.append(nn.Sequential(*blocks))
        self.body = nn.Sequential(*layers)

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1024, num_classes)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='leaky_relu')
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    @staticmethod
    def param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())


def make_darknet53_cifar(num_classes: int = 10, in_channels: int = 3) -> Darknet53:
    return Darknet53(num_classes=num_classes, in_channels=in_channels)

if __name__ == "__main__":
    m = make_darknet53_cifar(num_classes=10)
    y = m(torch.randn(2,3,32,32))
    print(y.shape)  # (2,10)
