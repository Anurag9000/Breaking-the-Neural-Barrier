"""
Inception-ResNet (v2-style, CIFAR variant) — single-model supervised.

CIFAR-friendly Inception-ResNet with residual scaling. Structure:
  Stem (32->16->8)
  5x Inception-ResNet-A (scale=0.1)
  Reduction-A
  10x Inception-ResNet-B (scale=0.1)
  Reduction-B
  5x Inception-ResNet-C (scale=0.1)
  GAP -> Dropout -> Linear(num_classes)

This mirrors popular IRNv2 designs while reducing channels for 32x32 inputs.
"""
from __future__ import annotations
import torch
import torch.nn as nn

__all__ = [
    "InceptionResNet",
    "IRN_A",
    "IRN_B",
    "IRN_C",
    "RedA",
    "RedB",
]

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k, s=1, p=0):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

class IRN_A(nn.Module):
    def __init__(self, in_ch: int, scale: float = 0.1):
        super().__init__()
        self.scale = scale
        b1 = [ConvBNReLU(in_ch, 32, 1)]
        b2 = [ConvBNReLU(in_ch, 32, 1), ConvBNReLU(32, 32, 3, p=1)]
        b3 = [ConvBNReLU(in_ch, 32, 1), ConvBNReLU(32, 48, 3, p=1), ConvBNReLU(48, 64, 3, p=1)]
        self.b1 = nn.Sequential(*b1)
        self.b2 = nn.Sequential(*b2)
        self.b3 = nn.Sequential(*b3)
        self.conv = nn.Sequential(
            nn.Conv2d(32 + 32 + 64, in_ch, kernel_size=1, bias=True),
            nn.BatchNorm2d(in_ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        out = self.conv(out)
        return self.act(x + self.scale * out)

class IRN_B(nn.Module):
    def __init__(self, in_ch: int, scale: float = 0.1):
        super().__init__()
        self.scale = scale
        b1 = [ConvBNReLU(in_ch, 192, 1)]
        b2 = [ConvBNReLU(in_ch, 128, 1), ConvBNReLU(128, 160, (1,7), p=(0,3)), ConvBNReLU(160, 192, (7,1), p=(3,0))]
        self.b1 = nn.Sequential(*b1)
        self.b2 = nn.Sequential(*b2)
        self.conv = nn.Sequential(
            nn.Conv2d(192 + 192, in_ch, kernel_size=1, bias=True),
            nn.BatchNorm2d(in_ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = torch.cat([self.b1(x), self.b2(x)], dim=1)
        out = self.conv(out)
        return self.act(x + self.scale * out)

class IRN_C(nn.Module):
    def __init__(self, in_ch: int, scale: float = 0.1):
        super().__init__()
        self.scale = scale
        b1 = [ConvBNReLU(in_ch, 192, 1)]
        b2 = [ConvBNReLU(in_ch, 192, 1), ConvBNReLU(192, 224, (1,3), p=(0,1)), ConvBNReLU(224, 256, (3,1), p=(1,0))]
        self.b1 = nn.Sequential(*b1)
        self.b2 = nn.Sequential(*b2)
        self.conv = nn.Sequential(
            nn.Conv2d(192 + 256, in_ch, kernel_size=1, bias=True),
            nn.BatchNorm2d(in_ch),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = torch.cat([self.b1(x), self.b2(x)], dim=1)
        out = self.conv(out)
        return self.act(x + self.scale * out)

class RedA(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.b1 = ConvBNReLU(in_ch, 192, 3, s=2, p=1)
        self.b2 = nn.Sequential(
            ConvBNReLU(in_ch, 128, 1),
            ConvBNReLU(128, 160, 3, p=1),
            ConvBNReLU(160, 192, 3, s=2, p=1),
        )
        self.b3 = nn.MaxPool2d(3, stride=2, padding=1)
        self.out_ch = 192 + 192 + in_ch

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)

class RedB(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.b1 = nn.Sequential(
            ConvBNReLU(in_ch, 192, 1),
            ConvBNReLU(192, 224, 3, s=2, p=1),
        )
        self.b2 = nn.Sequential(
            ConvBNReLU(in_ch, 192, 1),
            ConvBNReLU(192, 192, (1,7), p=(0,3)),
            ConvBNReLU(192, 224, (7,1), p=(3,0)),
            ConvBNReLU(224, 224, 3, s=2, p=1),
        )
        self.b3 = nn.MaxPool2d(3, stride=2, padding=1)
        self.out_ch = 224 + 224 + in_ch

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)

class InceptionResNet(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3, dropout: float = 0.5, scale: float = 0.1):
        super().__init__()
        # Stem (CIFAR)
        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, 32, 3, s=1, p=1),
            ConvBNReLU(32, 32, 3, s=1, p=1),
            ConvBNReLU(32, 64, 3, s=1, p=1),
            nn.MaxPool2d(3, stride=2, padding=1),  # 32->16
            ConvBNReLU(64, 80, 1),
            ConvBNReLU(80, 192, 3, s=1, p=1),
            nn.MaxPool2d(3, stride=2, padding=1),  # 16->8
        )
        # 5x A
        ch = 192
        self.a = nn.Sequential(*[IRN_A(ch, scale) for _ in range(5)])
        # Red A
        self.redA = RedA(ch)
        ch = self.redA.out_ch
        # 10x B
        self.b = nn.Sequential(*[IRN_B(ch, scale) for _ in range(10)])
        # Red B
        self.redB = RedB(ch)
        ch = self.redB.out_ch
        # 5x C
        self.c = nn.Sequential(*[IRN_C(ch, scale) for _ in range(5)])
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(ch, num_classes)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.a(x)
        x = self.redA(x)
        x = self.b(x)
        x = self.redB(x)
        x = self.c(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.fc(x)
        return x

    @staticmethod
    def param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())
