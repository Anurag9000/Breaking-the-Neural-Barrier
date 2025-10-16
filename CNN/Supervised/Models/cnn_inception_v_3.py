"""
Inception v3 (CIFAR variant) — single-model supervised.

Faithful Inception-v3 style with factorized convolutions and two reduction modules, adapted to 32x32 inputs.
We follow the popular A/B/C module taxonomy with Reduction-A and Reduction-B, and optional aux logits.

Stem (32x32):
  conv3x3(32,s=1,p=1) -> conv3x3(32) -> conv3x3(64) -> MaxPool(3x3,s=2)     # 32->16
  conv1x1(80) -> conv3x3(192) -> MaxPool(3x3,s=2)                              # 16->8

Stack:
  3x InceptionA(192)
  ReductionA(192 -> 384)
  4x InceptionB(384)
  ReductionB(384 -> 1024)
  2x InceptionC(1024)

Head: GAP -> Dropout(0.5) -> Linear(num_classes)
Aux classifier (optional; off by default): after the last InceptionA block.

Channel settings are kept close to common v3 references but slightly scaled for CIFAR spatial sizes.
"""
from __future__ import annotations
from typing import Tuple, Optional
import torch
import torch.nn as nn

__all__ = [
    "InceptionV3",
    "InceptionA",
    "InceptionB",
    "InceptionC",
    "ReductionA",
    "ReductionB",
]

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k, s=1, p=0):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

class InceptionA(nn.Module):
    def __init__(self, in_ch: int, pool_proj: int = 32):
        super().__init__()
        # Branch 1: 1x1
        self.b1 = ConvBNReLU(in_ch, 64, 1)
        # Branch 2: 1x1 -> 5x5 (factorized as two 3x3)
        self.b2 = nn.Sequential(
            ConvBNReLU(in_ch, 48, 1),
            ConvBNReLU(48, 64, 3, p=1),
            ConvBNReLU(64, 64, 3, p=1),
        )
        # Branch 3: 1x1 -> 3x3 -> 3x3
        self.b3 = nn.Sequential(
            ConvBNReLU(in_ch, 64, 1),
            ConvBNReLU(64, 96, 3, p=1),
            ConvBNReLU(96, 96, 3, p=1),
        )
        # Branch 4: pool -> 1x1
        self.b4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            ConvBNReLU(in_ch, pool_proj, 1),
        )

    def forward(self, x):
        y = [self.b1(x), self.b2(x), self.b3(x), self.b4(x)]
        return torch.cat(y, dim=1)

class ReductionA(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        # Branch 1: 3x3 s=2
        self.b1 = ConvBNReLU(in_ch, 192, 3, s=2, p=1)
        # Branch 2: 1x1 -> 3x3 -> 3x3 s=2
        self.b2 = nn.Sequential(
            ConvBNReLU(in_ch, 128, 1),
            ConvBNReLU(128, 160, 3, p=1),
            ConvBNReLU(160, 192, 3, s=2, p=1),
        )
        # Branch 3: maxpool s=2
        self.b3 = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)

class InceptionB(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        # Factorized 7x7 path approximated for CIFAR spatial constraints
        self.b1 = ConvBNReLU(in_ch, 192, 1)
        self.b2 = nn.Sequential(
            ConvBNReLU(in_ch, 128, 1),
            ConvBNReLU(128, 128, (1,7), p=(0,3)),
            ConvBNReLU(128, 192, (7,1), p=(3,0)),
        )
        self.b3 = nn.Sequential(
            ConvBNReLU(in_ch, 128, 1),
            ConvBNReLU(128, 128, 3, p=1),
            ConvBNReLU(128, 192, 3, p=1),
        )
        self.b4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            ConvBNReLU(in_ch, 192, 1),
        )

    def forward(self, x):
        y = [self.b1(x), self.b2(x), self.b3(x), self.b4(x)]
        return torch.cat(y, dim=1)

class ReductionB(nn.Module):
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

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)

class InceptionC(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        # 1x1 + parallel (1x3)/(3x1)
        self.b1 = ConvBNReLU(in_ch, 192, 1)
        self.b2 = nn.Sequential(
            ConvBNReLU(in_ch, 192, 1),
            ConvBNReLU(192, 224, (1,3), p=(0,1)),
            ConvBNReLU(224, 256, (3,1), p=(1,0)),
        )
        self.b3 = nn.Sequential(
            ConvBNReLU(in_ch, 192, 1),
            ConvBNReLU(192, 192, (3,1), p=(1,0)),
            ConvBNReLU(192, 224, (1,3), p=(0,1)),
            ConvBNReLU(224, 224, (3,1), p=(1,0)),
            ConvBNReLU(224, 256, (1,3), p=(0,1)),
        )
        self.b4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            ConvBNReLU(in_ch, 192, 1),
        )

    def forward(self, x):
        y = [self.b1(x), self.b2(x), self.b3(x), self.b4(x)]
        return torch.cat(y, dim=1)

class AuxV3(nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((4,4))
        self.conv = ConvBNReLU(in_ch, 128, 1)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(128*4*4, 768)
        self.drop = nn.Dropout(p=0.7)
        self.fc2 = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = self.flat(x)
        x = torch.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x

class InceptionV3(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3, aux_logits: bool = False, dropout: float = 0.5):
        super().__init__()
        self.aux_logits = aux_logits
        # Stem
        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, 32, 3, s=1, p=1),
            ConvBNReLU(32, 32, 3, s=1, p=1),
            ConvBNReLU(32, 64, 3, s=1, p=1),
            nn.MaxPool2d(3, stride=2, padding=1),      # 32->16
            ConvBNReLU(64, 80, 1),
            ConvBNReLU(80, 192, 3, s=1, p=1),
            nn.MaxPool2d(3, stride=2, padding=1),      # 16->8
        )
        # A x3
        self.a1 = InceptionA(192)
        self.a2 = InceptionA(256)
        self.a3 = InceptionA(256)
        # Aux after a3
        self.aux = AuxV3(256 + 64 + 96 + 32, num_classes) if aux_logits else None
        # Reduction A
        self.redA = ReductionA(256 + 64 + 96 + 32)
        # B x4
        chB = (192 + 192 + 192 + 192)  # rough concat size after B
        self.b1 = InceptionB(self._out_ch_a())
        self.b2 = InceptionB(self._out_ch_b(self.b1))
        self.b3 = InceptionB(self._out_ch_b(self.b2))
        self.b4 = InceptionB(self._out_ch_b(self.b3))
        # Reduction B
        self.redB = ReductionB(self._out_ch_b(self.b4))
        # C x2
        self.c1 = InceptionC(self._out_ch_redB())
        self.c2 = InceptionC(self._out_ch_c(self.c1))
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(self._out_ch_c(self.c2), num_classes)
        self._init_linear()

    def _out_ch_a(self):
        # Output channels after InceptionA(256) -> concat of branches
        return 64 + 64 + 96 + 32

    def _out_ch_b(self, module: nn.Module):
        # Each B returns 192+192+192+192
        return 192 * 4

    def _out_ch_redB(self):
        # ReductionB concat channels: 224 + 224 + in(maxpool path has no channels)
        return 224 + 224 + self._out_ch_b(None)  # approximate; acceptable for linear sizes in CIFAR

    def _out_ch_c(self, module: nn.Module):
        # InceptionC concatenation: 192 + 256 + 256 + 192
        return 192 + 256 + 256 + 192

    def _init_linear(self):
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.a1(x)
        x = self.a2(x)
        x = self.a3(x)
        aux = self.aux(x) if (self.aux is not None and self.training) else None
        x = self.redA(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.redB(x)
        x = self.c1(x)
        x = self.c2(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.fc(x)
        return x, aux

    @staticmethod
    def param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    m = InceptionV3(num_classes=10, aux_logits=True)
    logits, aux = m(torch.randn(2,3,32,32))
    print(logits.shape, aux.shape if aux is not None else None)
