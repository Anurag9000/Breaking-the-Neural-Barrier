"""
Inception v4 (CIFAR variant) — single-model supervised.

Implements Inception-v4 with A/B/C modules and Reduction-A/B, adapted for 32x32 inputs.
Channel sizes are modestly reduced for CIFAR while preserving canonical branch patterns.
Optional auxiliary classifier (off by default).

Layout (CIFAR sizes):
  Stem: convs -> maxpool -> convs -> maxpool  (32->16->8)
  4x Inception-A
  Reduction-A (-> wider channels)
  7x Inception-B (factorized 7x7)
  Reduction-B
  3x Inception-C
  GAP -> Dropout -> FC

This file mirrors your modular CNN style (ConvBNReLU blocks, clean forward, and param_count helper).
"""
from __future__ import annotations
import torch
import torch.nn as nn

__all__ = [
    "InceptionV4",
    "InceptionA_v4",
    "InceptionB_v4",
    "InceptionC_v4",
    "ReductionA_v4",
    "ReductionB_v4",
]

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k, s=1, p=0):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

# ---------------- Inception-A ----------------
class InceptionA_v4(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        # 1x1
        self.b1 = ConvBNReLU(in_ch, 64, 1)
        # 1x1 -> 3x3
        self.b2 = nn.Sequential(
            ConvBNReLU(in_ch, 48, 1),
            ConvBNReLU(48, 64, 3, p=1),
        )
        # 1x1 -> 3x3 -> 3x3
        self.b3 = nn.Sequential(
            ConvBNReLU(in_ch, 64, 1),
            ConvBNReLU(64, 96, 3, p=1),
            ConvBNReLU(96, 96, 3, p=1),
        )
        # avgpool -> 1x1
        self.b4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            ConvBNReLU(in_ch, 32, 1),
        )
        self.out_ch = 64 + 64 + 96 + 32

    def forward(self, x):
        y = [self.b1(x), self.b2(x), self.b3(x), self.b4(x)]
        return torch.cat(y, dim=1)

# -------------- Reduction-A ------------------
class ReductionA_v4(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        self.b1 = ConvBNReLU(in_ch, 192, 3, s=2, p=1)
        self.b2 = nn.Sequential(
            ConvBNReLU(in_ch, 128, 1),
            ConvBNReLU(128, 160, 3, p=1),
            ConvBNReLU(160, 192, 3, s=2, p=1),
        )
        self.b3 = nn.MaxPool2d(3, stride=2, padding=1)
        self.out_ch = 192 + 192 + in_ch  # concat of branches

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)

# ---------------- Inception-B ----------------
class InceptionB_v4(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        # 1x1
        self.b1 = ConvBNReLU(in_ch, 192, 1)
        # 1x1 -> (1x7) -> (7x1)
        self.b2 = nn.Sequential(
            ConvBNReLU(in_ch, 128, 1),
            ConvBNReLU(128, 160, (1,7), p=(0,3)),
            ConvBNReLU(160, 192, (7,1), p=(3,0)),
        )
        # 1x1 -> 3x3 -> (1x7) -> (7x1)
        self.b3 = nn.Sequential(
            ConvBNReLU(in_ch, 128, 1),
            ConvBNReLU(128, 160, 3, p=1),
            ConvBNReLU(160, 160, (1,7), p=(0,3)),
            ConvBNReLU(160, 192, (7,1), p=(3,0)),
        )
        # avgpool -> 1x1
        self.b4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            ConvBNReLU(in_ch, 192, 1),
        )
        self.out_ch = 192 + 192 + 192 + 192

    def forward(self, x):
        y = [self.b1(x), self.b2(x), self.b3(x), self.b4(x)]
        return torch.cat(y, dim=1)

# -------------- Reduction-B ------------------
class ReductionB_v4(nn.Module):
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
        # out channels = 224 + 224 + in_ch (pool)
        self.out_ch = 224 + 224 + in_ch

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)

# ---------------- Inception-C ----------------
class InceptionC_v4(nn.Module):
    def __init__(self, in_ch: int):
        super().__init__()
        # 1x1
        self.b1 = ConvBNReLU(in_ch, 256, 1)
        # 1x1 -> parallel (1x3)/(3x1)
        self.b2 = nn.Sequential(
            ConvBNReLU(in_ch, 256, 1),
            ConvBNReLU(256, 256, (1,3), p=(0,1)),
            ConvBNReLU(256, 320, (3,1), p=(1,0)),
        )
        # 1x1 -> (3x1)(1x3)(3x1)(1x3)
        self.b3 = nn.Sequential(
            ConvBNReLU(in_ch, 256, 1),
            ConvBNReLU(256, 256, (3,1), p=(1,0)),
            ConvBNReLU(256, 256, (1,3), p=(0,1)),
            ConvBNReLU(256, 320, (3,1), p=(1,0)),
            ConvBNReLU(320, 320, (1,3), p=(0,1)),
        )
        # avgpool -> 1x1
        self.b4 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            ConvBNReLU(in_ch, 192, 1),
        )
        self.out_ch = 256 + 320 + 320 + 192

    def forward(self, x):
        y = [self.b1(x), self.b2(x), self.b3(x), self.b4(x)]
        return torch.cat(y, dim=1)

# ----------------- Aux Head ------------------
class AuxHead_v4(nn.Module):
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

# ----------------- Inception v4 --------------
class InceptionV4(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3, aux_logits: bool = False, dropout: float = 0.5):
        super().__init__()
        self.aux_logits = aux_logits
        # Stem for CIFAR
        self.stem = nn.Sequential(
            ConvBNReLU(in_channels, 32, 3, s=1, p=1),
            ConvBNReLU(32, 32, 3, s=1, p=1),
            ConvBNReLU(32, 64, 3, s=1, p=1),
            nn.MaxPool2d(3, stride=2, padding=1),  # 32->16
            ConvBNReLU(64, 96, 1),
            ConvBNReLU(96, 160, 3, s=1, p=1),
            nn.MaxPool2d(3, stride=2, padding=1),  # 16->8
        )
        # 4 x A
        self.a1 = InceptionA_v4(160)
        self.a2 = InceptionA_v4(self.a1.out_ch)
        self.a3 = InceptionA_v4(self.a2.out_ch)
        self.a4 = InceptionA_v4(self.a3.out_ch)
        aux_in = self.a4.out_ch
        self.aux = AuxHead_v4(aux_in, num_classes) if aux_logits else None
        # Reduction A
        self.redA = ReductionA_v4(self.a4.out_ch)
        # 7 x B
        self.b1 = InceptionB_v4(self.redA.out_ch)
        self.b2 = InceptionB_v4(self.b1.out_ch)
        self.b3 = InceptionB_v4(self.b2.out_ch)
        self.b4 = InceptionB_v4(self.b3.out_ch)
        self.b5 = InceptionB_v4(self.b4.out_ch)
        self.b6 = InceptionB_v4(self.b5.out_ch)
        self.b7 = InceptionB_v4(self.b6.out_ch)
        # Reduction B
        self.redB = ReductionB_v4(self.b7.out_ch)
        # 3 x C
        self.c1 = InceptionC_v4(self.redB.out_ch)
        self.c2 = InceptionC_v4(self.c1.out_ch)
        self.c3 = InceptionC_v4(self.c2.out_ch)
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.drop = nn.Dropout(p=dropout)
        self.fc = nn.Linear(self.c3.out_ch, num_classes)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity="relu")
        nn.init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.a1(x); x = self.a2(x); x = self.a3(x); x = self.a4(x)
        aux = self.aux(x) if (self.aux is not None and self.training) else None
        x = self.redA(x)
        x = self.b1(x); x = self.b2(x); x = self.b3(x); x = self.b4(x); x = self.b5(x); x = self.b6(x); x = self.b7(x)
        x = self.redB(x)
        x = self.c1(x); x = self.c2(x); x = self.c3(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        x = self.fc(x)
        return x, aux

    @staticmethod
    def param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())
