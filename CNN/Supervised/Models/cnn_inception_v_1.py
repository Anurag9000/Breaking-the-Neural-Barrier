"""
Inception v1 (GoogLeNet, CIFAR variant) — single-model supervised.

This is a faithful CIFAR adaptation using the canonical Inception (v1) module layout
and the widely used channel configuration from the original GoogLeNet paper, with
stride/pool choices adjusted for 32x32 inputs:

Stem (for 32x32):
  conv(3x3,64,s=1,p=1) -> ReLU -> MaxPool(3x3,s=2)        # 32->16
  conv(1x1,64) -> ReLU -> conv(3x3,192,p=1) -> ReLU
  MaxPool(3x3,s=2)                                         # 16->8

Inception stack:
  3a: (64,  96,128, 16, 32, 32)
  3b: (128,128,192, 32, 96, 64)
  MaxPool(3x3,s=2)                                         # 8->4
  4a: (192, 96,208, 16, 48, 64)   [aux1 optional]
  4b: (160,112,224, 24, 64, 64)
  4c: (128,128,256, 24, 64, 64)
  4d: (112,144,288, 32, 64, 64)   [aux2 optional]
  4e: (256,160,320, 32,128,128)
  MaxPool(3x3,s=2)                                         # 4->2
  5a: (256,160,320, 32,128,128)
  5b: (384,192,384, 48,128,128)

Head: GAP -> Dropout(0.4) -> Linear(num_classes)

Auxiliary classifiers (optional; off by default): after 4a and 4d.
They use adaptive pooling to 4x4 on CIFAR, a 1x1 conv(128), then Linear(1024), Dropout(0.7), Linear(num_classes).
Loss = main + 0.3*(aux1+aux2) when enabled.

This file mirrors the clean modular style used in your CNN_* files.
"""
from __future__ import annotations
from typing import Tuple, Optional
import torch
import torch.nn as nn

__all__ = ["InceptionV1", "InceptionModule"]

class InceptionModule(nn.Module):
    def __init__(self,
                 in_ch: int,
                 c1x1: int,
                 c3x3_reduce: int, c3x3: int,
                 c5x5_reduce: int, c5x5: int,
                 pool_proj: int):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_ch, c1x1, kernel_size=1, bias=True),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_ch, c3x3_reduce, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c3x3_reduce, c3x3, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_ch, c5x5_reduce, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c5x5_reduce, c5x5, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_ch, pool_proj, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)

class AuxClassifier(nn.Module):
    def __init__(self, in_ch: int, num_classes: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv = nn.Conv2d(in_ch, 128, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128*4*4, 1024)
        self.dropout = nn.Dropout(p=0.7)
        self.fc2 = nn.Linear(1024, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.relu(self.conv(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class InceptionV1(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3, aux_logits: bool = False, dropout: float = 0.4):
        super().__init__()
        self.aux_logits = aux_logits
        # Stem
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 32->16
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 16->8

        # Inception stack
        self.in3a = InceptionModule(192, 64, 96, 128, 16, 32, 32)
        self.in3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 8->4

        self.in4a = InceptionModule(480, 192, 96, 208, 16, 48, 64)
        self.in4b = InceptionModule(512, 160, 112, 224, 24, 64, 64)
        self.in4c = InceptionModule(512, 128, 128, 256, 24, 64, 64)
        self.in4d = InceptionModule(512, 112, 144, 288, 32, 64, 64)
        self.in4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 4->2

        self.in5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.in5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        # Aux heads (optional)
        if aux_logits:
            self.aux1 = AuxClassifier(512, num_classes)  # after 4a output channels
            self.aux2 = AuxClassifier(528, num_classes)  # after 4d output channels
        else:
            self.aux1 = None
            self.aux2 = None

        # Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1024, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        # Stem
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)

        # 3x
        x = self.in3a(x)
        x = self.in3b(x)
        x = self.pool3(x)

        # 4x
        x = self.in4a(x)
        aux1 = self.aux1(x) if (self.aux_logits and self.training) else None
        x = self.in4b(x)
        x = self.in4c(x)
        x = self.in4d(x)
        aux2 = self.aux2(x) if (self.aux_logits and self.training) else None
        x = self.in4e(x)
        x = self.pool4(x)

        # 5x
        x = self.in5a(x)
        x = self.in5b(x)

        # Head
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x, aux1, aux2

    @staticmethod
    def param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    m = InceptionV1(num_classes=10, aux_logits=True)
    logits, a1, a2 = m(torch.randn(2,3,32,32))
    print(logits.shape, a1.shape if a1 is not None else None, a2.shape if a2 is not None else None)
