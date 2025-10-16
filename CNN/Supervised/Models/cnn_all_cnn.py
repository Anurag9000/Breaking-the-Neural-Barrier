"""
All-CNN (Springenberg et al., 2014) for CIFAR-10/100 — single-model supervised.

Implements the widely used All-CNN-C configuration (no pooling; downsampling by strided convs):

Input 32x32x3
Block1: conv3x3 96 -> ReLU -> conv3x3 96 -> ReLU -> conv3x3 s=2 96 -> ReLU  (32->16)
Dropout(0.5)
Block2: conv3x3 192 -> ReLU -> conv3x3 192 -> ReLU -> conv3x3 s=2 192 -> ReLU (16->8)
Dropout(0.5)
Block3: conv3x3 192 -> ReLU -> conv1x1 192 -> ReLU -> conv1x1 num_classes
GlobalAvgPool -> Flatten

Reference: "Striving for Simplicity: The All Convolutional Net" (2014).
This file matches your clean modular coding style.
"""
from __future__ import annotations
import torch
import torch.nn as nn

__all__ = ["AllCNN"]

class AllCNN(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3, p_drop: float = 0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 96, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, kernel_size=3, stride=2, padding=1, bias=True),  # downsample 32->16
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_drop),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, stride=2, padding=1, bias=True),  # downsample 16->8
            nn.ReLU(inplace=True),
            nn.Dropout(p=p_drop),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, num_classes, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = self.flatten(x)  # (N, num_classes)
        return x

    @staticmethod
    def param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    m = AllCNN(num_classes=10)
    y = m(torch.randn(2,3,32,32))
    print(y.shape)  # (2,10)
