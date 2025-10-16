"""
Network in Network (NiN) for CIFAR-10/100 — single-model supervised.

Implements the canonical NiN idea: replace large fully connected heads with MLP-conv (1x1 conv) blocks and
Global Average Pooling to produce class scores. CIFAR-friendly configuration mirrors widely used references:

Stage1: conv(5x5,192) -> ReLU -> conv(1x1,160) -> ReLU -> conv(1x1,96)  -> ReLU -> MaxPool(3x3,s=2) -> Dropout(0.5)
Stage2: conv(5x5,192) -> ReLU -> conv(1x1,192) -> ReLU -> conv(1x1,192) -> ReLU -> AvgPool(3x3,s=2) -> Dropout(0.5)
Stage3: conv(3x3,192) -> ReLU -> conv(1x1,192) -> ReLU -> conv(1x1,num_classes) -> GlobalAvgPool -> Flatten

Input: 3x32x32, Output: logits [N, num_classes]
"""
from __future__ import annotations
import torch
import torch.nn as nn

__all__ = ["NiN"]

class MLPConv(nn.Module):
    def __init__(self, in_ch: int, c_mid1: int, c_mid2: int, k: int, p: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, c_mid1, kernel_size=k, padding=p, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid1, c_mid2, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_mid2, c_mid2, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class NiN(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3, p_drop: float = 0.5):
        super().__init__()
        # Stage 1
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 192, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 160, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 96, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 32->16
            nn.Dropout(p=p_drop),
        )
        # Stage 2
        self.stage2 = nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=5, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=2),  # 16->8
            nn.Dropout(p=p_drop),
        )
        # Stage 3 + Classifier
        self.stage3 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, num_classes, kernel_size=1, padding=0, bias=True),
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
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.gap(x)
        x = self.flatten(x)  # (N, num_classes)
        return x

    @staticmethod
    def param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    m = NiN(num_classes=10)
    y = m(torch.randn(2,3,32,32))
    print(y.shape)  # (2,10)
