"""
LeNet-5 (original-style) for CIFAR-10/100 in PyTorch.
- Uses 5x5 VALID convolutions, AvgPool subsampling, and Tanh activations as in the canonical architecture.
- Adapted to 3 input channels (RGB) and arbitrary num_classes.
- Spatial progression for 32x32 input:
  32x32x3 -> C1 5x5 valid -> 28x28x6 -> AvgPool 2x2 -> 14x14x6
            -> C3 5x5 valid -> 10x10x16 -> AvgPool 2x2 -> 5x5x16
            -> C5 5x5 valid -> 1x1x120 -> FC(84) -> FC(num_classes)

This file mirrors the clean, single-module style used in the uploaded CNN_STL models.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LeNet5"]

class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        # Original LeNet-5 used 6 and 16 channels for C1/C3, and 120 for C5
        self.c1 = nn.Conv2d(in_channels, 6, kernel_size=5, padding=0, bias=True)
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(6, 16, kernel_size=5, padding=0, bias=True)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c5 = nn.Conv2d(16, 120, kernel_size=5, padding=0, bias=True)
        self.f6 = nn.Linear(120, 84)
        self.out = nn.Linear(84, num_classes)

        # Kaiming uniform is fine even with Tanh; Xavier could also be used.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # C1 -> Tanh -> S2
        x = torch.tanh(self.c1(x))
        x = self.s2(x)
        # C3 -> Tanh -> S4
        x = torch.tanh(self.c3(x))
        x = self.s4(x)
        # C5 -> Tanh
        x = torch.tanh(self.c5(x))  # (N, 120, 1, 1)
        x = x.view(x.size(0), -1)   # (N, 120)
        # F6 -> Tanh -> Out
        x = torch.tanh(self.f6(x))
        x = self.out(x)
        return x

    @staticmethod
    def param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    # Quick shape sanity test
    m = LeNet5(num_classes=10, in_channels=3)
    y = m(torch.randn(2,3,32,32))
    print(y.shape)  # (2,10)
