"""
Locally Connected CNN (LCL-CNN) for CIFAR-10/100 — single-model supervised.

Implements a custom LocallyConnected2d layer (no weight sharing), suitable for fixed spatial sizes.
Architecture (for 32x32 inputs):
  Conv3x3(64) -> ReLU -> MaxPool2 (32->16)
  LocallyConnected2d(64->128, k=3, stride=1, padding=1, input_size=16x16) -> ReLU -> MaxPool2 (16->8)
  Conv1x1(128->num_classes) -> GlobalAvgPool -> Flatten

Notes:
- LocallyConnected2d removes translation weight sharing: each (h,w) position has its own kernel.
- Parameter count grows with H*W; this config keeps it reasonable for CIFAR.
- This file matches the modular coding style in your uploaded CNN files.
"""
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LocallyConnected2d", "LCLNet"]

class LocallyConnected2d(nn.Module):
    """A 2D locally-connected layer (no weight sharing).

    Args:
        in_channels:   Number of input channels
        out_channels:  Number of output channels
        kernel_size:   int kernel size (assumed square)
        stride:        int stride
        padding:       int padding
        input_size:    (H, W) tuple for the expected input spatial size BEFORE this layer
        bias:          include bias per output position

    Shapes:
        Input:  (N, C_in, H_in, W_in)
        Output: (N, C_out, H_out, W_out)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 0, input_size: Tuple[int,int] = (32,32), bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.input_size = input_size

        H_in, W_in = input_size
        H_out = (H_in + 2*padding - kernel_size) // stride + 1
        W_out = (W_in + 2*padding - kernel_size) // stride + 1
        self.H_out = H_out
        self.W_out = W_out
        L = H_out * W_out

        # Weight for each spatial location
        # shape: (C_out, C_in, K*K, L)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size*kernel_size, L))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, L))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, C_in, H_in, W_in)
        N, C_in, H_in, W_in = x.shape
        assert (H_in, W_in) == self.input_size, f"Expected input spatial {self.input_size}, got {(H_in, W_in)}"
        # Unfold to patches: (N, C_in*K*K, L)
        patches = F.unfold(x, kernel_size=self.kernel_size, padding=self.padding, stride=self.stride)  # (N, C_in*K*K, L)
        L = patches.size(-1)
        assert L == self.H_out * self.W_out, "Unfold produced unexpected spatial length."
        # Reshape for einsum: weight (C_out, C_in, K*K, L), patches (N, C_in*K*K, L)
        patches = patches.view(N, C_in, self.kernel_size*self.kernel_size, L)
        # Out: (N, C_out, L)
        out = torch.einsum('ocpl,ncpl->nol', self.weight, patches)
        if self.bias is not None:
            out = out + self.bias.unsqueeze(0)  # (1, C_out, L)
        out = out.view(N, self.out_channels, self.H_out, self.W_out)
        return out

class LCLNet(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3, p_drop: float = 0.0):
        super().__init__()
        # Stage 1: conv + pool -> 16x16
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # Stage 2: locally connected on 16x16 -> 16x16
        self.lc = LocallyConnected2d(64, 128, kernel_size=3, stride=1, padding=1, input_size=(16,16), bias=True)
        self.act = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 16->8

        # Head: 1x1 conv to classes, GAP
        self.classifier = nn.Sequential(
            nn.Conv2d(128, num_classes, kernel_size=1, bias=True),
        )
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()

        self._init_linear_biases()

    def _init_linear_biases(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)                 # (N,64,16,16)
        x = self.lc(x)                     # (N,128,16,16)
        x = self.act(x)
        x = self.pool2(x)                  # (N,128,8,8)
        x = self.classifier(x)             # (N,num_classes,8,8)
        x = self.gap(x)                    # (N,num_classes,1,1)
        x = self.flatten(x)                # (N,num_classes)
        return x

    @staticmethod
    def param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    m = LCLNet(num_classes=10)
    y = m(torch.randn(2,3,32,32))
    print(y.shape)  # (2,10)
