"""
CSPNet (CSPDarknet-53 style, CIFAR variant) — single-model supervised.

Implements Cross Stage Partial networks per Wang et al., with a CSPDarknet-53-style backbone
adapted to 32×32 inputs. Each stage:
  • Downsample: 3×3 stride-2 conv to set output channels C.
  • Split: 1×1 conv to two tensors (C/2 each): route (skip) and main.
  • Main path: N residual units (1×1 reduce → 3×3 conv) operating at C/2 channels.
  • Transition: concatenate [route, main] → 1×1 conv to fuse back to C channels.

Stage repeats follow Darknet-53: [1, 2, 8, 8, 4]. We keep LeakyReLU(0.1) activations like the canonical design.
Head: GAP → Linear(num_classes).

This mirrors your modular coding style: clear blocks, stage module, factory, and param_count helper.
"""
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn

__all__ = ["CSPDarknet53", "make_cspdarknet53_cifar", "CSPStage", "ResidualUnit"]

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
    """Darknet-style residual: 1×1 reduce then 3×3 expand at the same width, residual add."""
    def __init__(self, ch: int):
        super().__init__()
        hidden = ch // 2
        self.cv1 = ConvBNAct(ch, hidden, 1)
        self.cv2 = ConvBNAct(hidden, ch, 3, p=1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x))

class CSPStage(nn.Module):
    """CSP stage: downsample → split → residual stack on main → concat → fuse 1×1."""
    def __init__(self, in_ch: int, out_ch: int, n_blocks: int):
        super().__init__()
        self.down = ConvBNAct(in_ch, out_ch, 3, s=2, p=1)
        c_half = out_ch // 2
        self.route = ConvBNAct(out_ch, c_half, 1)   # skip branch
        self.main1 = ConvBNAct(out_ch, c_half, 1)   # main branch pre
        self.blocks = nn.Sequential(*[ResidualUnit(c_half) for _ in range(n_blocks)])
        self.main2 = ConvBNAct(c_half, c_half, 1)   # post residual 1x1
        self.fuse = ConvBNAct(2*c_half, out_ch, 1)  # fusion conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        route = self.route(x)
        y = self.main2(self.blocks(self.main1(x)))
        x = torch.cat([y, route], dim=1)
        x = self.fuse(x)
        return x

class CSPDarknet53(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        # Stem
        self.stem = ConvBNAct(in_channels, 32, 3, s=1, p=1)
        # Stages with (out_channels, repeats)
        cfg = [
            (64,  1),
            (128, 2),
            (256, 8),
            (512, 8),
            (1024,4),
        ]
        in_ch = 32
        stages: List[nn.Module] = []
        for out_ch, n in cfg:
            stages.append(CSPStage(in_ch, out_ch, n_blocks=n))
            in_ch = out_ch
        self.body = nn.Sequential(*stages)
        # Head
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_ch, num_classes)
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


def make_cspdarknet53_cifar(num_classes: int = 10, in_channels: int = 3) -> CSPDarknet53:
    return CSPDarknet53(num_classes=num_classes, in_channels=in_channels)

if __name__ == "__main__":
    m = make_cspdarknet53_cifar(num_classes=10)
    y = m(torch.randn(2,3,32,32))
    print(y.shape)  # (2,10)
