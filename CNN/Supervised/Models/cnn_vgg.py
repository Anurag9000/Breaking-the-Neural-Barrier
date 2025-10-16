"""
VGG for CIFAR-10/100 (A/D/E) with optional BatchNorm — single-model supervised.

Implements the canonical VGG-11/16/19 (a.k.a. A/D/E) block layout adapted to 32x32 inputs:
- All convs are 3x3, stride=1, padding=1
- Five max-pool (2x2, stride 2) reduce 32->16->8->4->2->1, yielding 512x1x1 before the head
- Classifier: 512 -> num_classes (CIFAR variant)
- Optional BatchNorm after each conv (VGG-*-BN)

Variants:
A  = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M]    # VGG-11
D  = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M]  # VGG-16
E  = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M]  # VGG-19

This file mirrors your clean modular style (see CNN_STL.py) while keeping the widely accepted VGG definitions.
"""
from __future__ import annotations
import torch
import torch.nn as nn
from typing import List, Union

__all__ = ["VGG", "make_vgg"]

Cfg = List[Union[int, str]]
VGG_CFGS = {
    "A":  [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],             # VGG-11
    "D":  [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],  # VGG-16
    "E":  [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],  # VGG-19
}

class VGG(nn.Module):
    def __init__(self, cfg: Cfg, num_classes: int = 10, in_channels: int = 3, batch_norm: bool = False, dropout: float = 0.0):
        super().__init__()
        self.features = self._make_layers(cfg, in_channels, batch_norm)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 512x1x1 for CIFAR
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(512, num_classes),
        )
        self._init_weights()

    def _make_layers(self, cfg: Cfg, in_ch: int, bn: bool) -> nn.Sequential:
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_ch, v, kernel_size=3, padding=1, bias=not bn)
                if bn:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_ch = v
        return nn.Sequential(*layers)

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
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x

    @staticmethod
    def param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())


def make_vgg(variant: str = "D", num_classes: int = 10, in_channels: int = 3, batch_norm: bool = True, dropout: float = 0.0) -> VGG:
    if variant not in VGG_CFGS:
        raise ValueError(f"Unknown VGG variant '{variant}'. Choose from {list(VGG_CFGS.keys())}.")
    return VGG(cfg=VGG_CFGS[variant], num_classes=num_classes, in_channels=in_channels, batch_norm=batch_norm, dropout=dropout)

if __name__ == "__main__":
    m = make_vgg("D", num_classes=10, batch_norm=True)
    y = m(torch.randn(2,3,32,32))
    print(y.shape)  # (2,10)
