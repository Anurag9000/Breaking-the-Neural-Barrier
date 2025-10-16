"""
SqueezeNet (CIFAR) — single-model supervised.

Implements SqueezeNet v1.1-style backbone with Fire modules, adapted for 32×32 inputs.
Reference: Iandola et al., "SqueezeNet" (2016). The v1.1 topology shortens early layers and
keeps 1×1-heavy Fire modules with occasional 3×3 expand filters.

CIFAR-friendly changes:
  • Stem: 3×3 conv stride=1 (no early downsampling), then maxpool 3×3 s=2.
  • Fire stacks roughly mirroring v1.1 but tuned for small inputs.
  • Downsampling via maxpool placed after certain Fire modules.
  • Classifier: Dropout → 1×1 conv(num_classes) → ReLU → GAP.

This mirrors your modular style: Fire block, factory, and param_count helper.
"""
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn

__all__ = ["SqueezeNetCIFAR", "make_squeezenet_cifar", "Fire"]

class Fire(nn.Module):
    def __init__(self, in_ch: int, squeeze_ch: int, expand_ch: int):
        super().__init__()
        # Squeeze 1×1
        self.squeeze = nn.Conv2d(in_ch, squeeze_ch, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        # Expand: 1×1 and 3×3 (half-half)
        e1 = expand_ch // 2
        e3 = expand_ch - e1
        self.expand1x1 = nn.Conv2d(squeeze_ch, e1, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_ch, e3, kernel_size=3, padding=1)
        self.expand_activation = nn.ReLU(inplace=True)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.squeeze_activation(self.squeeze(x))
        return self.expand_activation(torch.cat([
            self.expand1x1(x),
            self.expand3x3(x)
        ], dim=1))

class SqueezeNetCIFAR(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3, version: str = "1.1"):
        super().__init__()
        assert version in {"1.0", "1.1"}
        self.version = version

        if version == "1.1":
            # v1.1 style
            features: List[nn.Module] = [
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64),
                Fire(64, 16, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 32, 128),
                Fire(128, 32, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 48, 192),
                Fire(192, 48, 192),
                Fire(192, 64, 256),
                Fire(256, 64, 256),
            ]
        else:
            # v1.0 (for completeness; still CIFAR-adapted by using stride=1 at stem)
            features = [
                nn.Conv2d(in_channels, 96, kernel_size=7, stride=1, padding=3),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64),
                Fire(128, 16, 64),
                Fire(128, 32, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128),
                Fire(256, 48, 192),
                Fire(384, 48, 192),
                Fire(384, 64, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256),
            ]

        self.features = nn.Sequential(*features)

        # Classifier: Dropout → 1×1 conv(num_classes) → ReLU → GAP
        self.dropout = nn.Dropout(p=0.5)
        # Determine final channels from features tail
        if version == "1.1":
            final_ch = 256
        else:
            final_ch = 512
        self.classifier_conv = nn.Conv2d(final_ch, num_classes, kernel_size=1)
        self.classifier_act = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        nn.init.normal_(self.classifier_conv.weight, mean=0.0, std=0.01)
        if self.classifier_conv.bias is not None:
            nn.init.zeros_(self.classifier_conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.dropout(x)
        x = self.classifier_act(self.classifier_conv(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    @staticmethod
    def param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())


def make_squeezenet_cifar(num_classes: int = 10, in_channels: int = 3, version: str = "1.1") -> SqueezeNetCIFAR:
    return SqueezeNetCIFAR(num_classes=num_classes, in_channels=in_channels, version=version)

if __name__ == "__main__":
    for v in ["1.1", "1.0"]:
        m = make_squeezenet_cifar(num_classes=10, version=v)
        y = m(torch.randn(2,3,32,32))
        print(v, y.shape)
