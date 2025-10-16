"""
AlexNet (CIFAR variant) in PyTorch — single-model supervised baseline.

This adapts the widely used CIFAR-10/100 AlexNet configuration:
- Input 32x32 RGB
- conv1: 64@5x5 s=1 p=2 -> pool 3x3 s=2
- conv2: 192@5x5 s=1 p=2 -> pool 3x3 s=2
- conv3: 384@3x3 s=1 p=1
- conv4: 256@3x3 s=1 p=1
- conv5: 256@3x3 s=1 p=1 -> pool 3x3 s=2
- head: flatten -> 4096 -> 4096 -> num_classes

Differences from the ImageNet AlexNet (224x224): smaller first kernel/stride, no LRN by default,
kept ReLU + Dropout in the classifier. Mirrors the straightforward, modular style of your STL CNN file.
"""
from __future__ import annotations
import torch
import torch.nn as nn

__all__ = ["AlexNet"]

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 32 -> 16

            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 16 -> 8

            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 8 -> 4
        )

        # For 32x32 input, after pools we have 256 x 4 x 4 = 4096
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

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
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    @staticmethod
    def param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    m = AlexNet(num_classes=10)
    y = m(torch.randn(2,3,32,32))
    print(y.shape)  # (2,10)
