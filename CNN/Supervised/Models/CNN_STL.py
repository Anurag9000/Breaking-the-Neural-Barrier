
import torch
import torch.nn as nn
from typing import List, Iterable


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class ConvNetSTL(nn.Module):
    """
    Simple STL-style ConvNet:
      - depth blocks of Conv-BN-ReLU with constant 'width' channels
      - optional 2x2 MaxPool after specified 1-based block indices (pooling_indices)
      - global average pool + linear head
    """
    def __init__(
        self,
        input_channels: int,
        num_classes: int,
        width: int = 64,
        depth: int = 4,
        pooling_indices: Iterable[int] = ()
    ):
        super().__init__()
        assert depth >= 1, "depth must be >= 1"
        assert width >= 1, "width must be >= 1"

        self.input_channels = int(input_channels)
        self.num_classes    = int(num_classes)
        self.width          = int(width)
        self.depth          = int(depth)
        # Normalize pooling indices as a sorted, unique list of positive ints
        self.pooling_indices = sorted({int(i) for i in pooling_indices if int(i) > 0})

        blocks = []
        in_ch = self.input_channels
        for i in range(1, self.depth + 1):
            blocks.append(ConvBNReLU(in_ch, self.width))
            in_ch = self.width
            # If this block index is in pooling_indices, add MaxPool
            if i in self.pooling_indices:
                blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.features = nn.Sequential(*blocks)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.width, self.num_classes)

        # lightweight init for linear head
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x


def stl_total_neurons(model: ConvNetSTL) -> int:
    """
    Total neurons for STL plot parity:
      sum of 'width' per conv block + head fan-in (= width)
      => width * depth + width = width * (depth + 1)
    """
    if not isinstance(model, ConvNetSTL):
        # best-effort: attempt to infer width/depth by scanning modules
        depth = 0
        width = None
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                width = m.out_channels  # assume constant width
                depth += 1
        if width is None:
            raise ValueError("Cannot infer width from model; pass ConvNetSTL instance.")
        return int(width * (depth + 1))
    return int(model.width * (model.depth + 1))
