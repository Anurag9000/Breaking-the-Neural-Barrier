import torch
import torch.nn as nn
from typing import Tuple


class DIPConvNet(nn.Module):
    def __init__(self, in_channels: int = 3, width: int = 64, depth: int = 5):
        super().__init__()
        self.in_channels = in_channels
        self.width = width
        self.depth = depth

        layers = []
        ch_in = in_channels
        ch = width
        for i in range(depth - 1):
            layers.append(nn.Conv2d(ch_in, ch, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            ch_in = ch
        layers.append(nn.Conv2d(ch_in, in_channels, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class DAEDIPConv(nn.Module):
    """
    Deep-image-prior-style Conv "DAE".

    - Input: fixed noise z; network outputs reconstruction x_rec.
    - In STL runner we optimise per image; here we expose a batch interface.
    """

    def __init__(self, in_channels: int = 3, width: int = 64, depth: int = 5):
        super().__init__()
        self.in_channels = in_channels
        self.width = width
        self.depth = depth
        self.net = DIPConvNet(in_channels, width, depth)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_rec = self.net(z)
        return x_rec, z


def dip_total_neurons(width: int, depth: int) -> int:
    return int(width * depth * 32)

