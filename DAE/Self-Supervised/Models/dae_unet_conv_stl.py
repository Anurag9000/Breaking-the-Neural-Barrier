import torch
import torch.nn as nn
from typing import List, Tuple


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle potential off‑by‑one due to pooling
        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x


class DAEUNetConv(nn.Module):
    """
    U‑Net style convolutional denoising autoencoder for CIFAR‑like images.

    Depth controls the number of encoder / decoder stages (downsampling
    operations). Width is the base channel count used at every level.

    Noise is injected in the training loop; this module just maps noisy inputs
    to reconstructions and returns the bottleneck activation as "latent".
    """

    def __init__(self, in_channels: int = 3, width: int = 64, depth: int = 4):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.width = width
        self.depth = depth

        enc_blocks: List[nn.Module] = []
        ch = in_channels
        for _ in range(depth):
            enc_blocks.append(ConvBlock(ch, width))
            ch = width
        self.enc_blocks = nn.ModuleList(enc_blocks)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = ConvBlock(width, width)

        up_blocks: List[nn.Module] = []
        for _ in range(depth):
            # Concatenate skip (width) with upsampled (width) -> 2*width in
            up_blocks.append(UpBlock(width, width))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.final_conv = nn.Conv2d(width, in_channels, kernel_size=1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        skips: List[torch.Tensor] = []
        for i, block in enumerate(self.enc_blocks):
            x = block(x)
            skips.append(x)
            if i < self.depth - 1:
                x = self.pool(x)
        x = self.bottleneck(x)
        return x, skips

    def decode(self, z: torch.Tensor, skips: List[torch.Tensor]) -> torch.Tensor:
        x = z
        for block, skip in zip(self.up_blocks, reversed(skips)):
            x = block(x, skip)
        x = self.final_conv(x)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z, skips = self.encode(x)
        x_rec = self.decode(z, skips)
        return x_rec, z


def dae_total_neurons(width: int, depth: int) -> int:
    """
    Simple scalar capacity metric used for plotting ADP behaviour.
    """
    return int(width * (depth + 1))

