import torch
import torch.nn as nn
from typing import List, Tuple

from .dae_gaussian_conv_stl import dae_total_neurons


class DilatedConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 2):
        super().__init__()
        padding = dilation
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DilatedDeconvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 2):
        super().__init__()
        padding = dilation
        self.deconv = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.deconv(x)))


class DAEDilatedConv(nn.Module):
    """
    Dilated convolutional denoising autoencoder for CIFAR-like images.

    Same layout as the Gaussian Conv DAE, but Conv/ConvTranspose use a fixed
    dilation factor to increase receptive field without additional pooling.
    """

    def __init__(
        self,
        in_channels: int = 3,
        width: int = 64,
        depth: int = 4,
        dilation: int = 2,
        pool_after: List[int] | None = None,
    ):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.width = width
        self.depth = depth
        self.dilation = dilation
        self.pool_after = set(pool_after or [])

        enc_blocks: List[nn.Module] = []
        ch_in = in_channels
        for i in range(1, depth + 1):
            ch_out = width
            enc_blocks.append(DilatedConvBlock(ch_in, ch_out, dilation=dilation))
            if i in self.pool_after:
                enc_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            ch_in = ch_out
        self.encoder = nn.Sequential(*enc_blocks)

        dec_blocks: List[nn.Module] = []
        ch_in = width
        for i in range(depth, 0, -1):
            if i in self.pool_after:
                dec_blocks.append(nn.ConvTranspose2d(ch_in, ch_in, kernel_size=2, stride=2))
            ch_out = width if i > 1 else in_channels
            if i > 1:
                dec_blocks.append(DilatedDeconvBlock(ch_in, ch_out, dilation=dilation))
            else:
                dec_blocks.append(
                    nn.ConvTranspose2d(ch_in, ch_out, kernel_size=3, padding=dilation, dilation=dilation)
                )
            ch_in = ch_out
        self.decoder = nn.Sequential(*dec_blocks)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z


__all__ = ["DAEDilatedConv", "dae_total_neurons"]

