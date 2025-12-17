import torch
import torch.nn as nn
from typing import Tuple


class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int):
        super().__init__()
        kernel_size = 3
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DeconvTCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int):
        super().__init__()
        kernel_size = 3
        padding = dilation * (kernel_size - 1) // 2
        self.deconv = nn.ConvTranspose1d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.deconv(x)))


class DAETCNSeq(nn.Module):
    """
    Temporal denoising autoencoder based on 1D TCN-style convolutions.

    - Input: sequences of shape (B, C, L).
    - Noise: Gaussian noise is added in the training loop.
    - Width controls the channel dimensionality of hidden layers.
    - Depth controls the number of temporal blocks (dilations grow with depth).
    """

    def __init__(self, in_channels: int = 1, width: int = 64, depth: int = 4):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.width = width
        self.depth = depth

        enc_blocks = []
        ch_in = in_channels
        for i in range(depth):
            dilation = 2**i
            ch_out = width
            enc_blocks.append(TCNBlock(ch_in, ch_out, dilation=dilation))
            ch_in = ch_out
        self.encoder = nn.Sequential(*enc_blocks)

        dec_blocks = []
        ch_in = width
        for i in reversed(range(depth)):
            dilation = 2**i
            ch_out = width if i > 0 else in_channels
            if i > 0:
                dec_blocks.append(DeconvTCNBlock(ch_in, ch_out, dilation=dilation))
            else:
                dec_blocks.append(
                    nn.ConvTranspose1d(
                        ch_in,
                        ch_out,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                    )
                )
            ch_in = ch_out
        self.decoder = nn.Sequential(*dec_blocks)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm1d):
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


def tcn_total_neurons(width: int, depth: int) -> int:
    return int(width * (depth + 1))

