import torch
import torch.nn as nn
from typing import List, Tuple

# -----------------------------------------------------------------------------
# AE_DILATED_STL: Autoencoder using atrous (dilated) convolutions in the encoder
# and decoder to enlarge receptive fields without pooling. You can still choose
# to pool after some blocks; dilation and pooling are orthogonal options.
# Dilation schedule can be set per block (e.g., 1,2,4,1).
# -----------------------------------------------------------------------------

class DilatedConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__()
        padding = dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class DilatedDeconvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__()
        padding = dilation
        # Mirror with ConvTranspose2d; keep dilation for symmetry of local mapping
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.deconv(x)))

class AE_DILATED_STL(nn.Module):
    """
    Dilated/Atrous convolutional autoencoder.

    Args:
        in_channels: input channels
        width: channels per conv block
        depth: number of conv blocks in encoder
        dilations: list of dilation factors per encoder block (len==depth) or single int
        pool_after: 1-based indices to apply 2x2 MaxPool after a block
    """
    def __init__(self, in_channels: int = 3, width: int = 64, depth: int = 4,
                 dilations: List[int] | int = 1, pool_after: List[int] = None):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.width = width
        self.depth = depth
        self.pool_after = set(pool_after or [])
        if isinstance(dilations, int):
            self.dilations = [int(dilations)] * depth
        else:
            assert len(dilations) == depth, 'dilations length must equal depth'
            self.dilations = [int(d) for d in dilations]

        # ---------------- Encoder ----------------
        enc_blocks = []
        ch_in = in_channels
        for i in range(1, depth + 1):
            d = self.dilations[i-1]
            ch_out = width
            enc_blocks.append(DilatedConvBlock(ch_in, ch_out, dilation=d))
            if i in self.pool_after:
                enc_blocks.append(nn.MaxPool2d(2, 2))
            ch_in = ch_out
        self.encoder = nn.Sequential(*enc_blocks)

        # ---------------- Decoder ----------------
        dec_blocks = []
        ch_in = width
        for i in range(depth, 0, -1):
            if i in self.pool_after:
                dec_blocks.append(nn.ConvTranspose2d(ch_in, ch_in, kernel_size=2, stride=2))
            d = self.dilations[i-1]
            ch_out = width if i > 1 else in_channels
            if i > 1:
                dec_blocks.append(DilatedDeconvBlock(ch_in, ch_out, dilation=d))
            else:
                # final projection without BN/ReLU; keep dilation for symmetry
                padding = d
                dec_blocks.append(nn.ConvTranspose2d(ch_in, ch_out, kernel_size=3, padding=padding, dilation=d))
            ch_in = ch_out
        self.decoder = nn.Sequential(*dec_blocks)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)
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


def ae_dilated_total_neurons(width: int, depth: int) -> int:
    return int(width * (depth + 1))
