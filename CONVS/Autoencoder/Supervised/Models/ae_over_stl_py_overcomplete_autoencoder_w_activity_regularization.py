import torch
import torch.nn as nn
from typing import List, Tuple

# -----------------------------------------------------------------------------
# AE_OVER_STL: Overcomplete convolutional autoencoder with an explicit
# 1x1 bottleneck expansion and symmetric decoder. Designed to be overcomplete
# when latent_mult > 1. Uses the same Conv-BN-ReLU style as AE_STL.
# The runner adds activity L1 regularization on the latent tensor.
# -----------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class DeconvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.deconv(x)))

class AE_OVER_STL(nn.Module):
    """
    Overcomplete Autoencoder with convolutional encoder/decoder.

    Args:
        in_channels: input channels (3 for RGB)
        width: channels per encoder block
        depth: number of Conv blocks in encoder
        latent_mult: multiplicative factor for bottleneck channels (>1 => overcomplete)
        pool_after: 1-based indices where a 2x2 MaxPool follows the block
    """
    def __init__(self, in_channels: int = 3, width: int = 64, depth: int = 4,
                 latent_mult: float = 2.0, pool_after: List[int] = None):
        super().__init__()
        assert depth >= 1
        assert latent_mult >= 1.0
        self.in_channels = in_channels
        self.width = width
        self.depth = depth
        self.latent_mult = float(latent_mult)
        self.pool_after = set(pool_after or [])

        # ------------------------- Encoder -------------------------
        enc_blocks = []
        ch_in = in_channels
        for i in range(1, depth + 1):
            ch_out = width
            enc_blocks.append(ConvBlock(ch_in, ch_out))
            if i in self.pool_after:
                enc_blocks.append(nn.MaxPool2d(2, 2))
            ch_in = ch_out
        self.encoder = nn.Sequential(*enc_blocks)

        # Bottleneck expansion (overcomplete latent)
        latent_ch = int(round(width * self.latent_mult))
        self.bottleneck = nn.Conv2d(width, latent_ch, kernel_size=1, bias=True)
        self.bottleneck_act = nn.ReLU(inplace=True)

        # ------------------------- Decoder -------------------------
        self.unbottleneck = nn.Conv2d(latent_ch, width, kernel_size=1, bias=True)

        dec_blocks = []
        ch_in = width
        for i in range(depth, 0, -1):
            if i in self.pool_after:
                dec_blocks.append(nn.ConvTranspose2d(ch_in, ch_in, kernel_size=2, stride=2))
            ch_out = width if i > 1 else in_channels
            if i > 1:
                dec_blocks.append(DeconvBlock(ch_in, ch_out))
            else:
                dec_blocks.append(nn.ConvTranspose2d(ch_in, ch_out, kernel_size=3, padding=1))
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
        z = self.encoder(x)
        z = self.bottleneck_act(self.bottleneck(z))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        y = self.unbottleneck(z)
        y = self.decoder(y)
        return y

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z


def ae_over_total_neurons(width: int, depth: int, latent_mult: float) -> int:
    """Capacity proxy consistent with STL style."""
    return int(width * (depth + 1) + (width * max(0.0, latent_mult - 1.0)))
