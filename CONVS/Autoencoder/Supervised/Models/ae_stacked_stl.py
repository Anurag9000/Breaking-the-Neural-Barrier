import torch
import torch.nn as nn
from typing import List, Tuple

# -----------------------------------------------------------------------------
# AE_STACKED_STL: Deep stacked convolutional autoencoder (single model).
# This variant increases depth (many conv blocks) and uses occasional 1x1
# bottleneck projections for channel mixing. No layer-wise pretraining here; the
# runner trains end-to-end to match your single-model constraint.
# -----------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.c = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.b = nn.BatchNorm2d(out_ch)
        self.a = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.a(self.b(self.c(x)))

class Mix1x1(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.pw = nn.Conv2d(ch, ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.pw(x)))

class AE_STACKED_STL(nn.Module):
    """
    Deep stacked AE with many blocks and optional pooling.

    Args:
        in_channels: input channels
        width: channels per block
        depth: number of blocks in encoder
        pool_after: 1-based indices to add MaxPool(2) after a block
        mix_every: insert a 1x1 mixing block after every k blocks (0=off)
    """
    def __init__(self, in_channels: int = 3, width: int = 64, depth: int = 8,
                 pool_after: List[int] = None, mix_every: int = 0):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.width = width
        self.depth = depth
        self.pool_after = set(pool_after or [])
        self.mix_every = int(mix_every)

        # --------- Encoder ---------
        enc_ops = []
        ch_in = in_channels
        for i in range(1, depth + 1):
            enc_ops.append(ConvBlock(ch_in, width))
            if self.mix_every and (i % self.mix_every == 0):
                enc_ops.append(Mix1x1(width))
            if i in self.pool_after:
                enc_ops.append(nn.MaxPool2d(2, 2))
            ch_in = width
        self.encoder = nn.Sequential(*enc_ops)

        # --------- Decoder ---------
        dec_ops = []
        ch_in = width
        for i in range(depth, 0, -1):
            if i in self.pool_after:
                dec_ops.append(nn.ConvTranspose2d(ch_in, ch_in, 2, stride=2))
            # Mirror stacked conv + mix
            if i > 1:
                dec_ops.append(ConvBlock(ch_in, width))
                if self.mix_every and (i % self.mix_every == 0):
                    dec_ops.append(Mix1x1(width))
            else:
                dec_ops.append(nn.ConvTranspose2d(ch_in, in_channels, 3, padding=1))
            ch_in = width if i > 1 else in_channels
        self.decoder = nn.Sequential(*dec_ops)

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


def ae_stacked_total_neurons(width: int, depth: int) -> int:
    return int(width * (depth + 1))
