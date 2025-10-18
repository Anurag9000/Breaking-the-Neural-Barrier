import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# -----------------------------------------------------------------------------
# AE_STL: Undercomplete convolutional autoencoder for CIFAR-like inputs
# Mirrors the style of CNN_STL.py: block-wise Conv-BN-ReLU with optional pools.
# Encoder depth = number of Conv blocks; width = channels per block.
# Decoder mirrors encoder using ConvTranspose2d. Final activation is identity;
# we train in normalized space using MSE loss in the runner.
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

class AE_STL(nn.Module):
    """
    Undercomplete Autoencoder with symmetric encoder/decoder.

    Args:
        in_channels: input channels (3 for RGB)
        width: base channels for each block
        depth: number of Conv blocks in encoder (decoder mirrors)
        pool_after: 1-based indices where a 2x2 MaxPool is inserted after the block
    """
    def __init__(self, in_channels: int = 3, width: int = 64, depth: int = 4, pool_after: List[int] = None):
        super().__init__()
        assert depth >= 1, "depth must be >= 1"
        self.in_channels = in_channels
        self.width = width
        self.depth = depth
        self.pool_after = set(pool_after or [])

        # ------------------------- Encoder -------------------------
        enc_blocks = []
        ch_in = in_channels
        for i in range(1, depth + 1):
            ch_out = width
            enc_blocks.append(ConvBlock(ch_in, ch_out))
            if i in self.pool_after:
                enc_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            ch_in = ch_out
        self.encoder = nn.Sequential(*enc_blocks)

        # Track pooling locations to mirror unpooling strides in decoder path.
        # We use ConvTranspose2d with stride=2 to mirror MaxPool positions.
        self._decoder_stride_positions = [i for i in range(1, depth + 1) if i in self.pool_after]

        # ------------------------- Decoder -------------------------
        dec_blocks = []
        ch_in = width
        # Mirror blocks in reverse order
        for i in range(depth, 0, -1):
            # If there was a pool after encoder block i, upsample here
            if i in self.pool_after:
                dec_blocks.append(nn.ConvTranspose2d(ch_in, ch_in, kernel_size=2, stride=2))
            # Mirror conv block
            ch_out = width if i > 1 else in_channels
            if i > 1:
                dec_blocks.append(DeconvBlock(ch_in, ch_out))
            else:
                # final layer to original channels, no BN/ReLU
                dec_blocks.append(nn.ConvTranspose2d(ch_in, ch_out, kernel_size=3, padding=1))
            ch_in = ch_out
        self.decoder = nn.Sequential(*dec_blocks)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
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


def ae_total_neurons(width: int, depth: int) -> int:
    """
    Mirrors the baseline STL scalar: width * (depth + 1). This keeps capacity plots
    consistent vs the supervised CNN baseline when comparing ADP variants later.
    """
    return int(width * (depth + 1))
