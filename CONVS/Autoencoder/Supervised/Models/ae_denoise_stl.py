import torch
import torch.nn as nn
from typing import List, Tuple

# -----------------------------------------------------------------------------
# AE_DENOISE_STL: Convolutional denoising autoencoder (Gaussian noise handled
# in the runner). Architecture mirrors AE_STL: Conv-BN-ReLU blocks with optional
# pooling in the encoder and symmetric ConvTranspose2d decoder.
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

class AE_DENOISE_STL(nn.Module):
    """
    Denoising Autoencoder with symmetric encoder/decoder.

    Args:
      in_channels: input channels (e.g., 3)
      width: channels per block
      depth: number of Conv blocks in the encoder
      pool_after: 1-based indices for 2x2 MaxPool after a block (mirrored as upsampling)
    """
    def __init__(self, in_channels: int = 3, width: int = 64, depth: int = 4, pool_after: List[int] = None):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.width = width
        self.depth = depth
        self.pool_after = set(pool_after or [])

        # ----------------- Encoder -----------------
        enc_blocks = []
        ch_in = in_channels
        for i in range(1, depth + 1):
            ch_out = width
            enc_blocks.append(ConvBlock(ch_in, ch_out))
            if i in self.pool_after:
                enc_blocks.append(nn.MaxPool2d(2, 2))
            ch_in = ch_out
        self.encoder = nn.Sequential(*enc_blocks)

        # ----------------- Decoder -----------------
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
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x_noisy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward expects noisy input and returns reconstruction of the clean signal.
        The runner will provide noisy input and compute MSE to the clean target.
        """
        z = self.encode(x_noisy)
        x_rec = self.decode(z)
        return x_rec, z


def ae_denoise_total_neurons(width: int, depth: int) -> int:
    return int(width * (depth + 1))
