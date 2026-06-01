import torch
import torch.nn as nn
from typing import List, Tuple

# -----------------------------------------------------------------------------
# AE_NOISE_STL: Convolutional autoencoder with internal regularization via
# dropout and feature Gaussian noise. Architecture mirrors AE_STL; we optionally
# insert (i) Dropout after activations and/or (ii) GaussianFeatureNoise modules
# inside encoder/decoder. This is distinct from DAE: here noise is *inside* the
# network, not only on inputs.
# -----------------------------------------------------------------------------

class GaussianFeatureNoise(nn.Module):
    def __init__(self, sigma: float = 0.0):
        super().__init__()
        self.sigma = float(sigma)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.sigma <= 0:
            return x
        return x + torch.randn_like(x) * self.sigma

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, p_drop: float = 0.0, feat_sigma: float = 0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity()
        self.noise = GaussianFeatureNoise(feat_sigma)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.conv(x)))
        x = self.drop(x)
        x = self.noise(x)
        return x

class DeconvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, p_drop: float = 0.0, feat_sigma: float = 0.0):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity()
        self.noise = GaussianFeatureNoise(feat_sigma)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn(self.deconv(x)))
        x = self.drop(x)
        x = self.noise(x)
        return x

class AE_NOISE_STL(nn.Module):
    """
    Dropout/Noise-regularized autoencoder.

    Args:
      in_channels: input channels (e.g., 3)
      width: channels per block (constant)
      depth: number of Conv blocks in encoder
      pool_after: 1-based indices where MaxPool2d(2) is inserted after the block
      p_drop_enc/dec: Dropout2d probs inside encoder/decoder blocks
      feat_sigma_enc/dec: Gaussian feature noise std inside encoder/decoder blocks
    """
    def __init__(self, in_channels: int = 3, width: int = 64, depth: int = 4,
                 pool_after: List[int] = None,
                 p_drop_enc: float = 0.0, p_drop_dec: float = 0.0,
                 feat_sigma_enc: float = 0.0, feat_sigma_dec: float = 0.0):
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
            enc_blocks.append(ConvBlock(ch_in, ch_out, p_drop=p_drop_enc, feat_sigma=feat_sigma_enc))
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
                dec_blocks.append(DeconvBlock(ch_in, ch_out, p_drop=p_drop_dec, feat_sigma=feat_sigma_dec))
            else:
                # final layer (no BN/ReLU/Dropout/Noise)
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z


def ae_noise_total_neurons(width: int, depth: int) -> int:
    return int(width * (depth + 1))
