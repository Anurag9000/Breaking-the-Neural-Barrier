import torch
import torch.nn as nn
from typing import List, Tuple

# -----------------------------------------------------------------------------
# AE_LOWRANK_STL: Convolutional AE with nuclear-norm regularization on the
# encoder output feature maps to encourage low-rank structure.
# The runner computes nuclear norm via torch.linalg.svd and adds to the loss.
# -----------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.c = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.b = nn.BatchNorm2d(out_ch)
        self.a = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.a(self.b(self.c(x)))

class DeconvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.c = nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.b = nn.BatchNorm2d(out_ch)
        self.a = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.a(self.b(self.c(x)))

class AE_LOWRANK_STL(nn.Module):
    """
    Low-rank-regularized convolutional autoencoder.

    Args:
        in_channels: input channels
        width: channels per block
        depth: number of encoder blocks
        pool_after: indices where MaxPool(2) is applied (mirrored in decoder)
    """
    def __init__(self, in_channels: int = 3, width: int = 64, depth: int = 4, pool_after: List[int] = None):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.width = width
        self.depth = depth
        self.pool_after = set(pool_after or [])

        # ---------------- Encoder ----------------
        enc = []
        ch_in = in_channels
        for i in range(1, depth + 1):
            enc.append(ConvBlock(ch_in, width))
            if i in self.pool_after:
                enc.append(nn.MaxPool2d(2, 2))
            ch_in = width
        self.encoder = nn.Sequential(*enc)

        # ---------------- Decoder ----------------
        dec = []
        ch_in = width
        for i in range(depth, 0, -1):
            if i in self.pool_after:
                dec.append(nn.ConvTranspose2d(ch_in, ch_in, 2, stride=2))
            ch_out = width if i > 1 else in_channels
            if i > 1:
                dec.append(DeconvBlock(ch_in, ch_out))
            else:
                dec.append(nn.ConvTranspose2d(ch_in, ch_out, 3, padding=1))
            ch_in = ch_out
        self.decoder = nn.Sequential(*dec)

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


def nuclear_norm_feature(z: torch.Tensor) -> torch.Tensor:
    """Compute average nuclear norm across batch for z of shape (B,C,H,W).
    We reshape per-sample to (C, H*W) and sum singular values. Returns mean.
    """
    B, C, H, W = z.shape
    total = 0.0
    for i in range(B):
        M = z[i].reshape(C, H * W)
        # Compute SVD; use full_matrices=False for economy
        # Add small epsilon for stability isn't necessary; torch handles it.
        s = torch.linalg.svdvals(M)
        total = total + s.sum()
    return total / max(B, 1)


def ae_lowrank_total_neurons(width: int, depth: int) -> int:
    return int(width * (depth + 1))
