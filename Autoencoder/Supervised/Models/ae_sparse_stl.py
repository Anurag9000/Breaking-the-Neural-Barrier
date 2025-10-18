import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# -----------------------------------------------------------------------------
# AE_SPARSE_STL: Sparse Autoencoder using convolutional encoder/decoder.
# Follows AE_STL style, with optional L1 or KL sparsity penalty applied to the
# latent activations (encoder output). The penalty is computed externally in
# the runner to preserve modularity.
# -----------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class DeconvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.deconv(x)))

class AE_SPARSE_STL(nn.Module):
    """
    Sparse convolutional autoencoder.
    Args:
        in_channels: input channels (e.g., 3)
        width: base channels per encoder block
        depth: number of Conv blocks
        pool_after: indices where MaxPool2d is inserted
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
                dec_blocks.append(nn.ConvTranspose2d(ch_in, ch_in, 2, stride=2))
            ch_out = width if i > 1 else in_channels
            if i > 1:
                dec_blocks.append(DeconvBlock(ch_in, ch_out))
            else:
                dec_blocks.append(nn.ConvTranspose2d(ch_in, ch_out, 3, padding=1))
            ch_in = ch_out
        self.decoder = nn.Sequential(*dec_blocks)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
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


def sparsity_penalty(z: torch.Tensor, mode: str = 'l1', rho: float = 0.05) -> torch.Tensor:
    """Computes sparsity penalty on latent activations (mean over batch+spatial).
    mode='l1' adds |z| mean; mode='kl' uses KL(rho||rho_hat) with rho_hat=mean(sigmoid(z))."""
    if mode == 'l1':
        return z.abs().mean()
    elif mode == 'kl':
        rho_hat = torch.sigmoid(z).mean()
        rho = torch.tensor(rho, device=z.device, dtype=z.dtype)
        kl = rho * torch.log((rho / (rho_hat + 1e-8)) + 1e-8) + (1 - rho) * torch.log(((1 - rho) / (1 - rho_hat + 1e-8)) + 1e-8)
        return kl
    else:
        raise ValueError("mode must be 'l1' or 'kl'")


def ae_sparse_total_neurons(width: int, depth: int) -> int:
    return int(width * (depth + 1))
