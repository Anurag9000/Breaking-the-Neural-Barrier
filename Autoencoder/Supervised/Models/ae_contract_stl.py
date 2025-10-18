import torch
import torch.nn as nn
from typing import List, Tuple

# -----------------------------------------------------------------------------
# AE_CONTRACT_STL: Contractive Autoencoder using convolutional blocks.
# Contractive penalty ≈ E_v[|| J_enc(x) v ||^2], where J_enc is Jacobian of
# encoder output wrt input, and v is a Rademacher noise vector (Hutchinson).
# Architecture mirrors AE_STL (Conv-BN-ReLU + optional pooling) and symmetric
# ConvTranspose2d decoder. The runner computes and adds the penalty.
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

class AE_CONTRACT_STL(nn.Module):
    """
    Contractive convolutional autoencoder with symmetric decoder.

    Args:
      in_channels: input channels (3 for RGB)
      width: channels per block
      depth: number of Conv blocks in encoder
      pool_after: 1-based indices where 2x2 MaxPool is inserted after the block
    """
    def __init__(self, in_channels: int = 3, width: int = 64, depth: int = 4, pool_after: List[int] = None):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.width = width
        self.depth = depth
        self.pool_after = set(pool_after or [])

        # --------------- Encoder ---------------
        enc_blocks = []
        ch_in = in_channels
        for i in range(1, depth + 1):
            ch_out = width
            enc_blocks.append(ConvBlock(ch_in, ch_out))
            if i in self.pool_after:
                enc_blocks.append(nn.MaxPool2d(2, 2))
            ch_in = ch_out
        self.encoder = nn.Sequential(*enc_blocks)

        # --------------- Decoder ---------------
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z


def contractive_penalty_hutchinson(encoder: nn.Module, x: torch.Tensor, iters: int = 1) -> torch.Tensor:
    """Hutchinson estimator of ||J||_F^2 for encoder output wrt input x.
    Returns average over `iters` random Rademacher probes.
    """
    penalty = 0.0
    for _ in range(max(1, iters)):
        x_req = x.detach().requires_grad_(True)
        z = encoder(x_req)
        # Rademacher vector v
        v = torch.randint_like(z, low=0, high=2, dtype=torch.int8).float() * 2 - 1  # {-1, +1}
        s = (z * v).sum()
        (g,) = torch.autograd.grad(s, x_req, create_graph=True)
        penalty = penalty + g.pow(2).sum(dim=(1, 2, 3))  # per-sample
    penalty = penalty / float(max(1, iters))
    return penalty.mean()


def ae_contract_total_neurons(width: int, depth: int) -> int:
    return int(width * (depth + 1))
