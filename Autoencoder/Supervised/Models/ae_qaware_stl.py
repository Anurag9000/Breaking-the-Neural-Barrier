import torch
import torch.nn as nn
from typing import List, Tuple

# -----------------------------------------------------------------------------
# AE_QAWARE_STL: Quantization-aware training using straight-through estimator
# (STE). We quantize the latent feature z (and optionally intermediate features)
# to n_bits during training with a fake-quant module; inference returns dequant.
# -----------------------------------------------------------------------------

class STEQuant(nn.Module):
    def __init__(self, n_bits: int = 8, per_channel: bool = False):
        super().__init__()
        self.n_bits = int(n_bits)
        self.per_channel = bool(per_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Symmetric uniform quant in [-1,1] with STE; x is clipped to [-1,1]
        x_clamp = x.clamp(-1.0, 1.0)
        qlevels = 2 ** self.n_bits - 1
        if self.per_channel and x.dim() == 4:
            # scale per-channel (N,C,H,W) -> (C,1,1)
            denom = x_clamp.detach().abs().amax(dim=(0,2,3), keepdim=True).clamp_min(1e-6)
            x_norm = x_clamp / denom
            xq = torch.round((x_norm + 1) * qlevels / 2)  # [0, q]
            x_dq = (xq * 2 / qlevels - 1) * denom
        else:
            x_norm = x_clamp
            xq = torch.round((x_norm + 1) * qlevels / 2)
            x_dq = xq * 2 / qlevels - 1
        # Straight-through gradient
        return x_dq + (x - x_clamp).detach()

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DeconvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.deconv(x)))

class AE_QAWARE_STL(nn.Module):
    def __init__(self, in_channels: int = 3, width: int = 64, depth: int = 4,
                 pool_after: List[int] = None, n_bits: int = 8, per_channel: bool = False,
                 quant_everywhere: bool = False):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.width = width
        self.depth = depth
        self.pool_after = set(pool_after or [])
        self.quant_everywhere = bool(quant_everywhere)
        self.q = STEQuant(n_bits=n_bits, per_channel=per_channel)

        enc, ch_in = [], in_channels
        for i in range(1, depth+1):
            enc.append(ConvBlock(ch_in, width))
            if self.quant_everywhere:
                enc.append(self.q)
            if i in self.pool_after:
                enc.append(nn.MaxPool2d(2,2))
            ch_in = width
        self.encoder = nn.Sequential(*enc)

        dec, ch_in = [], width
        for i in range(depth, 0, -1):
            if i in self.pool_after:
                dec.append(nn.ConvTranspose2d(ch_in, ch_in, 2, stride=2))
            ch_out = width if i>1 else in_channels
            if i>1:
                dec.append(DeconvBlock(ch_in, ch_out))
                if self.quant_everywhere:
                    dec.append(self.q)
            else:
                dec.append(nn.ConvTranspose2d(ch_in, ch_out, 3, padding=1))
            ch_in = ch_out
        self.decoder = nn.Sequential(*dec)

        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        # Quantize latent representation
        zq = self.q(z)
        return zq

    def decode(self, zq: torch.Tensor) -> torch.Tensor:
        return self.decoder(zq)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        zq = self.encode(x)
        x_rec = self.decode(zq)
        return x_rec, zq


def ae_qaware_total_neurons(width: int, depth: int) -> int:
    return int(width * (depth + 1))
