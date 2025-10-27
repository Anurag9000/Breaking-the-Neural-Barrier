# adp_fnet_all.py — Adaptive FNet (no attention), 6 ADP algorithms
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F


# helpers as before


class FNetMix(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d

    def forward(self, x):
        # 2D FFT on (tokens, channels)
        return torch.fft.rfft(torch.fft.rfft(x, dim=1), dim=2).real

    def widen(self, d):
        self.d = d


class FNetBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.mix = FNetMix(d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.GELU(),
            nn.Linear(4 * d, d),
        )

    def forward(self, x):
        x = x + self.mix(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def widen(self, d):
        self.ln1 = _resize_ln(self.ln1, d)
        self.mix.widen(d)
        self.ln2 = _resize_ln(self.ln2, d)
        self.mlp[0] = _resize_linear(self.mlp[0], 4 * d, d)
        self.mlp[2] = _resize_linear(self.mlp[2], d, 4 * d)


class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, d=64, ps=4):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, d, ps, ps)
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        x = self.conv(x)
        B, C, H, W = x.shape
        return self.ln(x.flatten(2).transpose(1, 2))

    def widen(self, d):
        old = self.conv
        new = nn.Conv2d(
            old.in_channels,
            d,
            old.kernel_size,
            old.stride,
            old.padding,
            bias=(old.bias is not None),
        )
        with torch.no_grad():
            oc = min(d, old.out_channels)
            new.weight[:oc].copy_(old.weight[:oc])
            if old.bias is not None and new.bias is not None:
                new.bias[:oc].copy_(old.bias[:oc])
        self.conv = new
        self.ln = _resize_ln(self.ln, d)


class AdaptiveFNet(nn.Module):
    def __init__(self, num_classes=10, in_ch=3, patch=4, d=128, depth=6):
        super().__init__()
        self.pe = PatchEmbed(in_ch, d, patch)
        self.blocks = nn.ModuleList([FNetBlock(d) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, num_classes)
        self.d = d
        self.depth = depth

    def forward(self, x):
        x = self.pe(x)
        for b in self.blocks:
            x = b(x)
        x = self.norm(x).mean(1)
        return self.head(x)

    # ADP primitives
    def append_depth(self):
        self.blocks.append(FNetBlock(self.d))
        self.depth += 1

    def widen_all(self, ex_k=16):
        self.d += ex_k
        self.pe.widen(self.d)
        for b in self.blocks:
            b.widen(self.d)
        self.norm = _resize_ln(self.norm, self.d)
        self.head = _resize_linear(self.head, self.head.out_features, self.d)

    def num_neurons(self):
        return int(self.d)


# TrainCfg/SearchCfg + six ADP algorithms: copy from RetNet (replace model class references)
ALGO_MAP = {}
