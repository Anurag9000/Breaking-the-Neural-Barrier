# adp_mixer_all.py — Adaptive MLP-Mixer, 6 ADP algorithms
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F


# helpers as before


class MixerBlock(nn.Module):
    def __init__(self, n_tokens, d, r_t=0.5, r_c=4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.mlp_t = nn.Sequential(
            nn.Linear(n_tokens, int(n_tokens * r_t)),
            nn.GELU(),
            nn.Linear(int(n_tokens * r_t), n_tokens),
        )
        self.ln2 = nn.LayerNorm(d)
        self.mlp_c = nn.Sequential(
            nn.Linear(d, int(d * r_c)),
            nn.GELU(),
            nn.Linear(int(d * r_c), d),
        )
        self.n_tokens = n_tokens
        self.d = d

    def forward(self, x):
        # x: (B, N, D)
        y = self.ln1(x).transpose(1, 2)
        y = self.mlp_t(y).transpose(1, 2)
        x = x + y
        x = x + self.mlp_c(self.ln2(x))
        return x

    def widen(self, d):
        self.ln1 = _resize_ln(self.ln1, d)
        self.ln2 = _resize_ln(self.ln2, d)
        self.mlp_c[0] = _resize_linear(self.mlp_c[0], int(d * 4.0), d)
        self.mlp_c[2] = _resize_linear(self.mlp_c[2], d, int(d * 4.0))
        self.d = d


class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, d=64, ps=4):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, d, ps, ps)
        self.ln = nn.LayerNorm(d)

    def forward(self, x):
        x = self.conv(x)
        B, C, H, W = x.shape
        return self.ln(x.flatten(2).transpose(1, 2)), H * W

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


class AdaptiveMixer(nn.Module):
    def __init__(self, num_classes=10, in_ch=3, patch=4, d=128, depth=8):
        super().__init__()
        self.pe = PatchEmbed(in_ch, d, patch)
        self.blocks = nn.ModuleList()
        self.d = d
        self.depth = depth
        # n_tokens depends on input (e.g., CIFAR-10 32×32 → (32 / patch)^2)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, num_classes)

    def forward(self, x):
        x, n = self.pe(x)
        if len(self.blocks) == 0:
            for _ in range(self.depth):
                self.blocks.append(MixerBlock(n, self.d))
        for b in self.blocks:
            x = b(x)
        x = self.norm(x).mean(1)
        return self.head(x)

    # ADP primitives
    def append_depth(self):
        n_tokens = self.blocks[0].n_tokens if len(self.blocks) > 0 else 64
        self.blocks.append(MixerBlock(n_tokens, self.d))
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


# TrainCfg/SearchCfg + six ADP algorithms: copy from RetNet
ALGO_MAP = {}
