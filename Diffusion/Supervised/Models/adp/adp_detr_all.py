# adp_detr_all.py — Adaptive DETR (simplified), 6 ADP algorithms
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F


# helpers as before


class PosEnc(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.row = nn.Parameter(torch.randn(1, 1, d // 2))
        self.col = nn.Parameter(torch.randn(1, 1, d // 2))

    def forward(self, B, N):
        return torch.cat(
            [self.row.expand(B, N, -1), self.col.expand(B, N, -1)],
            dim=-1,
        )


class EncoderLayer(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.sa = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.ReLU(),
            nn.Linear(4 * d, d),
        )

    def forward(self, x):
        x = x + self.sa(self.ln1(x), self.ln1(x), self.ln1(x))[0]
        x = x + self.mlp(self.ln2(x))
        return x

    def widen(self, d, h=None):
        return


class DecoderLayer(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.sa = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ca = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln3 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 4 * d),
            nn.ReLU(),
            nn.Linear(4 * d, d),
        )

    def forward(self, t, mem):
        t = t + self.sa(self.ln1(t), self.ln1(t), self.ln1(t))[0]
        t = t + self.ca(self.ln2(t), mem, mem)[0]
        t = t + self.mlp(self.ln3(t))
        return t

    def widen(self, d, h=None):
        return


class Backbone(nn.Module):
    def __init__(self, in_ch=3, d=256, ps=4):
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


class AdaptiveDETR(nn.Module):
    def __init__(
        self,
        num_classes=10,
        embed=256,
        enc_layers=3,
        dec_layers=3,
        heads=8,
        queries=50,
    ):
        super().__init__()
        self.bb = Backbone(3, embed, 4)
        self.pos = PosEnc(embed)
        self.encoder = nn.ModuleList(
            [EncoderLayer(embed, heads) for _ in range(enc_layers)]
        )
        self.decoder = nn.ModuleList(
            [DecoderLayer(embed, heads) for _ in range(dec_layers)]
        )
        self.query = nn.Parameter(torch.randn(1, queries, embed) * 0.02)
        self.norm = nn.LayerNorm(embed)
        self.bbox = nn.Linear(embed, 4)
        self.cls = nn.Linear(embed, 1)  # binary head (object vs background)
        self.embed = embed
        self.enc_layers = enc_layers
        self.dec_layers = dec_layers
        self.heads = heads

    def forward(self, x):
        mem = self.bb(x)
        B, N, D = mem.shape
        mem = mem + self.pos(B, N)
        z = self.query.expand(B, -1, -1)
        for l in self.encoder:
            mem = l(mem)
        for l in self.decoder:
            z = l(z, mem)
        z = self.norm(z)
        return self.cls(z).squeeze(-1), self.bbox(z)

    # ADP primitives (decoder depth & width)
    def append_depth(self):
        self.decoder.append(DecoderLayer(self.embed, self.heads))
        self.dec_layers += 1

    def widen_all(self, ex_k=32):
        self.embed += ex_k
        self.bb.widen(self.embed)

        # rebuild heads and normalization
        self.norm = _resize_ln(self.norm, self.embed)
        self.bbox = _resize_linear(self.bbox, 4, self.embed)
        self.cls = _resize_linear(self.cls, 1, self.embed)

    def num_neurons(self):
        return int(self.embed)


# TrainCfg/SearchCfg + six ADP algorithms: copy from RetNet;
# loss combines BCE + L1 on bbox (simplified); ES uses val loss.
ALGO_MAP = {}
