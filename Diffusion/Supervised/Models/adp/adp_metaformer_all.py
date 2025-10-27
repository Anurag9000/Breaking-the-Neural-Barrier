# adp_metaformer_all.py — Adaptive MetaFormer/EfficientFormer, 6 ADP algorithms

import torch
import torch.nn as nn


class DWConvMixer(nn.Module):
    def __init__(self, c, k=7):
        super().__init__()
        self.dw = nn.Conv2d(c, c, k, 1, k // 2, groups=c)

    def forward(self, x, hw):
        B, N, C = x.shape
        H, W = hw
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dw(x)
        return x.flatten(2).transpose(1, 2), (H, W)

    def widen(self, c):
        old = self.dw
        self.dw = nn.Conv2d(c, c, old.kernel_size[0], 1, old.padding[0], groups=c)


class AttnMixer(nn.Module):
    def __init__(self, c, h=4):
        super().__init__()
        self.h = h
        self.qkv = nn.Linear(c, 3 * c)
        self.proj = nn.Linear(c, c)

    def forward(self, x, hw):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.h, C // self.h)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = (q @ k.transpose(-2, -1)) / ((C // self.h) ** 0.5)
        y = (a.softmax(-1) @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(y), (hw[0], hw[1])

    def widen(self, c, h=None):
        self.qkv = _resize_linear(self.qkv, 3 * c, c)
        self.proj = _resize_linear(self.proj, c, c)
        self.h = h or self.h


class IdentityMixer(nn.Module):
    def forward(self, x, hw):
        return x, hw

    def widen(self, c):
        return


class MFBlock(nn.Module):
    def __init__(self, c, mixer="dw", heads=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(c)
        self.mixer = (
            DWConvMixer(c)
            if mixer == "dw"
            else (AttnMixer(c, heads) if mixer == "attn" else IdentityMixer())
        )
        self.ln2 = nn.LayerNorm(c)
        self.mlp = nn.Sequential(
            nn.Linear(c, 4 * c),
            nn.GELU(),
            nn.Linear(4 * c, c),
        )
        self.c = c
        self.mixer_name = mixer
        self.h = heads

    def forward(self, x, hw):
        x, _ = self.mixer(self.ln1(x), hw)
        x = x + self.mlp(self.ln2(x))
        return x, hw

    def widen(self, c, heads=None):
        self.ln1 = _resize_ln(self.ln1, c)
        self.mixer.widen(c)
        self.ln2 = _resize_ln(self.ln2, c)
        self.mlp[0] = _resize_linear(self.mlp[0], 4 * c, c)
        self.mlp[2] = _resize_linear(self.mlp[2], c, 4 * c)


class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, c=64, ps=4):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, c, ps, ps)
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        x = self.conv(x)
        B, C, H, W = x.shape
        return self.ln(x.flatten(2).transpose(1, 2)), (H, W)

    def widen(self, c):
        old = self.conv
        new = nn.Conv2d(
            old.in_channels,
            c,
            old.kernel_size,
            old.stride,
            old.padding,
            bias=(old.bias is not None),
        )
        with torch.no_grad():
            oc = min(c, old.out_channels)
            new.weight[:oc].copy_(old.weight[:oc])
            if old.bias is not None and new.bias is not None:
                new.bias[:oc].copy_(old.bias[:oc])
        self.conv = new
        self.ln = _resize_ln(self.ln, c)


class Down(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(4 * c)
        self.red = nn.Linear(4 * c, 2 * c)

    def forward(self, x, hw):
        B, N, C = x.shape
        H, W = hw
        x = x.view(B, H, W, C)
        x = torch.cat(
            [
                x[:, 0::2, 0::2, :],
                x[:, 1::2, 0::2, :],
                x[:, 0::2, 1::2, :],
                x[:, 1::2, 1::2, :],
            ],
            -1,
        ).view(B, -1, 4 * C)
        x = self.ln(x)
        x = self.red(x)
        return x, (H // 2, W // 2)

    def widen(self, newc):
        base = newc // 2
        self.ln = _resize_ln(self.ln, 4 * base)
        self.red = _resize_linear(self.red, newc, 4 * base)


class AdaptiveMetaFormer(nn.Module):
    def __init__(
        self,
        num_classes=10,
        in_ch=3,
        patch=4,
        dims=[64, 128, 256],
        depths=[2, 2, 2],
        mixer="dw",
        heads=[2, 4, 8],
    ):
        super().__init__()
        self.patch = PatchEmbed(in_ch, dims[0], patch)
        self.blocks = nn.ModuleList()
        self.downs = nn.ModuleList()

        for i in range(len(dims)):
            for _ in range(depths[i]):
                self.blocks.append(MFBlock(dims[i], mixer, heads[i]))
            if i < len(dims) - 1:
                self.downs.append(Down(dims[i]))

        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

        self.dims = list(dims)
        self.depths = list(depths)
        self.mixer = mixer
        self.heads = list(heads)

    def forward(self, x):
        x, hw = self.patch(x)
        bi = 0
        for i in range(len(self.dims)):
            for _ in range(self.depths[i]):
                x, hw = self.blocks[bi](x, hw)
                bi += 1
            if i < len(self.dims) - 1:
                x, hw = self.downs[i](x, hw)
        x = self.norm(x).mean(1)
        return self.head(x)

    # ADP primitives
    def append_depth(self, stage=None):
        si = (len(self.dims) - 1) if stage is None else stage
        bi = sum(self.depths[:si]) + self.depths[si]
        self.blocks.insert(
            bi, MFBlock(self.dims[si], self.mixer, self.heads[si])
        )
        self.depths[si] += 1

    def widen_all(self, ex_k=16):
        self.patch.widen(self.dims[0] + ex_k)
        self.dims[0] += ex_k

        # propagate widths per stage
        idx = 0
        for i in range(len(self.dims)):
            nd = self.dims[i]
            nh = None
            if (nd % self.heads[i]) != 0 and (nd % (self.heads[i] + 1)) == 0:
                nh = self.heads[i] + 1
                self.heads[i] = nh
            for _ in range(self.depths[i]):
                self.blocks[idx].widen(nd, nh)
                idx += 1
            if i < len(self.dims) - 1:
                self.downs[i].widen(self.dims[i + 1])

        self.norm = _resize_ln(self.norm, self.dims[-1])
        self.head = _resize_linear(
            self.head, self.head.out_features, self.dims[-1]
        )

    def num_neurons(self):
        return int(sum(self.dims))


# TrainCfg/SearchCfg + six ADP algos: copy verbatim from RetNet; ALGO_MAP identical.
ALGO_MAP = {}
