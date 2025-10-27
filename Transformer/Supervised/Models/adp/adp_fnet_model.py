import torch, torch.nn as nn
from dataclasses import dataclass

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=64, patch=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        self.norm = nn.BatchNorm2d(embed_dim)
    def forward(self, x):
        x = self.norm(self.proj(x))
        B,C,H,W = x.shape
        return x.flatten(2).transpose(1,2)

# FNet block: Fourier mixing along token dimension (no attention)
class FNetBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_ratio*dim), nn.GELU(), nn.Dropout(drop),
            nn.Linear(mlp_ratio*dim, dim), nn.Dropout(drop)
        )
    def forward(self, x):
        # x: B,N,D
        h = self.norm1(x)
        # real FFT along tokens
        y = torch.fft.rfft(h, dim=1).real
        # match length if rfft halves
        if y.size(1) != x.size(1):
            # simple interpolate back to N
            y = torch.nn.functional.interpolate(y.transpose(1,2), size=x.size(1), mode='linear', align_corners=False).transpose(1,2)
        x = x + y
        x = x + self.ff(x)
        return x

class FNetTiny(nn.Module):
    def __init__(self, num_classes=10, embed_dim=64, depth=4, patch=4):
        super().__init__()
        self.embed_dim = embed_dim; self.patch=patch
        self.tokenizer = PatchEmbed(3, embed_dim, patch)
        self.blocks = nn.ModuleList([FNetBlock(embed_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    def add_block(self):
        self.blocks.append(FNetBlock(self.embed_dim))
    def widen_all(self, ex_k):
        new_dim = self.embed_dim + ex_k
        new_tok = PatchEmbed(3, new_dim, self.patch)
        copy_conv2d(self.tokenizer.proj, new_tok.proj)
        copy_bn2d(self.tokenizer.norm, new_tok.norm)
        self.tokenizer = new_tok
        new_blocks = nn.ModuleList()
        for b in self.blocks:
            nb = FNetBlock(new_dim)
            transplant_block_fnet(b, nb)
            new_blocks.append(nb)
        self.blocks = new_blocks
        self.norm = nn.LayerNorm(new_dim)
        new_head = nn.Linear(new_dim, self.head.out_features)
        copy_linear_overlap(self.head, new_head)
        self.head = new_head
        self.embed_dim = new_dim
    def forward(self, x):
        x = self.tokenizer(x)
        for b in self.blocks:
            x = b(x)
        x = self.norm(x).mean(1)
        return self.head(x)

# overlap-copy helpers reused from RetNet
from ADP_RetNet_model import copy_conv2d, copy_bn2d, copy_linear_overlap, TrainCfg, ADPCfg, evaluate, train_inner

def transplant_block_fnet(old: FNetBlock, new: FNetBlock):
    for (ol, nl) in zip(old.ff, new.ff):
        if isinstance(ol, nn.Linear) and isinstance(nl, nn.Linear):
            copy_linear_overlap(ol, nl)

def build_fnet(num_classes=10, init_width=64, init_depth=2, patch=4):
    return FNetTiny(num_classes=num_classes, embed_dim=init_width, depth=init_depth, patch=patch)
