import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=64, patch=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
    def forward(self, x):
        x = self.proj(x)
        B,C,H,W = x.shape
        x = x.flatten(2).transpose(1,2)  # B, HW, C
        return x, H, W

class PVTBlock(nn.Module):
    def __init__(self, dim, nhead, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(dim*mlp_ratio), dim), nn.Dropout(dropout),
        )
    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x

class PyramidVisionTransformer(nn.Module):
    """Simplified PVT for classification: 4 stages with downsampling between stages."""
    def __init__(self, num_classes=10, in_ch=3, dims=(64,128,256,512), depths=(2,2,2,2), heads=(2,4,8,8)):
        super().__init__()
        self.stages = nn.ModuleList()
        in_dim = in_ch
        patch = 4
        for s, (dim, depth, nhead) in enumerate(zip(dims, depths, heads)):
            if s == 0:
                self.patch = PatchEmbed(in_ch, dim, patch)
            else:
                self.down = nn.Conv2d(prev_dim, dim, kernel_size=2, stride=2)
                self.stages.append(self.down)
            blocks = nn.ModuleList([PVTBlock(dim, nhead) for _ in range(depth)])
            self.stages.append(blocks)
            prev_dim = dim
        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        x, H, W = self.patch(x)
        idx = 0
        while idx < len(self.stages):
            mod = self.stages[idx]
            if isinstance(mod, nn.Conv2d):
                # convert token seq back to feature map, downsample, then to tokens
                B,L,C = x.shape
                h = x.transpose(1,2).view(B, C, H, W)
                h = mod(h)
                B,C2,H,W = h.shape
                x = h.flatten(2).transpose(1,2)
                idx += 1
                mod = self.stages[idx]
            for blk in mod:  # blocks
                x = blk(x)
            idx += 1
        x = self.norm(x.mean(dim=1))
        return self.head(x)
