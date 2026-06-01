import torch
import torch.nn as nn

class ConvNeXtStem(nn.Module):
    def __init__(self, in_ch=3, dims=(64,128)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, dims[0], 4, 4), nn.LayerNorm(dims[0], eps=1e-6),
            nn.Conv2d(dims[0], dims[0], 7, 1, 3, groups=dims[0]), nn.GELU(), nn.Conv2d(dims[0], dims[1], 1)
        )
    def forward(self, x): return self.stem(x)

class HybridConvNeXtViT(nn.Module):
    """ConvNeXt stem -> tokens -> Transformer encoder -> classifier."""
    def __init__(self, num_classes=10, img=32, dim=256, depth=6, nhead=8, mlp_ratio=4.0):
        super().__init__()
        self.stem = ConvNeXtStem(3, (dim//4, dim//2))
        self.proj = nn.Conv2d(dim//2, dim, 1)
        self.cls = nn.Parameter(torch.zeros(1,1,dim))
        self.pos = nn.Parameter(torch.zeros(1, 1+(img//4)*(img//4), dim))
        enc = nn.TransformerEncoderLayer(dim, nhead, int(dim*mlp_ratio), 0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02); nn.init.trunc_normal_(self.cls, std=0.02)
    def forward(self, x):
        x = self.stem(x)
        x = self.proj(x)
        B,C,H,W = x.shape
        x = x.flatten(2).transpose(1,2)
        B,N,D = x.shape
        x = torch.cat([self.cls.expand(B,1,D), x], 1) + self.pos[:, : N+1]
        x = self.encoder(x)
        x = self.norm(x[:,0])
        return self.head(x)
