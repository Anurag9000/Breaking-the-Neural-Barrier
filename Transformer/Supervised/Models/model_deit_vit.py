import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch=4, in_ch=3, dim=256):
        super().__init__(); self.proj = nn.Conv2d(in_ch, dim, patch, patch)
        self.grid = (img_size//patch, img_size//patch)
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1,2); return x

class DeiT(nn.Module):
    def __init__(self, num_classes=10, img=32, patch=4, dim=256, depth=6, nhead=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch = PatchEmbed(img, patch, 3, dim)
        self.cls = nn.Parameter(torch.zeros(1,1,dim))
        self.pos = nn.Parameter(torch.zeros(1, 1+(img//patch)**2, dim))
        enc = nn.TransformerEncoderLayer(dim, nhead, int(dim*mlp_ratio), dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, depth)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02); nn.init.trunc_normal_(self.cls, std=0.02)
    def forward(self, x):
        x = self.patch(x)
        B,N,D = x.size(); cls = self.cls.expand(B,-1,-1)
        x = torch.cat([cls, x], dim=1) + self.pos[:, :N+1]
        x = self.encoder(x)
        x = self.norm(x[:,0])
        return self.head(x)
