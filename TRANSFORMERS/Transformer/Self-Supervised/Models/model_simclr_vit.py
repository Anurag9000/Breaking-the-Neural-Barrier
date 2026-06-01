import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# Minimal ViT encoder reused
from model_mae_vit import PatchEmbed, TransformerEncoder

class ViTEncoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=384, depth=6, heads=6, mlp_ratio=4.0):
        super().__init__()
        self.patch = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.pos = nn.Parameter(torch.zeros(1, (img_size//patch_size)**2, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.encoder = TransformerEncoder(embed_dim, depth, heads, mlp_ratio)
        self.norm = nn.LayerNorm(embed_dim)
        self.pool = 'mean'

    def forward(self, x):
        x = self.patch(x) + self.pos
        x = self.encoder(x)
        x = self.norm(x)
        return x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

class MLPHead(nn.Module):
    def __init__(self, dim, hidden=2048, out=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, out)
        )
    def forward(self, x):
        return self.net(x)

class SimCLRViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=384, depth=6, heads=6, mlp_ratio=4.0,
                 proj_hidden=2048, proj_out=128):
        super().__init__()
        self.encoder = ViTEncoder(img_size, patch_size, embed_dim, depth, heads, mlp_ratio)
        self.proj = MLPHead(embed_dim, proj_hidden, proj_out)

    @staticmethod
    def nt_xent(z1, z2, temperature=0.2):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        z = torch.cat([z1, z2], dim=0)
        sim = (z @ z.t()) / temperature
        B = z1.size(0)
        mask = torch.eye(2*B, device=z.device).bool()
        sim = sim.masked_fill(mask, -9e15)
        targets = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)]).to(z.device)
        loss = F.cross_entropy(sim, targets)
        return loss

    def forward(self, x1, x2):
        h1 = self.encoder(x1)
        h2 = self.encoder(x2)
        z1 = self.proj(h1)
        z2 = self.proj(h2)
        return z1, z2
