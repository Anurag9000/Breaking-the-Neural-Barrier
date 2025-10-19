import torch
import torch.nn as nn
import torch.nn.functional as F
from model_simclr_vit import ViTEncoder

class BarlowTwinsViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=384, depth=6, heads=6, mlp_ratio=4.0,
                 proj_hidden=8192, proj_out=8192, lambd=0.0051):
        super().__init__()
        self.encoder = ViTEncoder(img_size, patch_size, embed_dim, depth, heads, mlp_ratio)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, proj_hidden), nn.BatchNorm1d(proj_hidden), nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_out), nn.BatchNorm1d(proj_out)
        )
        self.lambd = lambd

    def loss_fn(self, z1, z2):
        z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-6)
        z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-6)
        N, D = z1.size()
        c = (z1.T @ z2) / N
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = (c - torch.diag(torch.diagonal(c))).pow_(2).sum()
        return on_diag + self.lambd * off_diag

    def forward(self, x1, x2):
        h1 = self.encoder(x1); h2 = self.encoder(x2)
        z1 = self.proj(h1); z2 = self.proj(h2)
        return self.loss_fn(z1, z2)
