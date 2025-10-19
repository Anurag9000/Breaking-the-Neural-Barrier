import torch
import torch.nn as nn
import torch.nn.functional as F
from model_simclr_vit import ViTEncoder

class Predictor(nn.Module):
    def __init__(self, dim=384, hidden=512, out=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, out)
        )
    def forward(self, x):
        return self.net(x)

class Projection(nn.Module):
    def __init__(self, dim=384, hidden=2048, out=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True), nn.Linear(hidden, out)
        )
    def forward(self, x):
        return self.net(x)

class SimSiamViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=384, depth=6, heads=6, mlp_ratio=4.0,
                 proj_hidden=2048, proj_out=2048, pred_hidden=512):
        super().__init__()
        self.encoder = ViTEncoder(img_size, patch_size, embed_dim, depth, heads, mlp_ratio)
        self.proj = Projection(embed_dim, proj_hidden, proj_out)
        self.pred = Predictor(proj_out, pred_hidden, proj_out)

    @staticmethod
    def D(p, z):
        p = F.normalize(p, dim=-1)
        z = F.normalize(z, dim=-1)
        return -(p * z).sum(dim=-1).mean()

    def forward(self, x1, x2):
        h1 = self.encoder(x1); h2 = self.encoder(x2)
        z1 = self.proj(h1); z2 = self.proj(h2)
        p1 = self.pred(z1); p2 = self.pred(z2)
        # stop-grad on targets
        loss = self.D(p1, z2.detach())/2 + self.D(p2, z1.detach())/2
        return loss
