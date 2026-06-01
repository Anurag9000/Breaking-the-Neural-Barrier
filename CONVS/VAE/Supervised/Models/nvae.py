from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------
# NVAE (simplified): hierarchical conv VAE with multiple latent groups
# This is a compact educational variant capturing the core idea:
# multi-scale encoder/decoder with several stochastic latent groups z_i.
# --------------------------------------

@dataclass
class NVAEConfig:
    in_channels: int = 3
    width: int = 128
    z_groups: int = 3
    z_per_group: int = 16

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(ch)
    def forward(self, x):
        id = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + id, inplace=True)

class NVAE(nn.Module):
    def __init__(self, cfg: NVAEConfig):
        super().__init__()
        self.cfg = cfg
        w = cfg.width
        # Encoder pyramid
        self.enc1 = nn.Sequential(
            nn.Conv2d(cfg.in_channels, w, 3, 2, 1), nn.ReLU(True), ResBlock(w))
        self.enc2 = nn.Sequential(
            nn.Conv2d(w, w*2, 3, 2, 1), nn.ReLU(True), ResBlock(w*2))
        self.enc3 = nn.Sequential(
            nn.Conv2d(w*2, w*4, 3, 2, 1), nn.ReLU(True), ResBlock(w*4))
        # Per-group posterior heads (top-down conditioning during decode)
        D = cfg.z_per_group
        self.post3_mu = nn.Conv2d(w*4, D, 1)
        self.post3_lv = nn.Conv2d(w*4, D, 1)
        self.post2_mu = nn.Conv2d(w*2, D, 1)
        self.post2_lv = nn.Conv2d(w*2, D, 1)
        self.post1_mu = nn.Conv2d(w, D, 1)
        self.post1_lv = nn.Conv2d(w, D, 1)
        # Decoder pyramid (top-down): upsample + fuse z at each scale
        self.dec3 = nn.Sequential(ResBlock(w*4), nn.ConvTranspose2d(w*4, w*2, 4, 2, 1), nn.ReLU(True))
        self.dec2 = nn.Sequential(ResBlock(w*2 + D), nn.ConvTranspose2d(w*2 + D, w, 4, 2, 1), nn.ReLU(True))
        self.dec1 = nn.Sequential(ResBlock(w + D), nn.ConvTranspose2d(w + D, w, 3, 1, 1), nn.ReLU(True))
        self.out = nn.Conv2d(w, cfg.in_channels, 1)

    @staticmethod
    def reparam(mu, lv):
        std = torch.exp(0.5*lv); eps = torch.randn_like(std); return mu + eps*std

    def forward(self, x):
        h1 = self.enc1(x)        # 16x16
        h2 = self.enc2(h1)       # 8x8
        h3 = self.enc3(h2)       # 4x4
        # Posteriors at each scale
        mu3, lv3 = self.post3_mu(h3), self.post3_lv(h3)
        z3 = self.reparam(mu3, lv3)
        y2 = self.dec3(h3)       # 8x8, features
        mu2, lv2 = self.post2_mu(h2), self.post2_lv(h2)
        z2 = self.reparam(mu2, lv2)
        y2 = torch.cat([y2, z2], dim=1)
        y1 = self.dec2(y2)       # 16x16
        mu1, lv1 = self.post1_mu(h1), self.post1_lv(h1)
        z1 = self.reparam(mu1, lv1)
        y1 = torch.cat([y1, z1], dim=1)
        y0 = self.dec1(y1)
        x_hat = torch.sigmoid(self.out(y0))
        stats = (mu1, lv1, mu2, lv2, mu3, lv3)
        return x_hat, stats

    def loss(self, x, x_hat, stats):
        mu1, lv1, mu2, lv2, mu3, lv3 = stats
        B = x.size(0)
        recon = F.binary_cross_entropy(x_hat, x, reduction='sum')/B
        kl = 0.0
        for mu, lv in [(mu1,lv1),(mu2,lv2),(mu3,lv3)]:
            kl += -0.5*torch.sum(1 + lv - mu.pow(2) - lv.exp())/B
        return recon + kl, recon.detach(), kl.detach()
