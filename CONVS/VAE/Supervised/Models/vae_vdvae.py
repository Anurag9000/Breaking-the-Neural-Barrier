from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------
# VDVAE (very deep VAE) — simplified with multiple latent groups in sequence
# --------------------------------------

@dataclass
class VDVAEConfig:
    in_channels: int = 3
    width: int = 96
    groups: int = 6
    z_dim: int = 16

class Block(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(ch, ch, 3, 1, 1)
        )
    def forward(self, x):
        return F.relu(x + self.net(x), inplace=True)

class VDVAE(nn.Module):
    def __init__(self, cfg: VDVAEConfig):
        super().__init__()
        self.cfg = cfg
        w = cfg.width
        self.stem = nn.Conv2d(cfg.in_channels, w, 3, 1, 1)
        self.enc = nn.ModuleList([
            nn.Sequential(nn.Conv2d(w, w, 3, 2, 1), Block(w)) if i%2==0 else Block(w)
            for i in range(cfg.groups)
        ])
        self.post_mu = nn.ModuleList([nn.Conv2d(w, cfg.z_dim, 1) for _ in range(cfg.groups)])
        self.post_lv = nn.ModuleList([nn.Conv2d(w, cfg.z_dim, 1) for _ in range(cfg.groups)])
        self.dec_blocks = nn.ModuleList([Block(w) for _ in range(cfg.groups)])
        self.ups = nn.ModuleList([nn.ConvTranspose2d(w + cfg.z_dim, w, 4, 2, 1) for _ in range(cfg.groups//2)])
        self.head = nn.Conv2d(w, cfg.in_channels, 1)

    @staticmethod
    def reparam(mu, lv):
        std = torch.exp(0.5*lv); eps = torch.randn_like(std); return mu + eps*std

    def forward(self, x):
        h = self.stem(x)
        feats = []
        for i,blk in enumerate(self.enc):
            h = blk(h)
            feats.append(h)
        kls = []
        y = feats[-1]
        up_idx = 0
        for i in reversed(range(self.cfg.groups)):
            mu = self.post_mu[i](feats[i]); lv = self.post_lv[i](feats[i])
            z = self.reparam(mu, lv)
            y = torch.cat([y, z], dim=1)
            if i % 2 == 0 and up_idx < len(self.ups):
                y = self.ups[up_idx](y); up_idx += 1
            y = self.dec_blocks[i](y if y.size(1)==self.cfg.width else y[:, :self.cfg.width])
            kl_i = -0.5*torch.sum(1 + lv - mu.pow(2) - lv.exp())/x.size(0)
            kls.append(kl_i)
        x_hat = torch.sigmoid(self.head(y))
        kl = torch.stack(kls).sum()
        return x_hat, kl

    def loss(self, x, x_hat, kl):
        recon = F.binary_cross_entropy(x_hat, x, reduction='sum')/x.size(0)
        return recon + kl, recon.detach(), kl.detach()
