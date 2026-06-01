from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# 2-level Hierarchical VAE (HVAE)
# p(z2)=N(0,I), p(z1|z2)=N(mu1(z2), sigma1(z2)), p(x|z1)
# q(z2|x), q(z1|x,z2)
# ------------------------------

@dataclass
class HVAE2Config:
    in_channels: int = 3
    z1_dim: int = 32
    z2_dim: int = 32
    width: int = 128

class HVAE2(nn.Module):
    def __init__(self, cfg: HVAE2Config):
        super().__init__()
        self.cfg = cfg
        w = cfg.width
        # Encoder trunk
        self.enc = nn.Sequential(
            nn.Conv2d(cfg.in_channels, w, 3, 2, 1), nn.BatchNorm2d(w), nn.ReLU(True),
            nn.Conv2d(w, w*2, 3, 2, 1), nn.BatchNorm2d(w*2), nn.ReLU(True),
            nn.Conv2d(w*2, w*4, 3, 2, 1), nn.BatchNorm2d(w*4), nn.ReLU(True),
        )
        self.fc_z2_mu = nn.Linear(w*4*4*4, cfg.z2_dim)
        self.fc_z2_lv = nn.Linear(w*4*4*4, cfg.z2_dim)
        # q(z1|x,z2)
        self.fc_q1 = nn.Sequential(nn.Linear(w*4*4*4 + cfg.z2_dim, w*4), nn.ReLU(True))
        self.fc_z1_mu = nn.Linear(w*4, cfg.z1_dim)
        self.fc_z1_lv = nn.Linear(w*4, cfg.z1_dim)
        # p(z1|z2)
        self.fc_p1 = nn.Sequential(nn.Linear(cfg.z2_dim, w*4), nn.ReLU(True))
        self.fc_p1_mu = nn.Linear(w*4, cfg.z1_dim)
        self.fc_p1_lv = nn.Linear(w*4, cfg.z1_dim)
        # Decoder p(x|z1)
        self.dec_fc = nn.Linear(cfg.z1_dim, w*4*4*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(w*4, w*2, 4, 2, 1), nn.BatchNorm2d(w*2), nn.ReLU(True),
            nn.ConvTranspose2d(w*2, w, 4, 2, 1), nn.BatchNorm2d(w), nn.ReLU(True),
            nn.ConvTranspose2d(w, cfg.in_channels, 4, 2, 1),
        )

    @staticmethod
    def reparam(mu, lv):
        std = torch.exp(0.5*lv); eps = torch.randn_like(std); return mu + eps*std

    def forward(self, x):
        h = self.enc(x).view(x.size(0), -1)
        mu2, lv2 = self.fc_z2_mu(h), self.fc_z2_lv(h)
        z2 = self.reparam(mu2, lv2)
        # p(z1|z2)
        p1h = self.fc_p1(z2)
        mu1p, lv1p = self.fc_p1_mu(p1h), self.fc_p1_lv(p1h)
        # q(z1|x,z2)
        q1h = self.fc_q1(torch.cat([h, z2], dim=1))
        mu1q, lv1q = self.fc_z1_mu(q1h), self.fc_z1_lv(q1h)
        z1 = self.reparam(mu1q, lv1q)
        # decode
        dh = self.dec_fc(z1).view(x.size(0), -1, 4, 4)
        x_hat = torch.sigmoid(self.dec(dh))
        stats = (mu2, lv2, mu1q, lv1q, mu1p, lv1p)
        return x_hat, stats

    def loss(self, x, x_hat, stats):
        mu2, lv2, mu1q, lv1q, mu1p, lv1p = stats
        B = x.size(0)
        recon = F.binary_cross_entropy(x_hat, x, reduction='sum')/B
        # KL q(z2|x) || p(z2)
        kl2 = -0.5*torch.sum(1 + lv2 - mu2.pow(2) - lv2.exp())/B
        # KL q(z1|x,z2) || p(z1|z2)
        # For diagonal Gaussians, KL has analytic form
        var_q = lv1q.exp(); var_p = lv1p.exp()
        kl1 = 0.5*torch.sum(
            (var_q/var_p) + (mu1p - mu1q).pow(2)/var_p - 1 + (lv1p - lv1q)
        )/B
        return recon + kl1 + kl2, recon.detach(), (kl1+kl2).detach()
