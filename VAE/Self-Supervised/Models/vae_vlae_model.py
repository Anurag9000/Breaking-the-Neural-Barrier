import torch
import torch.nn as nn
import torch.nn.functional as F

# Variational Lossy Autoencoder (VLAE) — multi-scale decoder; higher-level z captures global structure

class Encoder(nn.Module):
    def __init__(self, in_ch=1, z_dim=32):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, 64, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(True),
        )
        self.mu = nn.Linear(256*3*3, z_dim)
        self.lv = nn.Linear(256*3*3, z_dim)
    def forward(self, x):
        h = self.down(x).view(x.size(0), -1)
        return self.mu(h), self.lv(h)

class MultiScaleDecoder(nn.Module):
    def __init__(self, out_ch=1, z_dim=32):
        super().__init__()
        self.fc = nn.Linear(z_dim, 256*3*3)
        self.block1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(True),
        )
        self.block2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU(True),
        )
        self.block3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, out_ch, 3, 1, 1),
        )
    def forward(self, z):
        h = self.fc(z).view(z.size(0), 256, 3, 3)
        h = self.block1(h)
        h = self.block2(h)
        return self.block3(h)

class VLAE(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, z_dim=32, recon='bce'):
        super().__init__()
        self.enc = Encoder(in_ch, z_dim)
        self.dec = MultiScaleDecoder(out_ch, z_dim)
        self.recon = recon
    @staticmethod
    def reparam(mu, lv):
        std=(0.5*lv).exp(); eps=torch.randn_like(std); return mu + eps*std
    def forward(self, x):
        mu, lv = self.enc(x)
        z = self.reparam(mu, lv)
        x_logits = self.dec(z)
        kl = -0.5*(1+lv - mu.pow(2) - lv.exp()).sum(1).mean()
        if self.recon=='bce':
            rec = F.binary_cross_entropy_with_logits(x_logits, x, reduction='sum')/x.size(0)
        else:
            rec = F.mse_loss(torch.sigmoid(x_logits), x, reduction='sum')/x.size(0)
        return rec + kl
