from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# PixelVAE: simple masked conv decoder inside VAE (single-model)
# NOTE: This is a compact PixelCNN-like decoder (A-masks)
# ------------------------------

class MaskedConv2d(nn.Conv2d):
    def __init__(self, in_ch, out_ch, k, mask_type='A', **kwargs):
        super().__init__(in_ch, out_ch, k, **kwargs)
        self.register_buffer('mask', torch.ones_like(self.weight))
        _, _, kh, kw = self.weight.size()
        yc, xc = kh//2, kw//2
        self.mask[:,:,yc,xc+ (mask_type=='B'):] = 0
        self.mask[:,:,yc+1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)

@dataclass
class PixelVAEConfig:
    in_channels: int = 3
    latent_dim: int = 64
    width: int = 64

class PixelVAE(nn.Module):
    def __init__(self, cfg: PixelVAEConfig):
        super().__init__()
        self.cfg = cfg
        w = cfg.width
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(cfg.in_channels, w, 3, 2, 1), nn.BatchNorm2d(w), nn.ReLU(True),
            nn.Conv2d(w, w*2, 3, 2, 1), nn.BatchNorm2d(w*2), nn.ReLU(True),
            nn.Conv2d(w*2, w*4, 3, 2, 1), nn.BatchNorm2d(w*4), nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(w*4*4*4, cfg.latent_dim)
        self.fc_lv = nn.Linear(w*4*4*4, cfg.latent_dim)
        self.fc = nn.Linear(cfg.latent_dim, w*2*32*32)
        # PixelCNN-lite decoder operating on feature map + channel conditioning
        self.init = MaskedConv2d(w*2, w*2, 7, padding=3)
        self.body = nn.Sequential(
            nn.ReLU(True), MaskedConv2d(w*2, w*2, 3, padding=1),
            nn.ReLU(True), MaskedConv2d(w*2, w*2, 3, padding=1),
        )
        self.out = nn.Conv2d(w*2, cfg.in_channels, 1)

    @staticmethod
    def reparam(mu, lv):
        std = torch.exp(0.5*lv); eps = torch.randn_like(std); return mu + eps*std

    def forward(self, x):
        h = self.enc(x).view(x.size(0), -1)
        mu, lv = self.fc_mu(h), self.fc_lv(h)
        z = self.reparam(mu, lv)
        feat = self.fc(z).view(x.size(0), -1, 32, 32)
        y = self.init(feat)
        y = self.body(y)
        x_hat = torch.sigmoid(self.out(y))
        return x_hat, mu, lv

    def loss(self, x, x_hat, mu, lv):
        recon = F.binary_cross_entropy(x_hat, x, reduction='sum')/x.size(0)
        kl = -0.5*torch.sum(1 + lv - mu.pow(2) - lv.exp())/x.size(0)
        return recon + kl, recon.detach(), kl.detach()
