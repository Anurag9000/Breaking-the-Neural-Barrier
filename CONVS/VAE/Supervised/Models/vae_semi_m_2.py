from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------
# Semi-supervised VAE (M2): y as discrete latent, relaxed with Gumbel-Softmax
# Labeled data: maximize ELBO with y fixed; Unlabeled: marginalize y via q(y|x)
# Here we implement a compact fully-labeled path that still uses the same components.
# --------------------------------------

@dataclass
class M2Config:
    in_channels: int = 3
    num_classes: int = 10
    latent_dim: int = 32
    width: int = 128
    tau: float = 0.67  # Gumbel temperature

class VAE_M2(nn.Module):
    def __init__(self, cfg: M2Config):
        super().__init__()
        self.cfg = cfg
        w = cfg.width
        self.cls = nn.Sequential(
            nn.Conv2d(cfg.in_channels, w, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(w, w*2, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(w*2, w*4, 3, 2, 1), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc_cls = nn.Linear(w*4, cfg.num_classes)
        # Encoder q(z|x,y)
        self.enc = nn.Sequential(
            nn.Conv2d(cfg.in_channels + cfg.num_classes, w, 3, 2, 1), nn.BatchNorm2d(w), nn.ReLU(True),
            nn.Conv2d(w, w*2, 3, 2, 1), nn.BatchNorm2d(w*2), nn.ReLU(True),
            nn.Conv2d(w*2, w*4, 3, 2, 1), nn.BatchNorm2d(w*4), nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(w*4*4*4, cfg.latent_dim)
        self.fc_lv = nn.Linear(w*4*4*4, cfg.latent_dim)
        # Decoder p(x|z,y)
        self.fc = nn.Linear(cfg.latent_dim + cfg.num_classes, w*4*4*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(w*4, w*2, 4, 2, 1), nn.BatchNorm2d(w*2), nn.ReLU(True),
            nn.ConvTranspose2d(w*2, w, 4, 2, 1), nn.BatchNorm2d(w), nn.ReLU(True),
            nn.ConvTranspose2d(w, cfg.in_channels, 4, 2, 1),
        )

    def gumbel_softmax(self, logits, tau, hard=False):
        u = torch.rand_like(logits)
        g = -torch.log(-torch.log(u + 1e-9) + 1e-9)
        y = F.softmax((logits + g)/tau, dim=-1)
        if hard:
            k = y.max(-1, keepdim=True)[1]
            y_hard = torch.zeros_like(y).scatter_(-1, k, 1.0)
            y = y_hard + (y - y.detach())
        return y

    @staticmethod
    def reparam(mu, lv):
        std = torch.exp(0.5*lv); eps = torch.randn_like(std); return mu + eps*std

    def forward(self, x, y=None):
        B, _, H, W = x.shape
        # q(y|x)
        cls_logits = self.fc_cls(self.cls(x).view(B,-1))
        if y is None:
            y_onehot = self.gumbel_softmax(cls_logits, self.cfg.tau, hard=True)
        else:
            y_onehot = F.one_hot(y, num_classes=self.cfg.num_classes).float()
        y_img = y_onehot.view(B, -1, 1, 1).expand(B, -1, H, W)
        # q(z|x,y)
        h = self.enc(torch.cat([x, y_img], dim=1)).view(B,-1)
        mu, lv = self.fc_mu(h), self.fc_lv(h)
        z = self.reparam(mu, lv)
        # p(x|z,y)
        dec_in = torch.cat([z, y_onehot], dim=1)
        h = self.fc(dec_in).view(B, -1, 4, 4)
        x_hat = torch.sigmoid(self.dec(h))
        return x_hat, mu, lv, cls_logits

    def loss(self, x, x_hat, mu, lv, cls_logits, y=None):
        recon = F.binary_cross_entropy(x_hat, x, reduction='sum')/x.size(0)
        kl = -0.5*torch.sum(1 + lv - mu.pow(2) - lv.exp())/x.size(0)
        ce = torch.tensor(0.0, device=x.device)
        if y is not None:
            ce = F.cross_entropy(cls_logits, y)
        return recon + kl + ce, recon.detach(), kl.detach(), ce.detach()
