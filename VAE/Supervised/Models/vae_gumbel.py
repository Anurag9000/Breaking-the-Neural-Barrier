from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------
# Concrete / Gumbel-Softmax VAE: discrete latent with straight-through relaxations
# --------------------------------------

@dataclass
class GumbelVAEConfig:
    in_channels: int = 3
    width: int = 128
    K: int = 16            # categories per discrete variable
    groups: int = 8        # number of discrete variables
    tau: float = 0.67

class GumbelVAE(nn.Module):
    def __init__(self, cfg: GumbelVAEConfig):
        super().__init__()
        self.cfg = cfg
        w = cfg.width
        self.enc = nn.Sequential(
            nn.Conv2d(cfg.in_channels, w, 3, 2, 1), nn.BatchNorm2d(w), nn.ReLU(True),
            nn.Conv2d(w, w*2, 3, 2, 1), nn.BatchNorm2d(w*2), nn.ReLU(True),
            nn.Conv2d(w*2, w*4, 3, 2, 1), nn.BatchNorm2d(w*4), nn.ReLU(True),
        )
        self.fc_logits = nn.Linear(w*4*4*4, cfg.groups*cfg.K)
        self.fc = nn.Linear(cfg.groups*cfg.K, w*4*4*4)
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

    def forward(self, x):
        h = self.enc(x).view(x.size(0), -1)
        logits = self.fc_logits(h).view(x.size(0), self.cfg.groups, self.cfg.K)
        qy = self.gumbel_softmax(logits, self.cfg.tau, hard=True)
        z = qy.view(x.size(0), -1)
        x_hat = torch.sigmoid(self.dec(self.fc(z).view(x.size(0), -1, 4, 4)))
        return x_hat, logits, qy

    def loss(self, x, x_hat, logits, qy):
        recon = F.binary_cross_entropy(x_hat, x, reduction='sum')/x.size(0)
        # KL to uniform categorical prior
        log_qy = torch.log(qy + 1e-9)
        kl = (qy * (log_qy - torch.log(torch.tensor(1.0/self.cfg.K, device=x.device)))).sum(dim=(1,2)).mean()
        return recon + kl, recon.detach(), kl.detach()
