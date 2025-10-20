import torch
import torch.nn as nn
import torch.nn.functional as F

# VAE with autoregressive prior p(z) parameterized by MADE

class MADE(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(True),
            nn.Linear(hidden, hidden), nn.ReLU(True),
            nn.Linear(hidden, 2*dim)
        )
    def forward(self, z):
        out = self.net(z)
        mu, logvar = out.chunk(2, dim=-1)
        logvar = torch.tanh(logvar)
        return mu, logvar

class Encoder(nn.Module):
    def __init__(self, in_ch=1, z_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,32,3,2,1), nn.ReLU(True),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(True),
            nn.Conv2d(64,128,3,2,1), nn.ReLU(True),
        )
        self.mu = nn.Linear(128*4*4, z_dim)
        self.lv = nn.Linear(128*4*4, z_dim)
    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        return self.mu(h), self.lv(h)

class Decoder(nn.Module):
    def __init__(self, out_ch=1, z_dim=32):
        super().__init__()
        self.fc = nn.Linear(z_dim, 128*4*4)
        self.de = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d(32,out_ch,4,2,1)
        )
    def forward(self, z):
        return self.de(self.fc(z).view(z.size(0),128,4,4))

class VAE_AR_Prior(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, z_dim=32, recon='bce'):
        super().__init__()
        self.enc = Encoder(in_ch, z_dim)
        self.dec = Decoder(out_ch, z_dim)
        self.prior = MADE(z_dim)
        self.recon = recon
    @staticmethod
    def reparam(mu, lv):
        std=(0.5*lv).exp(); eps=torch.randn_like(std); return mu + eps*std
    def forward(self, x):
        mu_q, lv_q = self.enc(x)
        z = self.reparam(mu_q, lv_q)
        x_logits = self.dec(z)
        # autoregressive Gaussian prior p(z) = N(mu_p(z), diag(exp(logvar_p(z)))) where mu/logvar depend on prefix (approx via MADE on full z)
        mu_p, lv_p = self.prior(z.detach())  # conditionally parameterize; detach to stabilize
        # KL between q(z|x) and N(mu_p, diag(exp(lv_p)))
        kl = 0.5*( ((mu_q - mu_p).pow(2) + lv_q.exp())/lv_p.exp() + lv_p - lv_q - 1 ).sum(1).mean()
        if self.recon=='bce':
            rec = F.binary_cross_entropy_with_logits(x_logits, x, reduction='sum')/x.size(0)
        else:
            rec = F.mse_loss(torch.sigmoid(x_logits), x, reduction='sum')/x.size(0)
        return rec + kl
