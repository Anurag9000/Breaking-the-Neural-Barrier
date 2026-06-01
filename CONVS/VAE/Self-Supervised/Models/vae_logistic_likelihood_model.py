import torch
import torch.nn as nn
import torch.nn.functional as F

# VAE with continuous Logistic likelihood p(x|z) = Logistic(mu(z), s)

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
    def forward(self,x):
        h=self.conv(x).view(x.size(0),-1)
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
        self.log_s = nn.Parameter(torch.tensor(0.0))
    def forward(self,z):
        mu = self.de(self.fc(z).view(z.size(0),128,4,4))
        return mu, self.log_s

class LogisticLikVAE(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, z_dim=32):
        super().__init__()
        self.enc = Encoder(in_ch, z_dim)
        self.dec = Decoder(out_ch, z_dim)
    @staticmethod
    def reparam(mu, lv):
        std=(0.5*lv).exp(); eps=torch.randn_like(std); return mu + eps*std
    def forward(self,x):
        mu,lv=self.enc(x)
        z=self.reparam(mu,lv)
        x_mu, log_s = self.dec(z)
        s = torch.exp(log_s)
        # logistic log-likelihood: -log s - 2*softplus((x - mu)/s)
        t = (x - x_mu)/s
        log_px = (-log_s - 2*F.softplus(t)).flatten(1).sum(-1)
        kl=-0.5*(1+lv - mu.pow(2)-lv.exp()).sum(1)
        return (-(log_px) + kl).mean()
