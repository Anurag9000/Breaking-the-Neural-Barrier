import torch
import torch.nn as nn
import torch.nn.functional as F

# Capacity-Annealed β-VAE with target C(t): loss = recon + beta * |KL - C(t)|

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
        self.fc=nn.Linear(z_dim,128*4*4)
        self.de=nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d(32,out_ch,4,2,1)
        )
    def forward(self,z):
        return self.de(self.fc(z).view(z.size(0),128,4,4))

class BetaCapacityVAE(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, z_dim=32, beta=4.0, C_max=25.0, steps=100000, recon='bce'):
        super().__init__()
        self.enc=Encoder(in_ch,z_dim)
        self.dec=Decoder(out_ch,z_dim)
        self.beta=beta; self.C_max=C_max; self.steps=steps
        self.recon=recon
        self.register_buffer('global_step', torch.zeros((), dtype=torch.long))
    @staticmethod
    def reparam(mu,lv):
        std=(0.5*lv).exp(); eps=torch.randn_like(std); return mu+eps*std
    def forward(self,x):
        mu,lv=self.enc(x)
        z=self.reparam(mu,lv)
        x_logits=self.dec(z)
        kl = -0.5*(1+lv - mu.pow(2)-lv.exp()).sum(1).mean()
        if self.recon=='bce':
            rec = F.binary_cross_entropy_with_logits(x_logits, x, reduction='sum')/x.size(0)
        else:
            rec = F.mse_loss(torch.sigmoid(x_logits), x, reduction='sum')/x.size(0)
        # linear schedule for C(t)
        t = float(self.global_step.item())
        C_t = min(self.C_max, self.C_max * t / max(1.0, self.steps))
        self.global_step += 1
        return rec + self.beta * torch.abs(kl - C_t)
