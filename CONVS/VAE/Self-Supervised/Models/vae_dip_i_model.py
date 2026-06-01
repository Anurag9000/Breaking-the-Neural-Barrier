import torch
import torch.nn as nn
import torch.nn.functional as F

# DIP-VAE-I: Penalize off-diagonal of Cov[μ(x)] so aggregated posterior approaches factorized prior

class Encoder(nn.Module):
    def __init__(self, in_ch=1, z_dim=32):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_ch,32,3,2,1), nn.ReLU(True),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(True),
            nn.Conv2d(64,128,3,2,1), nn.ReLU(True),
        )
        self.mu=nn.Linear(128*4*4,z_dim)
        self.lv=nn.Linear(128*4*4,z_dim)
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

class DIPVAE_I(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, z_dim=32, lam_off=10.0, lam_diag=1.0, recon='bce'):
        super().__init__()
        self.enc=Encoder(in_ch,z_dim)
        self.dec=Decoder(out_ch,z_dim)
        self.lam_off=lam_off; self.lam_diag=lam_diag
        self.recon=recon
    @staticmethod
    def reparam(mu,lv):
        std=(0.5*lv).exp(); eps=torch.randn_like(std); return mu+eps*std
    def cov_penalty(self, mu):
        # estimate Cov over batch
        mu_c = mu - mu.mean(0, keepdim=True)
        C = (mu_c.t() @ mu_c) / (mu.size(0) - 1 + 1e-6)
        off = C - torch.diag(torch.diag(C))
        return self.lam_off*(off.pow(2).sum()) + self.lam_diag*((torch.diag(C)-1).pow(2).sum())
    def forward(self,x):
        mu,lv=self.enc(x)
        z=self.reparam(mu,lv)
        x_logits=self.dec(z)
        kl=-0.5*(1+lv - mu.pow(2)-lv.exp()).sum(1).mean()
        if self.recon=='bce':
            rec=F.binary_cross_entropy_with_logits(x_logits,x,reduction='sum')/x.size(0)
        else:
            rec=F.mse_loss(torch.sigmoid(x_logits),x,reduction='sum')/x.size(0)
        pen=self.cov_penalty(mu)
        return rec + kl + pen
