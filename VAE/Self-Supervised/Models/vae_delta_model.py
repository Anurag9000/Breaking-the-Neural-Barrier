import torch
import torch.nn as nn
import torch.nn.functional as F

# δ-VAE: enforce minimum posterior variance to prevent collapse

class Encoder(nn.Module):
    def __init__(self, in_ch=1, z_dim=32, logvar_floor=-2.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch,32,3,2,1), nn.ReLU(True),
            nn.Conv2d(32,64,3,2,1), nn.ReLU(True),
            nn.Conv2d(64,128,3,2,1), nn.ReLU(True),
        )
        self.mu = nn.Linear(128*4*4, z_dim)
        self.lv_raw = nn.Linear(128*4*4, z_dim)
        self.logvar_floor = logvar_floor
    def forward(self,x):
        h=self.conv(x).view(x.size(0),-1)
        mu = self.mu(h)
        lv_hat = self.lv_raw(h)
        # softplus flooring: logvar = log(exp(lv_hat) + exp(floor))
        lv = torch.log(torch.exp(lv_hat) + torch.exp(torch.tensor(self.logvar_floor, device=x.device)))
        return mu, lv

class Decoder(nn.Module):
    def __init__(self, out_ch=1, z_dim=32):
        super().__init__()
        self.fc = nn.Linear(z_dim, 128*4*4)
        self.de = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d(64,32,4,2,1), nn.ReLU(True),
            nn.ConvTranspose2d(32,out_ch,4,2,1)
        )
    def forward(self,z):
        return self.de(self.fc(z).view(z.size(0),128,4,4))

class DeltaVAE(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, z_dim=32, logvar_floor=-2.0, recon='bce'):
        super().__init__()
        self.enc = Encoder(in_ch, z_dim, logvar_floor)
        self.dec = Decoder(out_ch, z_dim)
        self.recon = recon
    @staticmethod
    def reparam(mu, lv):
        std=(0.5*lv).exp(); eps=torch.randn_like(std); return mu + eps*std
    def forward(self,x):
        mu, lv = self.enc(x)
        z = self.reparam(mu, lv)
        x_logits = self.dec(z)
        kl = -0.5*(1+lv - mu.pow(2) - lv.exp()).sum(1).mean()
        if self.recon=='bce':
            rec = F.binary_cross_entropy_with_logits(x_logits, x, reduction='sum')/x.size(0)
        else:
            rec = F.mse_loss(torch.sigmoid(x_logits), x, reduction='sum')/x.size(0)
        return rec + kl
