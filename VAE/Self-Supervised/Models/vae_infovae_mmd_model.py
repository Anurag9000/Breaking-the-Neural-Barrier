import torch
import torch.nn as nn
import torch.nn.functional as F

# InfoVAE / MMD-VAE: replace KL with kernel MMD(q(z), p(z))

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
    def forward(self,z):
        return self.de(self.fc(z).view(z.size(0),128,4,4))

def rbf_mmd(x, y, sigmas=(1, 2, 4, 8, 16)):
    def pdist(a):
        a2 = (a*a).sum(1, keepdim=True)
        return a2 + a2.t() - 2*a@a.t()
    Kxx, Kyy, Kxy = 0,0,0
    Dx, Dy, Dxy = pdist(x), pdist(y), (x@x.t()).new_zeros(x.size(0), y.size(0))
    # cross distances
    x2 = (x*x).sum(1, keepdim=True)
    y2 = (y*y).sum(1, keepdim=True)
    Dxy = x2 + y2.t() - 2*x@y.t()
    for s in sigmas:
        gamma = 1.0/(2*s*s)
        Kxx += torch.exp(-gamma*Dx)
        Kyy += torch.exp(-gamma*Dy)
        Kxy += torch.exp(-gamma*Dxy)
    m = x.size(0); n = y.size(0)
    return (Kxx.sum() - Kxx.trace())/(m*(m-1)) + (Kyy.sum() - Kyy.trace())/(n*(n-1)) - 2*Kxy.mean()

class InfoVAE_MMD(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, z_dim=32, mmd_lambda=10.0, recon='bce'):
        super().__init__()
        self.enc = Encoder(in_ch, z_dim)
        self.dec = Decoder(out_ch, z_dim)
        self.lmb = mmd_lambda
        self.recon = recon
    @staticmethod
    def reparam(mu, lv):
        std=(0.5*lv).exp(); eps=torch.randn_like(std); return mu + eps*std
    def forward(self,x):
        mu, lv = self.enc(x)
        z = self.reparam(mu, lv)
        x_logits = self.dec(z)
        if self.recon=='bce':
            rec = F.binary_cross_entropy_with_logits(x_logits, x, reduction='sum')/x.size(0)
        else:
            rec = F.mse_loss(torch.sigmoid(x_logits), x, reduction='sum')/x.size(0)
        # compute MMD between z ~ q(z) and samples from p(z)=N(0,I)
        z_p = torch.randn_like(z)
        mmd = rbf_mmd(z, z_p)
        return rec + self.lmb * mmd
