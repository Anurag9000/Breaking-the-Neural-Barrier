import torch
import torch.nn as nn
import torch.nn.functional as F

# Householder-flow VAE (volume-preserving, logdet=0)

class Householder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.v = nn.Parameter(torch.randn(dim))
    def forward(self, z):
        v = self.v / (self.v.norm()+1e-8)
        # H = I - 2 v v^T
        proj = (z @ v).unsqueeze(-1) * v  # [B,D]
        z_out = z - 2*proj
        log_det = torch.zeros(z.size(0), device=z.device)
        return z_out, log_det

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
            nn.ConvTranspose2d(32,out_ch,4,2,1),
        )
    def forward(self,z):
        return self.de(self.fc(z).view(z.size(0),128,4,4))

class HouseholderFlowVAE(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, z_dim=32, n_flows=4, recon='bce'):
        super().__init__()
        self.enc = Encoder(in_ch, z_dim)
        self.dec = Decoder(out_ch, z_dim)
        self.flows = nn.ModuleList([Householder(z_dim) for _ in range(n_flows)])
        self.recon = recon
    @staticmethod
    def reparam(mu, lv):
        std=(0.5*lv).exp(); eps=torch.randn_like(std); return mu + eps*std
    def forward(self,x):
        mu, lv = self.enc(x)
        z = self.reparam(mu, lv)
        sum_logdet = torch.zeros(x.size(0), device=x.device)
        for f in self.flows:
            z, ld = f(z); sum_logdet += ld
        x_logits = self.dec(z)
        log_pz = -0.5*(z.pow(2) + torch.log(torch.tensor(2*3.141592653589793, device=x.device))).sum(-1)
        log_qz = -0.5*((z-mu).pow(2)/lv.exp() + lv + torch.log(torch.tensor(2*3.141592653589793, device=x.device))).sum(-1)
        if self.recon=='bce':
            rec = F.binary_cross_entropy_with_logits(x_logits, x, reduction='none').flatten(1).sum(-1)
        else:
            rec = F.mse_loss(torch.sigmoid(x_logits), x, reduction='none').flatten(1).sum(-1)
        elbo = -rec + log_pz - (log_qz - sum_logdet)
        return (-elbo).mean()
