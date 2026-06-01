import torch
import torch.nn as nn
import torch.nn.functional as F

# VAE with Laplace (L1) prior over z to encourage sparsity

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

class LaplacePriorVAE(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, z_dim=32, b=1.0, recon='bce'):
        super().__init__()
        self.enc = Encoder(in_ch, z_dim)
        self.dec = Decoder(out_ch, z_dim)
        self.b = b
        self.recon = recon
    @staticmethod
    def reparam(mu, lv):
        std=(0.5*lv).exp(); eps=torch.randn_like(std); return mu + eps*std
    def forward(self,x):
        mu,lv=self.enc(x)
        z=self.reparam(mu,lv)
        x_logits=self.dec(z)
        # KL(q||p_laplace) without closed form; use MC estimate: -E_q[log p(z)] + E_q[log q(z)]
        # log p_laplace(z) = -|z|/b - log(2b)
        log_pz = -(z.abs()/self.b).sum(-1) - z.size(1)*torch.log(torch.tensor(2*self.b, device=x.device))
        log_qz = -0.5*( (z-mu).pow(2)/lv.exp() + lv + torch.log(torch.tensor(2*3.141592653589793, device=x.device)) ).sum(-1)
        kl = (-(log_pz) - log_qz).mean()
        if self.recon=='bce':
            rec=F.binary_cross_entropy_with_logits(x_logits,x,reduction='sum')/x.size(0)
        else:
            rec=F.mse_loss(torch.sigmoid(x_logits),x,reduction='sum')/x.size(0)
        return rec + kl
