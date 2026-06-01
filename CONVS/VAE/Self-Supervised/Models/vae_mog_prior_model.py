import torch
import torch.nn as nn
import torch.nn.functional as F

# VAE with learnable Mixture-of-Gaussians prior p(z) = sum_k pi_k N(mu_k, diag(sig2_k))

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

class MoGPriorVAE(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, z_dim=32, K=10, recon='bce'):
        super().__init__()
        self.enc = Encoder(in_ch, z_dim)
        self.dec = Decoder(out_ch, z_dim)
        self.K = K
        self.recon = recon
        self.mu_k = nn.Parameter(torch.randn(K, z_dim)*0.1)
        self.lv_k = nn.Parameter(torch.zeros(K, z_dim))
        self.logits = nn.Parameter(torch.zeros(K))
    @staticmethod
    def reparam(mu, lv):
        std=(0.5*lv).exp(); eps=torch.randn_like(std); return mu + eps*std
    def log_mog(self, z):
        # log sum_k pi_k N(z|mu_k, lv_k)
        pi = torch.softmax(self.logits, dim=0)  # [K]
        z_ = z.unsqueeze(1)  # [B,1,D]
        mu = self.mu_k.unsqueeze(0)  # [1,K,D]
        lv = self.lv_k.unsqueeze(0)
        log_comp = -0.5*( ((z_-mu).pow(2)/lv.exp()) + lv + torch.log(torch.tensor(2*3.141592653589793, device=z.device)) ).sum(-1) + torch.log(pi.unsqueeze(0)+1e-8)
        return torch.logsumexp(log_comp, dim=1)
    def forward(self,x):
        mu, lv = self.enc(x)
        z = self.reparam(mu, lv)
        x_logits = self.dec(z)
        log_pz = self.log_mog(z)
        log_qz = -0.5*( (z-mu).pow(2)/lv.exp() + lv + torch.log(torch.tensor(2*3.141592653589793, device=x.device)) ).sum(-1)
        if self.recon=='bce':
            rec = F.binary_cross_entropy_with_logits(x_logits, x, reduction='none').flatten(1).sum(-1)
        else:
            rec = F.mse_loss(torch.sigmoid(x_logits), x, reduction='none').flatten(1).sum(-1)
        elbo = -rec + log_pz - log_qz
        return (-elbo).mean()
