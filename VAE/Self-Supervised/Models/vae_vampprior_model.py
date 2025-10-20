import torch
import torch.nn as nn
import torch.nn.functional as F

# VAMP Prior VAE: prior is mixture of encoder posteriors on learnable pseudo-inputs

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
    def forward(self, z):
        return self.de(self.fc(z).view(z.size(0),128,4,4))

class VampPriorVAE(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, z_dim=32, K=500, recon='bce'):
        super().__init__()
        self.enc = Encoder(in_ch, z_dim)
        self.dec = Decoder(out_ch, z_dim)
        self.recon = recon
        self.K = K
        # pseudo-inputs initialized near data mean
        self.pseudo = nn.Parameter(torch.rand(K, in_ch, 28, 28))
    @staticmethod
    def reparam(mu, lv):
        std=(0.5*lv).exp(); eps=torch.randn_like(std); return mu + eps*std
    def log_mix_q(self, z):
        # compute log (1/K sum_k q(z|u_k)) via log-sum-exp
        with torch.no_grad():
            # avoid exploding grads through enc for stability in mixture weights; grads still flow via separate call below if needed
            pass
        mu_k, lv_k = self.enc(self.pseudo)  # [K,D] x2
        z_ = z.unsqueeze(1)  # [B,1,D]
        mu = mu_k.unsqueeze(0)  # [1,K,D]
        lv = lv_k.unsqueeze(0)
        log_q = -0.5*( ((z_-mu).pow(2)/lv.exp()) + lv + torch.log(torch.tensor(2*3.141592653589793, device=z.device)) ).sum(-1)  # [B,K]
        return torch.logsumexp(log_q - torch.log(torch.tensor(self.K, device=z.device, dtype=torch.float)), dim=1)  # [B]
    def forward(self, x):
        mu, lv = self.enc(x)
        z = self.reparam(mu, lv)
        x_logits = self.dec(z)
        log_pz = self.log_mix_q(z)
        log_qz = -0.5*( (z-mu).pow(2)/lv.exp() + lv + torch.log(torch.tensor(2*3.141592653589793, device=x.device)) ).sum(-1)
        if self.recon=='bce':
            rec = F.binary_cross_entropy_with_logits(x_logits, x, reduction='none').flatten(1).sum(-1)
        else:
            rec = F.mse_loss(torch.sigmoid(x_logits), x, reduction='none').flatten(1).sum(-1)
        elbo = -rec + log_pz - log_qz
        return (-elbo).mean()
