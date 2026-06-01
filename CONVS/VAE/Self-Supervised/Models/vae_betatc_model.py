import torch
import torch.nn as nn
import torch.nn.functional as F

# β-TC VAE: weighted decomposition of KL; estimate total correlation with minibatch strategy
# ELBO ≈ E[log p(x|z)] - α * I(x; z) - β * TC(z) - γ * ∑ KL(q(z_j)||p(z_j)), with α=γ=1, β>1 typical

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

class BetaTCVAE(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, z_dim=32, beta=6.0, recon='bce'):
        super().__init__()
        self.enc = Encoder(in_ch, z_dim)
        self.dec = Decoder(out_ch, z_dim)
        self.recon = recon
        self.beta = beta

    @staticmethod
    def reparam(mu, lv):
        std=(0.5*lv).exp(); eps=torch.randn_like(std); return mu + eps*std

    def _log_q_z(self, z, mu, lv):
        # log q(z|x) under encoder for each sample (diagonal Gaussian)
        return -0.5*( (z-mu).pow(2)/lv.exp() + lv + torch.log(torch.tensor(2*3.141592653589793, device=z.device)) ).sum(-1)

    def forward(self, x):
        mu, lv = self.enc(x)
        z = self.reparam(mu, lv)
        x_logits = self.dec(z)
        # estimates
        # log q(z) via minibatch weighting (factorized importance sampling)
        mu = mu.detach(); lv = lv.detach()
        B = z.size(0)
        log_qz_x = self._log_q_z(z, mu, lv)  # [B]
        # compute log q(z) ≈ log(1/B ∑_i q(z|x_i))
        log_qzi = []
        for i in range(B):
            log_qzi.append(self._log_q_z(z[i:i+1], mu, lv))
        log_qz_matrix = torch.stack(log_qzi, dim=1)  # [1,B] -> [B,B]
        log_qz = torch.logsumexp(log_qz_matrix, dim=1) - torch.log(torch.tensor(B, device=x.device, dtype=torch.float))
        # log prod_j q(z_j) using product of marginals
        # for diagonal Gaussians, log q(z_j) uses scalar sums
        def log_sum_exp_marginals(z, mu, lv):
            B, D = z.size()
            z = z.unsqueeze(1)  # [B,1,D]
            mu = mu.unsqueeze(0)  # [1,B,D]
            lv = lv.unsqueeze(0)
            log_comp = -0.5*( (z-mu).pow(2)/lv.exp() + lv + torch.log(torch.tensor(2*3.141592653589793, device=x.device)) )  # [B,B,D]
            lse = torch.logsumexp(log_comp, dim=1) - torch.log(torch.tensor(B, device=x.device, dtype=torch.float))  # [B,D]
            return lse.sum(-1)  # [B]
        log_prod_qz = log_sum_exp_marginals(z, mu, lv)
        # total correlation TC = KL(q(z) || prod_j q(z_j)) = E[log q(z) - sum_j log q(z_j)]
        tc = (log_qz - log_prod_qz).mean()
        # remaining KL to prior (standard normal) minus TC equals sum of dimensionwise KLs
        kl_dim = -0.5*(1 + lv - mu.pow(2) - lv.exp()).sum(1).mean()
        # Reconstruction
        if self.recon=='bce':
            rec = F.binary_cross_entropy_with_logits(x_logits, x, reduction='sum')/B
        else:
            rec = F.mse_loss(torch.sigmoid(x_logits), x, reduction='sum')/B
        # β-TC objective (α=γ=1): rec + (kl_dim) + β*TC
        return rec + kl_dim + self.beta * tc
