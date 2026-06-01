import torch
import torch.nn as nn
import torch.nn.functional as F

# Mixture-of-Logistics Likelihood VAE (K components per pixel)

class Encoder(nn.Module):
    def __init__(self, in_ch=1, z_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(True),
        )
        self.mu = nn.Linear(128*4*4, z_dim)
        self.lv = nn.Linear(128*4*4, z_dim)
    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        return self.mu(h), self.lv(h)

class DecoderMoL(nn.Module):
    def __init__(self, out_ch=1, z_dim=32, K=5):
        super().__init__()
        self.K = K
        self.fc = nn.Linear(z_dim, 128*4*4)
        self.de = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),
        )
        # final conv to output per-pixel params: for 1 channel => K mus, K log_scales, K logits
        self.out = nn.Conv2d(32, out_ch * (3*K), 3, 1, 1)
    def forward(self, z):
        h = self.de(self.fc(z).view(z.size(0), 128, 4, 4))
        params = self.out(h)
        return params

class VAE_MoL(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, z_dim=32, K=5):
        super().__init__()
        self.enc = Encoder(in_ch, z_dim)
        self.dec = DecoderMoL(out_ch, z_dim, K)
        self.K = K
        self.delta = 1.0/256.0
    @staticmethod
    def reparam(mu, lv):
        std = (0.5*lv).exp(); eps = torch.randn_like(std); return mu + eps*std
    def mol_log_prob(self, x, params):
        B,C,H,W = x.shape
        K = self.K
        params = params.view(B, C, 3*K, H, W)
        mu, log_s, logit_pi = torch.split(params, [K, K, K], dim=2)
        s = torch.exp(log_s)
        pi = torch.softmax(logit_pi, dim=2)
        x = x.unsqueeze(2)  # [B,C,1,H,W]
        # discretized logistic per component
        invs = 1.0 / s
        plus = invs*(x - mu + 0.5*self.delta)
        minus = invs*(x - mu - 0.5*self.delta)
        cdf_plus = torch.sigmoid(plus)
        cdf_minus = torch.sigmoid(minus)
        probs = (cdf_plus - cdf_minus).clamp(min=1e-7)
        log_comp = torch.log(probs)
        log_mix = torch.logsumexp(torch.log(pi + 1e-8) + log_comp, dim=2)  # mix over K
        return log_mix.sum(dim=[1,2,3])  # sum over C,H,W
    def forward(self, x):
        mu_z, lv_z = self.enc(x)
        z = self.reparam(mu_z, lv_z)
        params = self.dec(z)
        log_px = self.mol_log_prob(x, params)
        kl = -0.5*(1 + lv_z - mu_z.pow(2) - lv_z.exp()).sum(1)
        return (-(log_px) + kl).mean()
