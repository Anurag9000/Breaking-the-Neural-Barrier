from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VampPriorConfig:
    in_channels: int = 3
    img_size: int = 32
    latent_dim: int = 32
    width: int = 128
    depth: int = 2
    n_pseudo_inputs: int = 500


def fside(sz:int,d:int)->int: return max(1, sz//(2**d))

class Enc(nn.Module):
    def __init__(self,in_ch,w,d,latent):
        super().__init__()
        ch=w; layers=[nn.Conv2d(in_ch,ch,3,2,1), nn.BatchNorm2d(ch), nn.ReLU(True)]
        for _ in range(d-1): layers+=[nn.Conv2d(ch,ch,3,2,1), nn.BatchNorm2d(ch), nn.ReLU(True)]
        self.net=nn.Sequential(*layers)
        fs=fside(32,d); self.mu=nn.Linear(ch*fs*fs, latent); self.lv=nn.Linear(ch*fs*fs, latent)
        self.fs=fs; self.ch=ch
    def forward(self,x):
        h=self.net(x); h=h.flatten(1); return self.mu(h), self.lv(h)

class Dec(nn.Module):
    def __init__(self,out_ch,w,d,latent):
        super().__init__()
        fs=fside(32,d); self.fc=nn.Linear(latent, w*fs*fs)
        ups=[]
        for _ in range(d-1): ups+=[nn.ConvTranspose2d(w,w,4,2,1), nn.BatchNorm2d(w), nn.ReLU(True)]
        ups+=[nn.ConvTranspose2d(w,out_ch,4,2,1)]
        self.ups=nn.Sequential(*ups)
        self.w=w; self.fs=fs
    def forward(self,z):
        h=self.fc(z); h=h.view(z.size(0), self.w, self.fs, self.fs)
        return torch.sigmoid(self.ups(h))

class VampPriorVAE(nn.Module):
    def __init__(self, cfg: VampPriorConfig):
        super().__init__()
        self.cfg=cfg
        self.enc=Enc(cfg.in_channels,cfg.width,cfg.depth,cfg.latent_dim)
        self.dec=Dec(cfg.in_channels,cfg.width,cfg.depth,cfg.latent_dim)
        # pseudo inputs in image space (learnable)
        self.pseudo = nn.Parameter(torch.randn(cfg.n_pseudo_inputs, cfg.in_channels, cfg.img_size, cfg.img_size))

    @staticmethod
    def reparameterize(mu, lv):
        std=(0.5*lv).exp(); return mu+torch.randn_like(std)*std

    def forward(self,x):
        mu,lv=self.enc(x); z=self.reparameterize(mu,lv); xr=self.dec(z)
        return xr, mu, lv

    def _log_normal(self, z, mu, lv):
        return -0.5*((z-mu)**2/ lv.exp() + lv + torch.log(torch.tensor(2*3.1415926535, device=z.device))).sum(dim=1)

    def loss_fn(self, x, xr, mu, lv) -> Dict[str, torch.Tensor]:
        recon=F.binary_cross_entropy(xr, x, reduction='mean')
        # VampPrior: p(z) = 1/K sum_k q(z|x_tilde_k)
        with torch.no_grad():
            # stop gradients in computing the prior mixture responsibilities
            pass
        # compute q(z|pseudo) parameters
        with torch.enable_grad():
            # allow gradients to flow to pseudo inputs (they are learnable)
            pseudo = self.pseudo
            pmu, plv = self.enc(pseudo)
        # Monte Carlo: use z sampled from q(z|x), compute log p(z)
        # log p(z) = log (1/K sum_k exp(log q(z|x_tilde_k)))
        # where log q is Normal with (pmu_k, plv_k)
        z = self.reparameterize(mu, lv)
        # compute log q(z|x_tilde_k) for each k via broadcasting
        # z: (B,D) -> (B,1,D); pmu/plv: (K,D) -> (1,K,D)
        B, D = z.size(); K = pmu.size(0)
        z_b = z.unsqueeze(1)
        pmu_b = pmu.unsqueeze(0)
        plv_b = plv.unsqueeze(0)
        log_q_z_given_pseudo = -0.5 * (((z_b - pmu_b) ** 2) / plv_b.exp() + plv_b + torch.log(torch.tensor(2*3.1415926535, device=z.device)))
        log_q_z_given_pseudo = log_q_z_given_pseudo.sum(dim=2)  # (B,K)
        log_pz = torch.logsumexp(log_q_z_given_pseudo - torch.log(torch.tensor(K, device=z.device, dtype=z.dtype)), dim=1)
        # KL = E_q [log q(z|x) - log p(z)]
        log_qz_x = self._log_normal(z, mu, lv)
        kl = (log_qz_x - log_pz).mean()
        loss = recon + kl
        return {"loss":loss, "recon":recon, "kl":kl}
