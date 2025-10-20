from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VaDEConfig:
    in_channels: int = 3
    img_size: int = 32
    latent_dim: int = 16
    n_components: int = 10
    width: int = 128
    depth: int = 2


def fside(sz:int,d:int)->int: return max(1, sz//(2**d))

class Enc(nn.Module):
    def __init__(self,in_ch,w,d,latent):
        super().__init__()
        ch=w; layers=[nn.Conv2d(in_ch,ch,3,2,1), nn.BatchNorm2d(ch), nn.ReLU(True)]
        for _ in range(d-1): layers+=[nn.Conv2d(ch,ch,3,2,1), nn.BatchNorm2d(ch), nn.ReLU(True)]
        self.net=nn.Sequential(*layers)
        fs=fside(32,d); self.mu=nn.Linear(ch*fs*fs, latent); self.lv=nn.Linear(ch*fs*fs, latent)
    def forward(self,x):
        h=self.net(x).flatten(1); return self.mu(h), self.lv(h)

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

class VaDE(nn.Module):
    def __init__(self, cfg: VaDEConfig):
        super().__init__()
        self.cfg=cfg
        self.enc=Enc(cfg.in_channels,cfg.width,cfg.depth,cfg.latent_dim)
        self.dec=Dec(cfg.in_channels,cfg.width,cfg.depth,cfg.latent_dim)
        self.pi = nn.Parameter(torch.ones(cfg.n_components)/cfg.n_components)
        self.mu_c = nn.Parameter(torch.randn(cfg.n_components, cfg.latent_dim))
        self.logvar_c = nn.Parameter(torch.zeros(cfg.n_components, cfg.latent_dim))

    @staticmethod
    def reparameterize(mu, lv):
        std=(0.5*lv).exp(); return mu+torch.randn_like(std)*std

    def forward(self,x):
        mu,lv=self.enc(x)
        z=self.reparameterize(mu,lv)
        xr=self.dec(z)
        # responsibilities q(c|x)
        log_pi = torch.log_softmax(self.pi, dim=0)
        # log N(z; mu_c, var_c)
        log_p_z_c = -0.5*((z.unsqueeze(1)-self.mu_c)**2/ self.logvar_c.exp().unsqueeze(0)
                          + self.logvar_c.exp().log().unsqueeze(0)).sum(dim=2)
        log_p_z_c = log_p_z_c + log_pi.unsqueeze(0)
        q_c_x = torch.softmax(log_p_z_c, dim=1)
        return xr, mu, lv, z, q_c_x

    def loss_fn(self,x,xr,mu,lv,z,q_c_x) -> Dict[str,torch.Tensor]:
        recon=F.binary_cross_entropy(xr,x,reduction='mean')
        # KL(q(z,c|x) || p(z,c)) = KL(q(z|x)||p(z)) + KL(q(c|x)||p(c|z)) simplified via VaDE objective
        # Use expected KL to GMM prior
        # E_q[c|x] [ KL(q(z|x) || N(mu_c, var_c)) ] + KL(q(c|x)||pi)
        var_c = self.logvar_c.exp()
        kl_z = 0.5 * (
            ((lv.exp().unsqueeze(1) + (mu.unsqueeze(1)-self.mu_c)**2) / var_c.unsqueeze(0) - 1
             + (self.logvar_c.unsqueeze(0) - lv.unsqueeze(1))).sum(dim=2)
        )
        kl_z = (q_c_x * kl_z).sum(dim=1).mean()
        kl_c = (q_c_x * (q_c_x.clamp_min(1e-9).log() - torch.log_softmax(self.pi, dim=0).unsqueeze(0))).sum(dim=1).mean()
        loss = recon + kl_z + kl_c
        return {"loss":loss, "recon":recon, "kl_z":kl_z, "kl_c":kl_c}
