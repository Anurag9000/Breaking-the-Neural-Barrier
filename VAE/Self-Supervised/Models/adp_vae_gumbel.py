from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GumbelVAEConfig:
    in_channels: int = 3
    img_size: int = 32
    categories: int = 10   # number of categories per latent variable
    n_latents: int = 32    # number of categorical latents
    width: int = 128
    depth: int = 2
    tau_start: float = 1.0
    tau_min: float = 0.5


def fside(sz:int,d:int)->int: return max(1, sz//(2**d))

def sample_gumbel_softmax(logits: torch.Tensor, tau: float, hard: bool = True) -> torch.Tensor:
    U = torch.rand_like(logits)
    g = -torch.log(-torch.log(U + 1e-9) + 1e-9)
    y = F.softmax((logits + g) / tau, dim=-1)
    if hard:
        y_hard = torch.zeros_like(y)
        y_hard.scatter_(-1, y.argmax(dim=-1, keepdim=True), 1.0)
        y = (y_hard - y).detach() + y
    return y

class Enc(nn.Module):
    def __init__(self,in_ch,w,d,cats,nlat):
        super().__init__()
        ch=w; layers=[nn.Conv2d(in_ch,ch,3,2,1), nn.BatchNorm2d(ch), nn.ReLU(True)]
        for _ in range(d-1): layers+=[nn.Conv2d(ch,ch,3,2,1), nn.BatchNorm2d(ch), nn.ReLU(True)]
        fs=fside(32,d); self.net=nn.Sequential(*layers)
        self.logits = nn.Linear(ch*fs*fs, nlat*cats)
        self.nlat=nlat; self.cats=cats
    def forward(self,x):
        h=self.net(x); h=h.flatten(1); logits=self.logits(h)
        return logits.view(x.size(0), self.nlat, self.cats)

class Dec(nn.Module):
    def __init__(self,out_ch,w,d,cats,nlat):
        super().__init__()
        in_ch = nlat * cats
        ch=w
        layers=[nn.Conv2d(in_ch, ch, 1), nn.ReLU(True)]
        for _ in range(d-1): layers+=[nn.ConvTranspose2d(ch,ch,4,2,1), nn.BatchNorm2d(ch), nn.ReLU(True)]
        layers+=[nn.ConvTranspose2d(ch, out_ch, 4, 2, 1)]
        self.net=nn.Sequential(*layers)
    def forward(self, y_onehot):
        # y_onehot: (B, nlat, cats) -> map to (B, nlat*cats, 1, 1) then upsample via convT
        B, L, C = y_onehot.shape
        z = y_onehot.view(B, L*C, 1, 1)
        return torch.sigmoid(self.net(z))

class GumbelVAE(nn.Module):
    def __init__(self, cfg: GumbelVAEConfig):
        super().__init__()
        self.cfg=cfg; self._tau=cfg.tau_start
        self.enc=Enc(cfg.in_channels,cfg.width,cfg.depth,cfg.categories,cfg.n_latents)
        self.dec=Dec(cfg.in_channels,cfg.width,cfg.depth,cfg.categories,cfg.n_latents)

    def set_tau(self, tau: float):
        self._tau = max(self.cfg.tau_min, tau)

    def forward(self,x, hard: bool=True):
        logits=self.enc(x)
        y=sample_gumbel_softmax(logits, self._tau, hard=hard)  # (B,L,C)
        xr=self.dec(y)
        return xr, logits, y

    def loss_fn(self, x, xr, logits, y) -> Dict[str, torch.Tensor]:
        recon=F.binary_cross_entropy(xr, x, reduction='mean')
        # Categorical KL (against uniform prior over categories)
        q = F.softmax(logits, dim=-1)
        log_q = torch.log(q + 1e-9)
        kl = (q * (log_q - torch.log(torch.tensor(1.0/self.cfg.categories, device=q.device)))).sum(dim=[1,2]).mean()
        loss = recon + kl
        return {"loss":loss, "recon":recon, "kl":kl, "tau": torch.tensor(self._tau, device=x.device)}
