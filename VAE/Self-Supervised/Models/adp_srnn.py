from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SRNNConfig:
    input_dim: int = 96
    hidden_dim: int = 128
    z_dim: int = 16
    seq_len: int = 32

class SRNN(nn.Module):
    def __init__(self, cfg: SRNNConfig):
        super().__init__()
        self.cfg=cfg
        self.rnn = nn.GRU(cfg.input_dim + cfg.z_dim, cfg.hidden_dim, batch_first=True)
        self.prior_mu = nn.Linear(cfg.hidden_dim, cfg.z_dim)
        self.prior_lv = nn.Linear(cfg.hidden_dim, cfg.z_dim)
        self.enc_mu = nn.Linear(cfg.hidden_dim + cfg.input_dim, cfg.z_dim)
        self.enc_lv = nn.Linear(cfg.hidden_dim + cfg.input_dim, cfg.z_dim)
        self.dec = nn.Linear(cfg.hidden_dim + cfg.z_dim, cfg.input_dim)

    @staticmethod
    def reparameterize(mu, lv):
        std=(0.5*lv).exp(); return mu+torch.randn_like(std)*std

    def forward(self, x_seq):
        B,T,D = x_seq.size()
        h = torch.zeros(1,B,self.cfg.hidden_dim, device=x_seq.device)
        z_t = torch.zeros(B, self.cfg.z_dim, device=x_seq.device)
        kls=[]; recons=[]
        for t in range(T):
            prior_mu = self.prior_mu(h[-1])
            prior_lv = self.prior_lv(h[-1])
            enc_in = torch.cat([x_seq[:,t,:], h[-1]], dim=-1)
            enc_mu = self.enc_mu(enc_in)
            enc_lv = self.enc_lv(enc_in)
            z_t = self.reparameterize(enc_mu, enc_lv)
            dec_in = torch.cat([h[-1], z_t], dim=-1)
            x_hat = torch.sigmoid(self.dec(dec_in))
            recon_t = F.binary_cross_entropy(x_hat, x_seq[:,t,:], reduction='none').sum(dim=1)
            kl_t = -0.5*torch.sum(1+enc_lv - (enc_mu-prior_mu).pow(2) - enc_lv.exp()/prior_lv.exp(), dim=1)
            recons.append(recon_t); kls.append(kl_t)
            rnn_in = torch.cat([x_seq[:,t,:], z_t], dim=-1).unsqueeze(1)
            _, h = self.rnn(rnn_in, h)
        recon = torch.stack(recons,1).mean(); kl = torch.stack(kls,1).mean(); loss=recon+kl
        return loss, recon, kl
