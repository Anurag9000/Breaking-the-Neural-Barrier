from dataclasses import dataclass
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VRNNConfig:
    input_dim: int = 96   # e.g., CIFAR rows (32x3)
    hidden_dim: int = 128
    latent_dim: int = 16
    seq_len: int = 32

# We treat each image as a sequence of 32 steps, each step is a 3x32 row flattened (96 dims)
class VRNN(nn.Module):
    def __init__(self, cfg: VRNNConfig):
        super().__init__()
        self.cfg=cfg
        self.phi_x = nn.Linear(cfg.input_dim, cfg.hidden_dim)
        self.phi_z = nn.Linear(cfg.latent_dim, cfg.hidden_dim)
        self.rnn = nn.GRU(cfg.hidden_dim*2, cfg.hidden_dim, batch_first=True)
        # prior p(z_t|h_{t-1})
        self.prior_mu = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        self.prior_lv = nn.Linear(cfg.hidden_dim, cfg.latent_dim)
        # enc q(z_t|x_t,h_{t-1})
        self.enc_mu = nn.Linear(cfg.hidden_dim*2, cfg.latent_dim)
        self.enc_lv = nn.Linear(cfg.hidden_dim*2, cfg.latent_dim)
        # dec p(x_t|z_t,h_{t-1})
        self.dec = nn.Linear(cfg.hidden_dim*2, cfg.input_dim)

    @staticmethod
    def reparameterize(mu, lv):
        std=(0.5*lv).exp(); return mu+torch.randn_like(std)*std

    def forward(self, x_seq):
        B,T,D = x_seq.size()
        h = torch.zeros(1, B, self.cfg.hidden_dim, device=x_seq.device)
        kls=[]; recons=[]
        for t in range(T):
            phi_x_t = torch.tanh(self.phi_x(x_seq[:,t,:]))
            # prior
            prior_mu_t = self.prior_mu(h[-1])
            prior_lv_t = self.prior_lv(h[-1])
            # encoder
            enc_in = torch.cat([phi_x_t, h[-1]], dim=-1)
            enc_mu_t = self.enc_mu(enc_in)
            enc_lv_t = self.enc_lv(enc_in)
            z_t = self.reparameterize(enc_mu_t, enc_lv_t)
            phi_z_t = torch.tanh(self.phi_z(z_t))
            # decoder
            dec_in = torch.cat([phi_z_t, h[-1]], dim=-1)
            x_hat_t = torch.sigmoid(self.dec(dec_in))
            # rnn step
            rnn_in = torch.cat([phi_x_t, phi_z_t], dim=-1).unsqueeze(1)
            _, h = self.rnn(rnn_in, h)
            # losses
            recon_t = F.binary_cross_entropy(x_hat_t, x_seq[:,t,:], reduction='none').sum(dim=1)
            kl_t = -0.5*torch.sum(1+enc_lv_t - (enc_mu_t-prior_mu_t).pow(2) - enc_lv_t.exp()/prior_lv_t.exp(), dim=1)
            recons.append(recon_t); kls.append(kl_t)
        recon = torch.stack(recons, dim=1).mean()
        kl = torch.stack(kls, dim=1).mean()
        loss = recon + kl
        return loss, recon, kl
