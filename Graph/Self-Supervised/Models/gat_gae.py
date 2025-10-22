import torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import negative_sampling

class GATEncoder(nn.Module):
    def __init__(self, in_dim, hid=128, heads=4, layers=2, dropout=0.2, out=64):
        super().__init__()
        self.layers = nn.ModuleList([GATv2Conv(in_dim if i==0 else hid, hid//heads, heads=heads, dropout=dropout) for i in range(layers)])
        self.mu = nn.Linear(hid, out); self.logvar = nn.Linear(hid, out); self.dropout=dropout
    def forward(self, x, ei):
        for c in self.layers:
            x = F.elu(c(x, ei)); x = F.dropout(x, p=self.dropout, training=self.training)
        return self.mu(x), self.logvar(x)

@dataclass
class Config:
    lr:float=1e-3; epochs:int=300; patience:int=40; kl_beta:float=1e-4; ckpt:str="ckpt_gat_vgae.pt"
    device:str="cuda" if torch.cuda.is_available() else "cpu"

class VGAE_GAT(nn.Module):
    def __init__(self, in_dim, cfg:Config):
        super().__init__()
        self.enc = GATEncoder(in_dim); self.cfg=cfg
    def reparam(self, mu, logv):
        std = torch.exp(0.5*logv); eps = torch.randn_like(std); return mu + eps*std
    def loss(self, data):
        mu, logv = self.enc(data.x, data.edge_index)
        z = self.reparam(mu, logv)
        pos = data.edge_index
        neg = negative_sampling(pos, num_nodes=data.x.size(0))
        pos_logit = (z[pos[0]]*z[pos[1]]).sum(dim=-1)
        neg_logit = (z[neg[0]]*z[neg[1]]).sum(dim=-1)
        recon = - (F.logsigmoid(pos_logit).mean() + F.logsigmoid(-neg_logit).mean())
        kl = -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())
        return recon + self.cfg.kl_beta*kl
