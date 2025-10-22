import torch, torch.nn as nn, torch.nn.functional as F, random
from dataclasses import dataclass
from adp_gat_graphcl import GATEncoder, aug_edge_drop, aug_node_drop, aug_feat_mask, aug_subgraph
AUGS=[aug_edge_drop, aug_node_drop, aug_feat_mask, aug_subgraph]

def barlow_loss(z1, z2, lambd=5e-3):
    z1 = (z1 - z1.mean(0)) / (z1.std(0)+1e-9)
    z2 = (z2 - z2.mean(0)) / (z2.std(0)+1e-9)
    N = z1.size(0)
    c = (z1.T @ z2) / N
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = (c - torch.diag(torch.diagonal(c))).pow(2).sum()
    return on_diag + lambd * off_diag

@dataclass
class Config:
    lr:float=1e-3; epochs:int=400; patience:int=50; ckpt:str="ckpt_gat_barlow.pt"
    device:str="cuda" if torch.cuda.is_available() else "cpu"

class Barlow_GAT(nn.Module):
    def __init__(self, in_dim, cfg:Config):
        super().__init__(); self.enc=GATEncoder(in_dim); self.cfg=cfg
    def loss(self, data):
        a1, a2 = random.choice(AUGS), random.choice(AUGS)
        z1, z2 = self.enc(a1(data)), self.enc(a2(data))
        return barlow_loss(z1, z2)
