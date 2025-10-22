import torch, torch.nn as nn, torch.nn.functional as F, random, copy
from dataclasses import dataclass
from adp_gat_graphcl import GATEncoder, aug_edge_drop, aug_node_drop, aug_feat_mask, aug_subgraph, nt_xent
AUGS=[aug_edge_drop, aug_node_drop, aug_feat_mask, aug_subgraph]

@dataclass
class Config:
    weight_noise:float=0.05; temperature:float=0.2; lr:float=1e-3; epochs:int=400; patience:int=50; ckpt:str="ckpt_gat_simgrace.pt"
    device:str="cuda" if torch.cuda.is_available() else "cpu"

def perturb_weights(model, sigma):
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.add_(sigma*torch.randn_like(p))

class SimGRACE_GAT(nn.Module):
    def __init__(self, in_dim, cfg:Config):
        super().__init__(); self.enc=GATEncoder(in_dim); self.cfg=cfg
    def loss(self, data):
        a1, a2 = random.choice(AUGS), random.choice(AUGS)
        v1, v2 = a1(data), a2(data)
        # view 1: encode with perturbed weights
        backup = copy.deepcopy(self.enc.state_dict())
        perturb_weights(self.enc, self.cfg.weight_noise)
        z1 = self.enc(v1)
        self.enc.load_state_dict(backup)
        z2 = self.enc(v2)
        return nt_xent(z1, z2, self.cfg.temperature)
