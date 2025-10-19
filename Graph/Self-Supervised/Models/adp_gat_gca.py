import torch, torch.nn as nn, torch.nn.functional as F, random
from dataclasses import dataclass
from adp_gat_graphcl import GATEncoder, aug_edge_drop, aug_node_drop, aug_feat_mask, aug_subgraph, nt_xent
AUGS = [aug_edge_drop, aug_node_drop, aug_feat_mask, aug_subgraph]

@dataclass
class Config:
    temperature:float=0.2; lr:float=1e-3; epochs:int=400; patience:int=50; ckpt:str="ckpt_gat_gca.pt"
    device:str="cuda" if torch.cuda.is_available() else "cpu"

class GCA_GAT(nn.Module):
    def __init__(self, in_dim, cfg:Config):
        super().__init__()
        self.enc = GATEncoder(in_dim)
        self.cfg = cfg
        self.strength = nn.Parameter(torch.tensor([0.2,0.2,0.2,0.8]))  # adaptive probs
    def pick_aug(self):
        p = torch.softmax(self.strength, dim=0).detach().cpu().numpy()
        a1, a2 = random.choices(AUGS, weights=p, k=2)
        return a1, a2
    def loss(self, data):
        a1, a2 = self.pick_aug()
        v1, v2 = a1(data), a2(data)
        z1, z2 = self.enc(v1), self.enc(v2)
        return nt_xent(z1, z2, self.cfg.temperature)
