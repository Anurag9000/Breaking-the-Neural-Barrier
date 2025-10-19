import torch, torch.nn as nn, torch.nn.functional as F, random
from dataclasses import dataclass
from adp_gat_graphcl import GATEncoder, aug_edge_drop, aug_node_drop, aug_feat_mask, aug_subgraph, nt_xent
AUGS = [aug_edge_drop, aug_node_drop, aug_feat_mask, aug_subgraph]

@dataclass
class Config:
    temperature:float=0.2; lr:float=1e-3; epochs:int=400; patience:int=50; ckpt:str="ckpt_gat_joao.pt"
    device:str="cuda" if torch.cuda.is_available() else "cpu"

class JOAO_GAT(nn.Module):
    def __init__(self, in_dim, cfg:Config):
        super().__init__()
        self.enc = GATEncoder(in_dim); self.cfg = cfg
        self.w = nn.Parameter(torch.ones(len(AUGS))/len(AUGS))
    def loss(self, data):
        p = torch.softmax(self.w, dim=0)
        a1 = random.choices(AUGS, weights=p.detach().cpu().tolist(), k=1)[0]
        a2 = random.choices(AUGS, weights=p.detach().cpu().tolist(), k=1)[0]
        z1 = self.enc(a1(data)); z2 = self.enc(a2(data))
        return nt_xent(z1, z2, self.cfg.temperature)
