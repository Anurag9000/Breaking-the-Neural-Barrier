import torch, torch.nn as nn, torch.nn.functional as F, random
from dataclasses import dataclass
from torch_geometric.nn import TransformerConv, global_mean_pool
from adp_gat_graphcl import aug_edge_drop, aug_node_drop, aug_feat_mask, aug_subgraph, nt_xent
AUGS = [aug_edge_drop, aug_node_drop, aug_feat_mask, aug_subgraph]

class GraphTransformer(nn.Module):
    def __init__(self, in_dim, hid=128, out=128, heads=4, layers=3, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList([TransformerConv(in_dim if i==0 else hid, hid//heads, heads=heads, dropout=dropout) for i in range(layers)])
        self.proj = nn.Linear(hid, out); self.dropout = dropout
    def forward(self, data):
        x, ei = data.x, data.edge_index
        for conv in self.layers:
            x = F.relu(conv(x, ei))
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, getattr(data, 'batch', torch.zeros(x.size(0), dtype=torch.long, device=x.device)))
        return self.proj(g)

@dataclass
class Config:
    temperature:float=0.2
    lr:float=1e-3
    epochs:int=400
    patience:int=50
    ckpt:str="ckpt_gtr_graphcl.pt"
    device:str="cuda" if torch.cuda.is_available() else "cpu"

class GraphCL_GTR(nn.Module):
    def __init__(self, in_dim, cfg:Config):
        super().__init__()
        self.enc = GraphTransformer(in_dim)
        self.cfg = cfg
    def two_views(self, data):
        a1, a2 = random.choice(AUGS), random.choice(AUGS)
        return a1(data), a2(data)
    def contrastive_loss(self, data):
        v1, v2 = self.two_views(data)
        z1, z2 = self.enc(v1), self.enc(v2)
        return nt_xent(z1, z2, self.cfg.temperature)
