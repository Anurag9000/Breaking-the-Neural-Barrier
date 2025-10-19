import torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass
from torch_geometric.nn import GATv2Conv, global_mean_pool

class GAT(nn.Module):
    def __init__(self, in_dim, hid=128, out=128, heads=4, layers=2, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList([GATv2Conv(in_dim if i==0 else hid, hid//heads, heads=heads, dropout=dropout) for i in range(layers)])
        self.readout = nn.Linear(hid, out)
        self.dropout = dropout
    def forward(self, data):
        x, ei = data.x, data.edge_index
        for c in self.convs:
            x = F.elu(c(x, ei)); x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, getattr(data, 'batch', torch.zeros(x.size(0), dtype=torch.long, device=x.device)))
        return x, self.readout(g)

# Mutual information estimator (InfoNCE-style between node and graph summaries)
def local_global_infomax(h_node, h_graph):
    h_node = F.normalize(h_node, dim=-1)
    h_graph = F.normalize(h_graph, dim=-1)
    pos = torch.sum(h_node * h_graph[h_node.new_zeros(h_node.size(0), dtype=torch.long)], dim=-1)  # single-graph batch
    # negatives: shuffle graph vector
    neg_g = h_graph[torch.randperm(h_graph.size(0))]
    neg = torch.sum(h_node * neg_g[0].expand_as(h_node), dim=-1)
    logits = torch.stack([pos, neg], dim=1)
    labels = torch.zeros(h_node.size(0), dtype=torch.long, device=h_node.device)
    return F.cross_entropy(logits, labels)

@dataclass
class Config:
    lr:float=1e-3; epochs:int=300; patience:int=40; ckpt:str="ckpt_gat_infograph.pt"
    device:str="cuda" if torch.cuda.is_available() else "cpu"

class InfoGraph_GAT(nn.Module):
    def __init__(self, in_dim, cfg:Config):
        super().__init__()
        self.enc = GAT(in_dim); self.cfg = cfg
    def loss(self, data):
        h_node, h_graph = self.enc(data)
        return local_global_infomax(h_node, h_graph)
