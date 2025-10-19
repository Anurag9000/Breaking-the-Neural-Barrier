import torch
import torch.nn as nn
from torch_geometric.nn import HGTConv

class HGTNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=7, heads=4, layers=2, metadata=(['n'],[('n','e','n')]), dropout=0.5):
        super().__init__()
        self.lin_in = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([HGTConv(hidden_dim, hidden_dim, metadata, heads=heads) for _ in range(layers-1)])
        self.lin_out = nn.Linear(hidden_dim, out_dim)
        self.dropout=dropout; self.metadata=metadata
    def forward(self, x_dict, edge_index_dict):
        x = {'n': self.lin_in(x_dict['n'])}
        for conv in self.layers:
            x = conv(x, edge_index_dict)
            x = {k: torch.relu(v) for k,v in x.items()}
        return self.lin_out(x['n'])
