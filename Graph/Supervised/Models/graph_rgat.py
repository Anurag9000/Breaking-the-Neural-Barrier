import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGATConv

class RGATNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, out_dim=7, num_layers=2, heads=4, num_relations=1, dropout=0.5):
        super().__init__()
        assert num_layers>=2
        self.dropout=dropout
        self.layers = nn.ModuleList()
        self.layers.append(RGATConv(in_dim, hidden_dim, heads=heads, num_relations=num_relations, concat=True))
        for _ in range(num_layers-2):
            self.layers.append(RGATConv(hidden_dim*heads, hidden_dim, heads=heads, num_relations=num_relations, concat=True))
        self.out = RGATConv(hidden_dim*heads, out_dim, heads=1, num_relations=num_relations, concat=False)
    def forward(self, x, edge_index, edge_type=None):
        if edge_type is None:
            edge_type = x.new_zeros(edge_index.size(1), dtype=torch.long)
        for conv in self.layers:
            x = F.elu(conv(x, edge_index, edge_type))
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out(x, edge_index, edge_type)
