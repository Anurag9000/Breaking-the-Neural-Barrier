import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class DeepGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim=32, out_dim=7, num_layers=6, heads=4, dropout=0.5):
        super().__init__()
        self.dropout=dropout
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        dim_in = in_dim
        for i in range(num_layers-1):
            conv = GATConv(dim_in, hidden_dim, heads=heads, concat=True, dropout=dropout)
            self.layers.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim*heads))
            dim_in = hidden_dim*heads
        self.out = GATConv(dim_in, out_dim, heads=1, concat=False, dropout=dropout)
    def forward(self, x, edge_index):
        for conv, ln in zip(self.layers, self.norms):
            h = conv(x, edge_index)
            h = F.elu(h)
            x = ln(x + F.dropout(h, p=self.dropout, training=self.training))
        return self.out(x, edge_index)
