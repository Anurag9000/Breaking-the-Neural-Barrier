import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

class GATv2Net(nn.Module):
    def __init__(self, in_dim, hidden_dim=16, out_dim=7, num_layers=2, heads=4, dropout=0.5):
        super().__init__()
        assert num_layers >= 2
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.layers.append(GATv2Conv(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True, share_weights=False))
        for _ in range(num_layers-2):
            self.layers.append(GATv2Conv(hidden_dim*heads, hidden_dim, heads=heads, dropout=dropout, concat=True, share_weights=False))
        self.out = GATv2Conv(hidden_dim*heads, out_dim, heads=1, dropout=dropout, concat=False, share_weights=False)

    def forward(self, x, edge_index):
        for conv in self.layers:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out(x, edge_index)
