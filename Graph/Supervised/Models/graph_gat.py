import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GATNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=8, out_dim=7, num_layers=2, heads=8, dropout=0.6):
        super().__init__()
        assert num_layers >= 2, "num_layers >= 2 required"
        self.dropout = dropout
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout, concat=True))
        # hidden layers
        for _ in range(num_layers-2):
            self.layers.append(GATConv(hidden_dim*heads, hidden_dim, heads=heads, dropout=dropout, concat=True))
        # output layer (average over heads)
        self.out = GATConv(hidden_dim*heads, out_dim, heads=1, dropout=dropout, concat=False)

    def forward(self, x, edge_index):
        for conv in self.layers:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out(x, edge_index)
        return x
