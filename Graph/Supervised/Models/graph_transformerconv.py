import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv

class TransformerConvNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=7, num_layers=2, heads=4, dropout=0.5):
        super().__init__()
        assert num_layers >= 2
        self.dropout = dropout
        self.layers = nn.ModuleList()
        self.layers.append(TransformerConv(in_channels=in_dim, out_channels=hidden_dim, heads=heads, dropout=dropout, beta=False))
        for _ in range(num_layers-2):
            self.layers.append(TransformerConv(in_channels=hidden_dim*heads, out_channels=hidden_dim, heads=heads, dropout=dropout, beta=False))
        self.out = TransformerConv(in_channels=hidden_dim*heads, out_channels=out_dim, heads=1, dropout=dropout, beta=False)

    def forward(self, x, edge_index):
        for conv in self.layers:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.out(x, edge_index)
