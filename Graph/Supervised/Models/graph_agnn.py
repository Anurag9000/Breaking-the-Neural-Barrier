import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import AGNNConv

class AGNNNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=7, num_layers=2, dropout=0.5):
        super().__init__()
        assert num_layers >= 2
        self.dropout = dropout
        self.lin_in = nn.Linear(in_dim, hidden_dim)
        self.props = nn.ModuleList([AGNNConv(require_grad=True) for _ in range(num_layers-1)])
        self.lin_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin_in(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        for conv in self.props:
            x = conv(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin_out(x)
