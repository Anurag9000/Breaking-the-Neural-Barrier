import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCN2Conv, BatchNorm

class GCNII(nn.Module):
    """
    GCNII: deep GCN with initial residual and identity mapping.
    Uses GCN2Conv stacked many times.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 64, alpha: float = 0.1, theta: float = 0.5,
                 dropout: float = 0.5, use_batchnorm: bool = True):
        super().__init__()
        assert num_layers >= 2
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batchnorm else None
        self.use_bn = use_batchnorm

        # Input projection
        self.lin_in = nn.Linear(in_channels, hidden_channels)
        # Middle stacked GCN2Conv layers
        for _ in range(num_layers - 2):
            self.convs.append(GCN2Conv(hidden_channels, alpha=alpha, theta=theta, layer=None))
            if self.use_bn:
                self.bns.append(BatchNorm(hidden_channels))
        # Output projection
        self.lin_out = nn.Linear(hidden_channels, out_channels)

        self.alpha = alpha
        self.theta = theta
        self.num_layers = num_layers

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_in.weight); nn.init.zeros_(self.lin_in.bias)
        nn.init.xavier_uniform_(self.lin_out.weight); nn.init.zeros_(self.lin_out.bias)
        for c in self.convs:
            c.reset_parameters()
        if self.bns is not None:
            for bn in self.bns:
                bn.reset_running_stats(); bn.reset_parameters()

    def forward(self, x, edge_index):
        x0 = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin_in(x)
        x = F.relu(x)
        h = x
        # GCN2 layers keep reference to x0 via initial residual
        for i, conv in enumerate(self.convs):
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = conv(h, x, edge_index)  # conv(h, x0, edge)
            if self.use_bn:
                h = self.bns[i](h)
            h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.lin_out(h)
        return out

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
