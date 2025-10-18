import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, JumpingKnowledge, BatchNorm

class JKNet(nn.Module):
    """
    Jumping Knowledge Network with GCN backbone.
    Modes: 'cat' (concatenate), 'max', or 'lstm' aggregation.
    Single-model, end-to-end.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 4, jk_mode: str = 'cat', dropout: float = 0.5,
                 use_batchnorm: bool = True):
        super().__init__()
        assert num_layers >= 2
        assert jk_mode in {'cat', 'max', 'lstm'}
        self.dropout = dropout
        self.use_bn = use_batchnorm

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batchnorm else None

        # First layer
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True, normalize=True))
        if self.use_bn:
            self.bns.append(BatchNorm(hidden_channels))
        # Hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True, normalize=True))
            if self.use_bn:
                self.bns.append(BatchNorm(hidden_channels))

        self.jk = JumpingKnowledge(mode=jk_mode, channels=hidden_channels, num_layers=num_layers)
        out_in = hidden_channels * num_layers if jk_mode == 'cat' else hidden_channels
        self.lin = nn.Linear(out_in, out_channels)

    def reset_parameters(self):
        for c in self.convs:
            c.reset_parameters()
        if self.bns is not None:
            for bn in self.bns:
                bn.reset_running_stats(); bn.reset_parameters()
        if hasattr(self.jk, 'reset_parameters'):
            self.jk.reset_parameters()
        nn.init.xavier_uniform_(self.lin.weight); nn.init.zeros_(self.lin.bias)

    def forward(self, x, edge_index):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            xs.append(x)
        x = self.jk(xs)
        x = self.lin(x)
        return x

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
