import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MixHopConv, BatchNorm

class MixHop(nn.Module):
    """
    MixHop: mixes multiple adjacency powers in the same layer.
    powers: list of hop exponents per layer (e.g., [0,1,2]).
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, powers=(0,1,2), dropout: float = 0.5,
                 use_batchnorm: bool = True):
        super().__init__()
        assert num_layers >= 2
        self.dropout = dropout
        self.use_bn = use_batchnorm
        self.powers = tuple(powers)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batchnorm else None

        # First MixHop layer
        self.convs.append(MixHopConv(in_channels, hidden_channels, self.powers))
        if self.use_bn:
            self.bns.append(BatchNorm(hidden_channels * len(self.powers)))
        in_dim = hidden_channels * len(self.powers)

        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(MixHopConv(in_dim, hidden_channels, self.powers))
            if self.use_bn:
                self.bns.append(BatchNorm(hidden_channels * len(self.powers)))
            in_dim = hidden_channels * len(self.powers)

        # Final linear head
        self.lin = nn.Linear(in_dim, out_channels)

    def reset_parameters(self):
        for c in self.convs:
            c.reset_parameters()
        if self.bns is not None:
            for bn in self.bns:
                bn.reset_running_stats(); bn.reset_parameters()
        nn.init.xavier_uniform_(self.lin.weight); nn.init.zeros_(self.lin.bias)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin(x)
        return x

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
