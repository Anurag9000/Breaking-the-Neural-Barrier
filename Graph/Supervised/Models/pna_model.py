import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PNAConv, BatchNorm
from torch_geometric.utils import degree

class PNA(nn.Module):
    """
    Principal Neighbourhood Aggregation (PNA).
    Requires in-degree statistics (deg histogram) to set scalers.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 aggregators=None, scalers=None, towers: int = 4, pre_layers: int = 1,
                 post_layers: int = 1, divide_input: bool = True, num_layers: int = 3,
                 dropout: float = 0.5, use_batchnorm: bool = True, deg=None):
        super().__init__()
        assert num_layers >= 2
        self.dropout = dropout
        self.use_bn = use_batchnorm

        if aggregators is None:
            aggregators = ['mean', 'min', 'max', 'std']
        if scalers is None:
            scalers = ['identity', 'amplification', 'attenuation']

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batchnorm else None

        self.convs.append(PNAConv(in_channels, hidden_channels, aggregators=aggregators,
                                  scalers=scalers, deg=deg, towers=towers,
                                  pre_layers=pre_layers, post_layers=post_layers,
                                  divide_input=divide_input))
        if self.use_bn:
            self.bns.append(BatchNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(PNAConv(hidden_channels, hidden_channels, aggregators=aggregators,
                                      scalers=scalers, deg=deg, towers=towers,
                                      pre_layers=pre_layers, post_layers=post_layers,
                                      divide_input=divide_input))
            if self.use_bn:
                self.bns.append(BatchNorm(hidden_channels))
        self.lin = nn.Linear(hidden_channels, out_channels)

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
