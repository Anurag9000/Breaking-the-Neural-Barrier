import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm

class GraphSAGE_Mean(nn.Module):
    """
    GraphSAGE with mean aggregator (single-model, inductive).
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, dropout: float = 0.5, use_batchnorm: bool = True):
        super().__init__()
        assert num_layers >= 2
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batchnorm else None
        self.use_bn = use_batchnorm
        self.dropout = dropout

        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr='mean'))
        if self.use_bn:
            self.bns.append(BatchNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr='mean'))
            if self.use_bn:
                self.bns.append(BatchNorm(hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr='mean'))

    def reset_parameters(self):
        for m in self.convs:
            m.reset_parameters()
        if self.bns is not None:
            for bn in self.bns:
                bn.reset_running_stats(); bn.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
