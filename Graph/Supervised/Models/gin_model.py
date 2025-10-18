import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, BatchNorm

class GIN(nn.Module):
    """
    Graph Isomorphism Network (GIN / GIN-ε via eps trainable flag in GINConv MLP).
    Uses per-layer MLPs inside GINConv. Output head is a linear classifier.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 5, dropout: float = 0.5, use_batchnorm: bool = True,
                 train_eps: bool = False):
        super().__init__()
        assert num_layers >= 2
        self.dropout = dropout
        self.use_bn = use_batchnorm
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batchnorm else None

        def mlp(c_in, c_out):
            return nn.Sequential(
                nn.Linear(c_in, c_out), nn.ReLU(inplace=True), nn.Linear(c_out, c_out)
            )

        # Input
        self.layers.append(GINConv(mlp(in_channels, hidden_channels), train_eps=train_eps))
        if self.use_bn:
            self.bns.append(BatchNorm(hidden_channels))
        # Hidden
        for _ in range(num_layers - 2):
            self.layers.append(GINConv(mlp(hidden_channels, hidden_channels), train_eps=train_eps))
            if self.use_bn:
                self.bns.append(BatchNorm(hidden_channels))
        # Output conv (keep conv form for consistency; linear head applied by caller if graph-level)
        self.out_lin = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for m in self.layers:
            m.reset_parameters()
        if self.bns is not None:
            for bn in self.bns:
                bn.reset_running_stats(); bn.reset_parameters()
        nn.init.xavier_uniform_(self.out_lin.weight); nn.init.zeros_(self.out_lin.bias)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out_lin(x)
        return x

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
