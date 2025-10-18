import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm

class GCN(nn.Module):
    """
    Canonical 2–8 layer GCN (Kipf & Welling) for node/graph classification.
    - Single-model, end-to-end; no EMA/teacher.
    - Optionally inserts BatchNorm + Dropout per layer.
    - Global pooling left to runner for graph-level tasks.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, dropout: float = 0.5, use_batchnorm: bool = True):
        super().__init__()
        assert num_layers >= 2, "GCN num_layers must be >= 2"
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batchnorm else None
        self.dropout = dropout
        self.use_bn = use_batchnorm

        # input layer
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=True, normalize=True))
        if self.use_bn:
            self.bns.append(BatchNorm(hidden_channels))
        # hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True, normalize=True))
            if self.use_bn:
                self.bns.append(BatchNorm(hidden_channels))
        # output layer
        self.convs.append(GCNConv(hidden_channels, out_channels, cached=True, normalize=True))

        # Kaiming init for internal linear weights used by GCNConv is handled internally.

    def reset_parameters(self):
        for m in self.convs:
            m.reset_parameters()
        if self.bns is not None:
            for bn in self.bns:
                bn.reset_running_stats()
                bn.reset_parameters()

    def forward(self, x, edge_index):
        # All layers except last: ReLU(+BN)+Dropout
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
