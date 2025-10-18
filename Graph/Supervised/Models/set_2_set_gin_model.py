import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, Set2Set, global_mean_pool, BatchNorm

class MLP(nn.Sequential):
    def __init__(self, c_in, c_out):
        super().__init__(
            nn.Linear(c_in, c_out), nn.ReLU(inplace=True), nn.Linear(c_out, c_out)
        )

class GIN_Set2Set(nn.Module):
    """
    GIN backbone with Set2Set readout for graph classification.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=5, dropout=0.5, use_batchnorm=True):
        super().__init__()
        assert num_layers >= 2
        self.dropout = dropout
        self.use_bn = use_batchnorm

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batchnorm else None

        self.layers.append(GINConv(MLP(in_channels, hidden_channels)))
        if self.use_bn:
            self.bns.append(BatchNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(GINConv(MLP(hidden_channels, hidden_channels)))
            if self.use_bn:
                self.bns.append(BatchNorm(hidden_channels))

        self.set2set = Set2Set(hidden_channels, processing_steps=3)
        self.lin = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def reset_parameters(self):
        for m in self.layers:
            m.reset_parameters()
        if self.bns is not None:
            for bn in self.bns:
                bn.reset_running_stats(); bn.reset_parameters()
        self.set2set.reset_parameters()
        for m in self.lin:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = self.set2set(x, batch)
        out = self.lin(g)
        return out

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
