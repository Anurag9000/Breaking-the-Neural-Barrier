import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_mean_pool, BatchNorm

class RGCNNet(nn.Module):
    """
    Relational Graph Convolutional Network (R-GCN) for node/graph classification.
    Single-model, basis-decomposed weights.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations,
                 num_layers=3, num_bases=None, dropout=0.5, use_batchnorm=True, graph_level=False):
        super().__init__()
        assert num_layers >= 2
        self.dropout = dropout
        self.graph_level = graph_level
        self.use_bn = use_batchnorm

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batchnorm else None

        self.layers.append(RGCNConv(in_channels, hidden_channels, num_relations, num_bases=num_bases))
        if self.use_bn:
            self.bns.append(BatchNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(RGCNConv(hidden_channels, hidden_channels, num_relations, num_bases=num_bases))
            if self.use_bn:
                self.bns.append(BatchNorm(hidden_channels))
        self.layers.append(RGCNConv(hidden_channels, out_channels, num_relations, num_bases=num_bases))

        if graph_level:
            self.lin = nn.Sequential(
                nn.Linear(out_channels, hidden_channels), nn.ReLU(inplace=True), nn.Dropout(dropout),
                nn.Linear(hidden_channels, out_channels)
            )
        else:
            self.lin = nn.Identity()

    def reset_parameters(self):
        for c in self.layers:
            c.reset_parameters()
        if self.bns is not None:
            for bn in self.bns:
                bn.reset_running_stats(); bn.reset_parameters()
        if isinstance(self.lin, nn.Sequential):
            for m in self.lin:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, edge_type, batch=None):
        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, edge_index, edge_type)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x, edge_index, edge_type)
        if self.graph_level and batch is not None:
            g = global_mean_pool(x, batch)
            x = self.lin(g)
        return x

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
