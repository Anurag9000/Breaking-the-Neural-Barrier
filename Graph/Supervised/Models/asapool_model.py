import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ASAPooling, global_mean_pool, BatchNorm

class ASAPNet(nn.Module):
    """
    ASAPooling-based graph classifier (single model).
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_blocks=3, ratio=0.5, dropout=0.5, use_batchnorm=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        c_in = in_channels
        for i in range(num_blocks):
            conv = GCNConv(c_in, hidden_channels)
            bn = BatchNorm(hidden_channels) if use_batchnorm else None
            pool = ASAPooling(hidden_channels, ratio=ratio)
            self.blocks.append(nn.ModuleDict({'conv': conv, 'bn': bn, 'pool': pool}))
            c_in = hidden_channels
        self.lin = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
        self.dropout = dropout
        self.use_bn = use_batchnorm

    def reset_parameters(self):
        for b in self.blocks:
            b['conv'].reset_parameters()
            b['pool'].reset_parameters()
            if b['bn'] is not None:
                b['bn'].reset_running_stats(); b['bn'].reset_parameters()
        for m in self.lin:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, batch):
        for b in self.blocks:
            x = b['conv'](x, edge_index)
            if b['bn'] is not None:
                x = b['bn'](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x, edge_index, _, batch, _, _ = b['pool'](x, edge_index, None, batch)
        g = global_mean_pool(x, batch)
        out = self.lin(g)
        return out

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
