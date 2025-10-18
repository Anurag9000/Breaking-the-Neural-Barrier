import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TopKPooling, SAGPooling, global_mean_pool, BatchNorm

class ConvPoolBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool_type='topk', ratio=0.5, use_bn=True, dropout=0.0):
        super().__init__()
        self.conv = GCNConv(in_ch, out_ch)
        self.bn = BatchNorm(out_ch) if use_bn else None
        if pool_type == 'topk':
            self.pool = TopKPooling(out_ch, ratio=ratio)
        elif pool_type == 'sag':
            self.pool = SAGPooling(out_ch, ratio=ratio)
        else:
            raise ValueError('pool_type must be topk or sag')
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = self.conv(x, edge_index)
        if self.bn is not None:
            x = self.bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, edge_index, _, batch, _, _ = self.pool(x, edge_index, None, batch)
        return x, edge_index, batch

class TopKOrSAGNet(nn.Module):
    """
    Stacked GCN + {TopKPooling | SAGPooling} blocks for graph classification.
    pool_type: 'topk' or 'sag'
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_blocks=3, pool_type='topk', ratio=0.5, dropout=0.5, use_batchnorm=True):
        super().__init__()
        self.blocks = nn.ModuleList()
        c_in = in_channels
        for i in range(num_blocks):
            self.blocks.append(ConvPoolBlock(c_in, hidden_channels, pool_type=pool_type, ratio=ratio,
                                             use_bn=use_batchnorm, dropout=dropout))
            c_in = hidden_channels
        self.lin = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def reset_parameters(self):
        for b in self.blocks:
            if hasattr(b.conv, 'reset_parameters'): b.conv.reset_parameters()
            if hasattr(b.pool, 'reset_parameters'): b.pool.reset_parameters()
            if hasattr(b.bn, 'reset_parameters') and b.bn is not None:
                b.bn.reset_running_stats(); b.bn.reset_parameters()
        for m in self.lin:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, batch):
        for blk in self.blocks:
            x, edge_index, batch = blk(x, edge_index, batch)
        g = global_mean_pool(x, batch)
        out = self.lin(g)
        return out

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
