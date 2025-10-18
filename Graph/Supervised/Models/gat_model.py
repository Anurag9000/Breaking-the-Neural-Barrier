import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, BatchNorm

class GAT(nn.Module):
    """
    Graph Attention Network (Velickovic et al.)
    - Multi-head attention; final layer uses averaging across heads by default.
    - Single-model, no EMA/teacher.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, heads: int = 8, out_heads: int = 1,
                 dropout: float = 0.6, attn_dropout: float = 0.6,
                 use_batchnorm: bool = False, concat: bool = True):
        super().__init__()
        assert num_layers >= 2
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.use_bn = use_batchnorm
        self.concat = concat

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batchnorm else None

        # Input layer
        self.layers.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=attn_dropout, concat=True))
        if self.use_bn:
            self.bns.append(BatchNorm(hidden_channels * heads))
        # Hidden layers
        for _ in range(num_layers - 2):
            in_ch = hidden_channels * heads
            self.layers.append(GATConv(in_ch, hidden_channels, heads=heads, dropout=attn_dropout, concat=True))
            if self.use_bn:
                self.bns.append(BatchNorm(hidden_channels * heads))
        # Output layer: often set concat=False and heads=out_heads, then mean
        in_ch = hidden_channels * heads
        self.out_conv = GATConv(in_ch, out_channels, heads=out_heads, dropout=attn_dropout, concat=False)

    def reset_parameters(self):
        for m in self.layers:
            m.reset_parameters()
        self.out_conv.reset_parameters()
        if self.bns is not None:
            for bn in self.bns:
                bn.reset_running_stats(); bn.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.out_conv(x, edge_index)
        return x

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
