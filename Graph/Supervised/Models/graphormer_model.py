import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, BatchNorm

class GraphTransformer(nn.Module):
    """
    Graph Transformer (Graphormer-style bias, simplified):
    - Uses TransformerConv with edge_enc as additive bias.
    - Supports learnable node degree embeddings as structural prior.
    - Single-model encoder; final linear classifier for node tasks.
    Note: For full Graphormer (SPD/centrality), extend edge_bias accordingly.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 4, heads: int = 8, dropout: float = 0.1,
                 use_batchnorm: bool = False, use_degree_bias: bool = True,
                 max_degree: int = 512):
        super().__init__()
        assert num_layers >= 2
        self.dropout = dropout
        self.use_bn = use_batchnorm
        self.use_deg = use_degree_bias
        self.heads = heads

        self.lin_in = nn.Linear(in_channels, hidden_channels)
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batchnorm else None

        for _ in range(num_layers - 1):
            self.layers.append(TransformerConv(hidden_channels, hidden_channels // heads, heads=heads, beta=True, dropout=dropout))
            if self.use_bn:
                self.bns.append(BatchNorm(hidden_channels))
        self.out_proj = nn.Linear(hidden_channels, out_channels)

        if self.use_deg:
            self.deg_emb = nn.Embedding(max_degree + 1, hidden_channels)
        else:
            self.deg_emb = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_in.weight); nn.init.zeros_(self.lin_in.bias)
        nn.init.xavier_uniform_(self.out_proj.weight); nn.init.zeros_(self.out_proj.bias)
        for m in self.layers:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        if self.bns is not None:
            for bn in self.bns:
                bn.reset_running_stats(); bn.reset_parameters()
        if self.deg_emb is not None:
            nn.init.normal_(self.deg_emb.weight, std=0.02)

    def forward(self, x, edge_index, edge_attr=None, deg=None):
        # x: [N, F] node features
        # edge_attr: optional edge features to act as bias (beta in TransformerConv)
        h = self.lin_in(x)
        if self.use_deg and deg is not None:
            deg = deg.clamp_min(0).clamp_max(self.deg_emb.num_embeddings - 1)
            h = h + self.deg_emb(deg)
        h = F.dropout(h, p=self.dropout, training=self.training)
        for i, layer in enumerate(self.layers):
            h = layer(h, edge_index, edge_attr=edge_attr)
            if self.use_bn:
                h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.out_proj(h)
        return out

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
