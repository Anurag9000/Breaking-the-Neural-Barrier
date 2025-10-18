import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import EdgeConv, global_mean_pool, BatchNorm

class MLP(nn.Sequential):
    def __init__(self, c_in, c_out):
        super().__init__(
            nn.Linear(c_in, c_out), nn.ReLU(inplace=True), nn.Linear(c_out, c_out)
        )

class EdgeConvNet(nn.Module):
    """
    EdgeConv (DGCNN) style model for graph/point-cloud graphs.
    Uses provided graph edges; if using kNN, construct edges before feeding.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=3, dropout=0.5, use_batchnorm=True):
        super().__init__()
        assert num_layers >= 2
        self.dropout = dropout
        self.use_bn = use_batchnorm

        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batchnorm else None

        # EdgeConv expects an MLP on concatenated (x_i || x_j - x_i), so feature dim doubles internally
        self.layers.append(EdgeConv(MLP(2 * in_channels, hidden_channels)))
        if self.use_bn:
            self.bns.append(BatchNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(EdgeConv(MLP(2 * hidden_channels, hidden_channels)))
            if self.use_bn:
                self.bns.append(BatchNorm(hidden_channels))
        self.lin = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def reset_parameters(self):
        for l in self.layers:
            l.reset_parameters()
        if self.bns is not None:
            for bn in self.bns:
                bn.reset_running_stats(); bn.reset_parameters()
        for m in self.lin:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, batch):
        h = x
        for i, conv in enumerate(self.layers):
            h = conv(h, edge_index)
            if self.use_bn:
                h = self.bns[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        g = global_mean_pool(h, batch)
        out = self.lin(g)
        return out

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
