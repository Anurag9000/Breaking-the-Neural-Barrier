import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP, BatchNorm

class APPNPNet(nn.Module):
    """
    APPNP: MLP predictor + personalized PageRank propagation.
    Single-model: 2-layer MLP -> APPNP propagation -> logits.
    """
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 mlp_layers: int = 2, K: int = 10, alpha: float = 0.1,
                 dropout: float = 0.5, use_batchnorm: bool = False):
        super().__init__()
        assert mlp_layers >= 1
        self.dropout = dropout
        self.use_bn = use_batchnorm

        mlp = []
        c_in = in_channels
        for i in range(mlp_layers - 1):
            mlp.append(nn.Linear(c_in, hidden_channels))
            if use_batchnorm:
                mlp.append(nn.BatchNorm1d(hidden_channels))
            mlp.append(nn.ReLU(inplace=True))
            mlp.append(nn.Dropout(dropout))
            c_in = hidden_channels
        mlp.append(nn.Linear(c_in, out_channels))
        self.mlp = nn.Sequential(*mlp)

        self.prop = APPNP(K=K, alpha=alpha, dropout=dropout)

    def reset_parameters(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
            elif hasattr(m, 'reset_running_stats'):
                m.reset_running_stats(); m.reset_parameters()
        self.prop.reset_parameters()

    def forward(self, x, edge_index):
        x0 = self.mlp(x)
        x = self.prop(x0, edge_index)
        return x

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
