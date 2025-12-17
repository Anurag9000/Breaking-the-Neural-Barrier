import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class GraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        x: (N, F), adj: (N, N) assumed normalized
        """
        h = adj @ x
        h = self.lin(h)
        h = self.bn(h)
        return F.relu(h)


class DAEGraphNodeFeat(nn.Module):
    """
    Graph node-feature denoising autoencoder.

    - Input: node features X (N, F_in) and dense (or sparse) adjacency A.
    - Noise: applied to X in the training loop (e.g. Gaussian or masking).
    - Width: hidden feature size for intermediate graph conv layers.
    - Depth: number of encoder graph conv layers; decoder mirrors encoder.
    """

    def __init__(self, in_dim: int, width: int = 64, depth: int = 2):
        super().__init__()
        assert depth >= 1
        self.in_dim = in_dim
        self.width = width
        self.depth = depth

        enc_layers = []
        dim_in = in_dim
        for _ in range(depth):
            enc_layers.append(GraphConv(dim_in, width))
            dim_in = width
        self.encoder = nn.ModuleList(enc_layers)

        dec_layers = []
        dim_in = width
        for i in range(depth, 0, -1):
            dim_out = width if i > 1 else in_dim
            dec_layers.append(GraphConv(dim_in, dim_out) if i > 1 else GraphConv(dim_in, dim_out))
            dim_in = dim_out
        self.decoder = nn.ModuleList(dec_layers)

    def encode(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.encoder:
            h = layer(h, adj)
        return h

    def decode(self, z: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = z
        for i, layer in enumerate(self.decoder):
            # For final layer we drop ReLU by clamping through identity activation
            h = layer(h, adj)
            if i == len(self.decoder) - 1:
                # Do not force nonlinearity on last projection
                pass
        return h

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x, adj)
        x_rec = self.decode(z, adj)
        return x_rec, z


def graph_node_dae_total_neurons(width: int, depth: int) -> int:
    return int(width * (depth + 1))

