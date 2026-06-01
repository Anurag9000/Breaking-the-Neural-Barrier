import torch
import torch.nn as nn
from typing import Tuple


class RowMLP(nn.Module):
    """
    Simple MLP applied row-wise to an adjacency matrix.
    Input: (N, N) adjacency row; Output: (N,) reconstructed row.
    """

    def __init__(self, n_nodes: int, width: int, depth: int):
        super().__init__()
        layers = []
        in_dim = n_nodes
        for _ in range(max(depth - 1, 0)):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.ReLU(inplace=True))
            in_dim = width
        layers.append(nn.Linear(in_dim, n_nodes))
        self.net = nn.Sequential(*layers)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        # a: (N, N)
        return self.net(a)


class DAEGraphLink(nn.Module):
    """
    Graph link-structure denoising autoencoder.

    - Input: adjacency matrix A (N, N) with entries in [0,1] or {0,1}.
    - Noise: applied to A in the training loop (e.g., random edge dropout).
    - Width: hidden size for intermediate MLP layers.
    - Depth: number of layers in the row-wise MLP (depth>=1).
    """

    def __init__(self, n_nodes: int, width: int = 64, depth: int = 2):
        super().__init__()
        assert depth >= 1
        self.n_nodes = n_nodes
        self.width = width
        self.depth = depth
        self.encoder = RowMLP(n_nodes, width, depth)
        # For this simple model, encoder and decoder share the same structure;
        # the "latent" is just the hidden representation before the final layer.

    def forward(self, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        adj: (N, N) dense adjacency.
        Returns: (adj_rec, latent), where latent is the hidden representation
        just before the final linear layer.
        """
        a = adj
        # We expose latent as the penultimate layer output; obtain it via hooks.
        x = a
        # Manually run through layers to capture latent
        latent = None
        modules = list(self.encoder.net.children())
        for i, layer in enumerate(modules):
            x = layer(x)
            if i == len(modules) - 2:
                latent = x
        adj_rec = x
        if latent is None:
            latent = adj_rec
        return adj_rec, latent


def graph_link_dae_total_neurons(width: int, depth: int) -> int:
    return int(width * depth)

