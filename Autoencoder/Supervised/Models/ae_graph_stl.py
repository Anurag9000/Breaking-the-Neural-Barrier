import torch
import torch.nn as nn
from typing import Tuple

# -----------------------------------------------------------------------------
# AE_GRAPH_STL: Graph AE operating on a grid of image patches (no external GNN
# deps). We build a fixed 4-neighbor adjacency over PxP patch tokens and apply a
# simple message-passing layer: H' = ReLU(BN(W_self*H + W_nei*mean(H_neighbors))).
# Decoder mirrors layers and projects back to pixel patches, then unpatchify.
# -----------------------------------------------------------------------------

class GraphLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin_self = nn.Linear(dim, dim, bias=False)
        self.lin_nei = nn.Linear(dim, dim, bias=False)
        self.bn = nn.BatchNorm1d(dim)
        self.act = nn.ReLU(inplace=True)
    def forward(self, h, adj_idx, P):
        # h: (B, N, D), adj_idx: (N, K) neighbor indices, grid P x P
        B, N, D = h.shape
        K = adj_idx.size(1)
        # gather neighbors
        nei = h.gather(dim=1, index=adj_idx.unsqueeze(-1).expand(B, N, K, D))  # (B,N,K,D)
        nei_mean = nei.mean(dim=2)  # (B,N,D)
        out = self.lin_self(h) + self.lin_nei(nei_mean)
        # BN over features: flatten batch*nodes
        out = self.bn(out.reshape(B*N, D)).reshape(B, N, D)
        return self.act(out)

def build_grid_neighbors(P: int, device):
    # 4-neighborhood on P x P grid producing indices (N,K=4)
    idx = []
    for r in range(P):
        for c in range(P):
            neigh = []
            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                rr, cc = r+dr, c+dc
                if 0 <= rr < P and 0 <= cc < P:
                    neigh.append(rr*P+cc)
                else:
                    neigh.append(r*P+c)  # self if out-of-bounds
            idx.append(neigh)
    return torch.tensor(idx, dtype=torch.long, device=device)  # (N,4)

class AE_GRAPH_STL(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 4, dim: int = 128, depth: int = 4):
        super().__init__()
        self.ps = patch_size
        self.dim = dim
        self.depth = depth
        self.enc_layers = nn.ModuleList([GraphLayer(dim) for _ in range(depth)])
        self.dec_layers = nn.ModuleList([GraphLayer(dim) for _ in range(depth)])
        self.embed = nn.Linear(in_channels * patch_size * patch_size, dim)
        self.head = nn.Linear(dim, in_channels * patch_size * patch_size)
        self.register_buffer('adj_idx', torch.empty(0, dtype=torch.long), persistent=False)

    def _patchify(self, x):
        B, C, H, W = x.shape
        ps = self.ps
        assert H % ps == 0 and W % ps == 0
        P = H // ps
        x = x.view(B, C, P, ps, P, ps).permute(0,2,4,1,3,5)  # (B,P,P,C,ps,ps)
        x = x.reshape(B, P*P, C*ps*ps)
        return x, P

    def _unpatchify(self, patches, P):
        B, N, F = patches.shape
        ps = self.ps; C = 3
        x = patches.view(B, P, P, C, ps, ps).permute(0,3,1,4,2,5).reshape(B, C, P*ps, P*ps)
        return x

    def encode(self, x: torch.Tensor):
        patches, P = self._patchify(x)
        h = self.embed(patches)
        if self.adj_idx.numel() == 0 or self.adj_idx.size(0) != P*P:
            self.adj_idx = build_grid_neighbors(P, x.device)
        adj = self.adj_idx  # (N,4)
        for g in self.enc_layers:
            h = g(h, adj, P)
        return h, P

    def decode(self, h, P):
        adj = self.adj_idx
        for g in self.dec_layers:
            h = g(h, adj, P)
        patches = self.head(h)
        return self._unpatchify(patches, P)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, P = self.encode(x)
        x_rec = self.decode(h, P)
        return x_rec, h


def ae_graph_total_neurons(dim: int, depth: int) -> int:
    return int(dim * (depth + 1))
