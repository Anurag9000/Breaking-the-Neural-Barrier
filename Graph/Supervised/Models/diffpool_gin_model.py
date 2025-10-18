import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, BatchNorm
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn.dense.diff_pool import diff_pool
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import to_dense_batch

class MLP(nn.Sequential):
    def __init__(self, c_in, c_out):
        super().__init__(
            nn.Linear(c_in, c_out), nn.ReLU(inplace=True), nn.Linear(c_out, c_out)
        )

class GINBlock(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, dropout=0.0, train_eps=False):
        super().__init__()
        self.conv = GINConv(MLP(in_ch, out_ch), train_eps=train_eps)
        self.bn = BatchNorm(out_ch) if bn else None
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        if self.bn is not None:
            x = self.bn(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class DiffPoolGIN(nn.Module):
    """
    Two-stage DiffPool with GIN backbones (single model, end-to-end).
    Stage k uses an 'embed GNN' to produce node embeddings Z_k and an 'assign GNN' to produce cluster assignment S_k.
    We then call diff_pool on dense graphs.
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 cluster_ratio: float = 0.25, dropout: float = 0.5, train_eps: bool = False):
        super().__init__()
        self.dropout = dropout
        self.cluster_ratio = cluster_ratio

        # Stage 1 GNNs
        self.gnn_embed_1 = nn.Sequential(
            GINBlock(in_channels, hidden_channels, bn=True, dropout=dropout, train_eps=train_eps),
            GINBlock(hidden_channels, hidden_channels, bn=True, dropout=dropout, train_eps=train_eps)
        )
        self.gnn_assign_1 = nn.Sequential(
            GINBlock(in_channels, hidden_channels, bn=True, dropout=dropout, train_eps=train_eps),
            GINBlock(hidden_channels, hidden_channels, bn=True, dropout=dropout, train_eps=train_eps),
            nn.Linear(hidden_channels, int(hidden_channels * cluster_ratio))
        )

        # Stage 2 GNNs
        self.gnn_embed_2 = nn.Sequential(
            GINBlock(hidden_channels, hidden_channels, bn=True, dropout=dropout, train_eps=train_eps)
        )
        self.gnn_assign_2 = nn.Sequential(
            GINBlock(hidden_channels, hidden_channels, bn=True, dropout=dropout, train_eps=train_eps),
            nn.Linear(hidden_channels, int(hidden_channels * cluster_ratio))
        )

        # Classifier after pooled graph representation
        self.lin_graph = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(hidden_channels, out_channels)
        )

    def reset_parameters(self):
        def reset_seq(m):
            for c in m.children():
                if hasattr(c, 'reset_parameters'): c.reset_parameters()
        reset_seq(self.gnn_embed_1); reset_seq(self.gnn_assign_1)
        reset_seq(self.gnn_embed_2); reset_seq(self.gnn_assign_2)
        for m in self.lin_graph:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def _dense(self, x, edge_index, batch):
        x_dense, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch, max_num_nodes=x_dense.size(1))
        return x_dense, adj, mask

    def forward(self, x, edge_index, batch):
        # Stage 1
        z1 = self.gnn_embed_1[0](x, edge_index)
        z1 = self.gnn_embed_1[1](z1, edge_index)
        s1 = self.gnn_assign_1[0](x, edge_index)
        s1 = self.gnn_assign_1[1](s1, edge_index)
        s1 = self.gnn_assign_1[2](s1)

        x_dense, adj, mask = self._dense(x, edge_index, batch)
        z1_dense, _, _ = self._dense(z1, edge_index, batch)
        s1_dense, _, _ = self._dense(s1, edge_index, batch)

        x1, adj1, l1, e1 = diff_pool(z1_dense, adj, s1_dense, mask)

        # Stage 2 (one more coarsening)
        # Build a fake edge_index is not needed in dense; use dense again
        # We need to run small GINs over the current dense graph: emulate via 1x1 conv (MLP) after pooling
        # Simpler: apply another diff_pool with assignment from a small GNN on original graph-level embeddings
        # Here we approximate by linear maps on x1 for stability
        z2 = self.gnn_embed_2[0].conv.nn(x1)  # use inside MLP of GIN as an MLP on dense feats
        s2 = self.gnn_assign_2[0].conv.nn(x1)
        s2 = self.gnn_assign_2[1](s2)
        x2, adj2, l2, e2 = diff_pool(z2, adj1, s2)

        # Readout: mean over final coarse nodes per graph in dense (batch dimension kept)
        # x2: [B, N_coarse, C]
        graph_emb = x2.mean(dim=1)
        out = self.lin_graph(graph_emb)
        return out

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
