import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import k_hop_subgraph, drnl_node_labeling
from torch_geometric.nn import GCNConv, global_mean_pool, BatchNorm

class SEALSubgraphGNN(nn.Module):
    """
    Minimal SEAL-style subgraph GNN for link prediction.
    - Extract k-hop enclosing subgraph around (u,v), label nodes with DRNL, run a GNN to classify edge existence.
    - Single-model end-to-end.
    """
    def __init__(self, in_channels, hidden_channels, num_layers=3, dropout=0.5, use_batchnorm=True):
        super().__init__()
        assert num_layers >= 2
        self.dropout = dropout
        self.use_bn = use_batchnorm

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batchnorm else None

        self.convs.append(GCNConv(in_channels + 1, hidden_channels))  # +1 for DRNL label embedding (as scalar)
        if self.use_bn:
            self.bns.append(BatchNorm(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            if self.use_bn:
                self.bns.append(BatchNorm(hidden_channels))
        self.lin = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden_channels, 2)
        )

    def reset_parameters(self):
        for c in self.convs:
            c.reset_parameters()
        if self.bns is not None:
            for bn in self.bns:
                bn.reset_running_stats(); bn.reset_parameters()
        for m in self.lin:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x, edge_index, batch):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        g = global_mean_pool(x, batch)
        out = self.lin(g)
        return out


def extract_enclosing_subgraph(data, u, v, num_hops=2):
    # Extract k-hop around u and v, union subgraph
    (nodes, edge_index, _, edge_mask) = k_hop_subgraph([u, v], num_hops, data.edge_index, relabel_nodes=True)
    # Node features: concat original x and DRNL labels
    x_sub = data.x[nodes]
    z = drnl_node_labeling(edge_index, 0, 1, num_nodes=nodes.size(0))  # anchors 0,1 after relabel
    z = z.float().unsqueeze(-1)
    x_aug = torch.cat([x_sub, z], dim=-1)
    return nodes, edge_index, x_aug
