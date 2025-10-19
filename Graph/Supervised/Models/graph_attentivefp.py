import torch
import torch.nn as nn
from torch_geometric.nn.models import AttentiveFP

class AttentiveFPNet(nn.Module):
    """Wrapper over PyG's AttentiveFP for graph-level tasks.
    Exposes a simple forward(graph_batch) -> logits per graph.
    """
    def __init__(self, in_dim, hidden_dim=64, out_dim=2, num_layers=2, dropout=0.2):
        super().__init__()
        self.model = AttentiveFP(in_channels=in_dim,
                                 hidden_channels=hidden_dim,
                                 out_channels=out_dim,
                                 num_layers=num_layers,
                                 dropout=dropout)
    def forward(self, x, edge_index, batch, edge_attr=None):
        return self.model(x, edge_index, edge_attr, batch)
