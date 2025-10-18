import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv

class HGTNet(nn.Module):
    """
    Heterogeneous Graph Transformer (HGT) encoder for node classification.
    Single-model; uses type-specific projections and attention with relation-aware bias.
    """
    def __init__(self, metadata, in_channels_dict, hidden_channels, out_channels,
                 num_layers: int = 2, heads: int = 8, dropout: float = 0.2):
        super().__init__()
        assert num_layers >= 1
        self.dropout = dropout

        self.proj = nn.ModuleDict({nt: nn.Linear(in_channels_dict[nt], hidden_channels) for nt in in_channels_dict})
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(HGTConv(hidden_channels, hidden_channels, metadata, heads=heads))
        self.classifier = nn.Linear(hidden_channels, out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.proj.values():
            nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
        for m in self.layers:
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()
        nn.init.xavier_uniform_(self.classifier.weight); nn.init.zeros_(self.classifier.bias)

    def forward(self, x_dict, edge_index_dict, target_type: str):
        h_dict = {k: F.dropout(F.relu(self.proj[k](x)), p=self.dropout, training=self.training) for k, x in x_dict.items()}
        for conv in self.layers:
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {k: F.dropout(F.relu(v), p=self.dropout, training=self.training) for k, v in h_dict.items()}
        out = self.classifier(h_dict[target_type])
        return out

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
