import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HANConv

class HANNet(nn.Module):
    """
    Heterogeneous Attention Network (HAN) for node classification on hetero graphs.
    Single-model encoder; metapath-based semantic-level attention.
    """
    def __init__(self, metadata, in_channels_dict, hidden_channels, out_channels,
                 metapaths, num_layers: int = 2, heads: int = 8, dropout: float = 0.6):
        super().__init__()
        assert num_layers >= 1
        self.dropout = dropout
        self.metapaths = metapaths

        # First HANConv projects per-type features to hidden and aggregates per metapath
        self.layers = nn.ModuleList()
        self.layers.append(HANConv(in_channels_dict, hidden_channels, metapaths, heads=heads))
        for _ in range(num_layers - 1):
            self.layers.append(HANConv({k: hidden_channels for k in in_channels_dict.keys()},
                                       hidden_channels, metapaths, heads=heads))
        # Output classifier for the target node type
        self.classifier = nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        for conv in self.layers:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()
        nn.init.xavier_uniform_(self.classifier.weight); nn.init.zeros_(self.classifier.bias)

    def forward(self, x_dict, edge_index_dict, target_type: str):
        h_dict = x_dict
        for conv in self.layers:
            h_dict = conv(h_dict, edge_index_dict)
            h_dict = {k: F.elu(v) for k, v in h_dict.items()}
            h_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in h_dict.items()}
        logits = self.classifier(h_dict[target_type])
        return logits

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
