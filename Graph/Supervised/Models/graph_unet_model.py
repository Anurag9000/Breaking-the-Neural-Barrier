import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphUNet, global_mean_pool

class GraphUNetNet(nn.Module):
    """
    Graph U-Net for graph classification (single model).
    """
    def __init__(self, in_channels, hidden_channels, out_channels,
                 depth=3, pool_ratios=0.5, dropout=0.5):
        super().__init__()
        self.unet = GraphUNet(in_channels, hidden_channels, out_channels,
                              depth=depth, pool_ratios=pool_ratios)
        self.dropout = dropout

    def reset_parameters(self):
        self.unet.reset_parameters()

    def forward(self, x, edge_index, batch):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.unet(x, edge_index)
        # GraphUNet returns node-level logits; do mean readout for graph classification
        g = global_mean_pool(x, batch)
        return g

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
