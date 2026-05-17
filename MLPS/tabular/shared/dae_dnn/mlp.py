import torch
import torch.nn as nn
from typing import Iterable, List


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_widths: Iterable[int], out_dim: int, use_bn: bool = True):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden_widths = [int(w) for w in hidden_widths]
        self.use_bn = bool(use_bn)

        layers: List[nn.Module] = []
        prev = self.in_dim
        for w in self.hidden_widths:
            layers.append(nn.Linear(prev, w))
            if self.use_bn:
                layers.append(nn.BatchNorm1d(w))
            layers.append(nn.ReLU(inplace=True))
            prev = w
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, self.out_dim)

    def forward(self, x, return_embedding: bool = False):
        x = x.view(x.size(0), -1)
        h = self.backbone(x)
        out = self.head(h)
        if return_embedding:
            return out, h
        return out
