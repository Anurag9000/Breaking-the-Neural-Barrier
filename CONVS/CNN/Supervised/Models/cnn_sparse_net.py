"""
SparseNet for CIFAR-10/100 — single-model supervised.

Implements sparsified dense connectivity (exponentially spaced skip concatenations)
per the SparseNet idea: the l-th layer aggregates a *subset* of prior features using
offsets S = {1, 2, 4, 8, ...} (bounded by available layers), rather than all previous layers
as in DenseNet. This reduces concatenation growth while preserving multi-scale feature reuse.

Design (CIFAR-friendly):
- Stem: 3×3 conv with 2k channels.
- 3 SparseBlocks, each with n layers. Each layer:
    x_l = Conv3×3( BN(ReLU(concat(x_{l−s} for s\in S_l plus current block input))) )
  The layer outputs `growth_rate` channels which are appended to the block's feature set.
- Transitions between blocks: BN→ReLU→1×1 conv (θ·C) → AvgPool2d(2).
- Head: BN→ReLU→GAP→Linear(num_classes).

Parameters:
- growth_rate k (e.g., 12, 24)
- layers_per_block n (e.g., 16) controlling depth L ≈ 3n + stem/head (conv count)
- compression θ in transitions (default 0.5)
- optional dropout after the 3×3 conv in each sparse layer

This mirrors your modular code style: clear blocks, factory, and param_count helper.
"""
from __future__ import annotations
from typing import List
import math
import torch
import torch.nn as nn

__all__ = ["SparseNetCIFAR", "make_sparsenet_cifar", "SparseLayer", "SparseBlock", "Transition"]

class _BNReLU(nn.Sequential):
    def __init__(self, ch: int):
        super().__init__(nn.BatchNorm2d(ch), nn.ReLU(inplace=True))

class SparseLayer(nn.Module):
    """A single SparseNet layer with exponentially spaced concatenation.

    Given a list of previous tensors [x0 (block input), x1, x2, ..., x_{l-1}],
    this layer selects a subset with offsets S={1,2,4,8,...} and concatenates them
    (including the immediate predecessor), applies BN-ReLU-Conv3x3 to produce k new channels.
    """
    def __init__(self, in_ch_dynamic: int, growth_rate: int, drop_rate: float = 0.0):
        super().__init__()
        # in_ch_dynamic is a maximum placeholder to initialize BN; we will create BN on the fly
        # but for performance we keep a single BN by expecting correct in_ch at runtime using Lazy modules.
        self.drop = nn.Dropout(p=drop_rate) if drop_rate > 0 else nn.Identity()
        self.conv = nn.Conv2d(in_channels=0, out_channels=growth_rate, kernel_size=3, padding=1, bias=False)
        # Use Lazy modules so the concatenated channel count can vary per layer
        self.bn = nn.LazyBatchNorm2d()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.LazyConv2d(out_channels=growth_rate, kernel_size=3, padding=1, bias=False)
        
        # Init deferred until shape known; Lazy layers will init at first forward.

    @staticmethod
    def _select_indices(num_feats: int) -> List[int]:
        # Indices to pick from the feature list (0-based), using exponential offsets from last
        # Always include the last feature (num_feats-1), then last-2^i while >=0.
        idxs = []
        last = num_feats - 1
        if last >= 0:
            idxs.append(last)
        offset = 1
        while last - offset >= 0:
            idxs.append(last - offset)
            offset <<= 1
        # Always include the block input index 0 as well (if not already)
        if 0 not in idxs:
            idxs.append(0)
        # Keep ascending order for stable concatenation
        idxs = sorted(set(idxs))
        return idxs

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        # feats: list of tensors produced so far in the block, where feats[0] is the block input
        idxs = self._select_indices(len(feats))
        x = torch.cat([feats[i] for i in idxs], dim=1)
        x = self.relu(self.bn(x))
        x = self.conv(x)
        x = self.drop(x)
        return x

class SparseBlock(nn.Module):
    def __init__(self, num_layers: int, in_ch: int, growth_rate: int, drop_rate: float):
        super().__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.layers = nn.ModuleList([SparseLayer(in_ch + i*growth_rate, growth_rate, drop_rate) for i in range(num_layers)])
        self.out_ch = in_ch + num_layers * growth_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats: List[torch.Tensor] = [x]
        for layer in self.layers:
            y = layer(feats)
            feats.append(y)
        return torch.cat(feats, dim=1)

class Transition(nn.Module):
    def __init__(self, in_ch: int, theta: float = 0.5):
        super().__init__()
        out_ch = int(math.floor(in_ch * theta))
        self.bn_relu = _BNReLU(in_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2)
        self.out_ch = out_ch
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn_relu(x)
        x = self.conv(x)
        x = self.pool(x)
        return x

class SparseNetCIFAR(nn.Module):
    def __init__(self, growth_rate: int = 12, layers_per_block: int = 16, compression: float = 0.5,
                 drop_rate: float = 0.0, num_classes: int = 10, in_channels: int = 3):
        super().__init__()
        k = growth_rate
        n = layers_per_block
        theta = compression

        # Stem
        num_features = 2 * k
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=3, stride=1, padding=1, bias=False)

        # Block 1
        self.sb1 = SparseBlock(n, num_features, k, drop_rate)
        num_features = self.sb1.out_ch
        self.tr1 = Transition(num_features, theta)
        num_features = self.tr1.out_ch

        # Block 2
        self.sb2 = SparseBlock(n, num_features, k, drop_rate)
        num_features = self.sb2.out_ch
        self.tr2 = Transition(num_features, theta)
        num_features = self.tr2.out_ch

        # Block 3 (no transition after)
        self.sb3 = SparseBlock(n, num_features, k, drop_rate)
        num_features = self.sb3.out_ch

        # Head
        self.bn_relu = _BNReLU(num_features)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(num_features, num_classes)
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.sb1(x)
        x = self.tr1(x)
        x = self.sb2(x)
        x = self.tr2(x)
        x = self.sb3(x)
        x = self.bn_relu(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    @staticmethod
    def param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())


def make_sparsenet_cifar(growth_rate: int = 12, layers_per_block: int = 16, compression: float = 0.5,
                          drop_rate: float = 0.0, num_classes: int = 10, in_channels: int = 3) -> SparseNetCIFAR:
    return SparseNetCIFAR(growth_rate=growth_rate, layers_per_block=layers_per_block, compression=compression,
                          drop_rate=drop_rate, num_classes=num_classes, in_channels=in_channels)
