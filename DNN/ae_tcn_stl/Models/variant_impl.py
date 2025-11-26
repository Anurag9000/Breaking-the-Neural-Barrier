from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from Autoencoder.common.core import ADPBackbone, PhysicsCorrection
from Autoencoder.common.data_utils import CaseMetadata


def _make_tcn_block(in_ch: int, out_ch: int, kernel_size: int, dilation: int) -> nn.Sequential:
    padding = (kernel_size - 1) * dilation
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
        nn.ReLU(),
        nn.Conv1d(out_ch, out_ch, kernel_size=1),
        nn.ReLU(),
    )


class TemporalEncoder(nn.Module):
    def __init__(self, in_ch: int, channels: int, dilations: Tuple[int, ...]):
        super().__init__()
        blocks = []
        last = in_ch
        for d in dilations:
            blocks.append(_make_tcn_block(last, channels, kernel_size=3, dilation=d))
            last = channels
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        h = x
        for block in self.blocks:
            h = block(h)
        return h


class AETCNModel(nn.Module):
    def __init__(
        self,
        meta: CaseMetadata,
        *,
        temporal_len: int,
        tcn_channels: int,
        tcn_dilations: Tuple[int, ...],
        hidden: int,
        latent_dim: int,
        proj_newton_steps: int,
    ):
        super().__init__()
        self.meta = meta
        self.temporal_len = temporal_len
        self.tcn = TemporalEncoder(2, tcn_channels, tcn_dilations)
        self.graph_conv = GCNConv(tcn_channels, hidden)
        self.pool_proj = nn.Linear(hidden, latent_dim)
        self.backbone = ADPBackbone(latent_dim, hidden, depth=2, out_dim=hidden)
        self.pg_head = nn.Linear(hidden, meta.n_gen)
        self.qg_head = nn.Linear(hidden, meta.n_gen)
        self.va_head = nn.Linear(hidden, meta.n_bus)
        self.vm_head = nn.Linear(hidden, meta.n_bus)
        self.physics = PhysicsCorrection(meta.case_name, mask=None, steps=proj_newton_steps)
        self.register_buffer("bounds_lo", self.physics.bound_layer.lo.squeeze(0))
        self.register_buffer("bounds_hi", self.physics.bound_layer.hi.squeeze(0))
        self.register_buffer("y_bus_real", meta.y_bus.real)
        self.register_buffer("y_bus_imag", meta.y_bus.imag)
        self.register_buffer("gen_bus_idx", meta.gen_bus_idx.clone())
        self.register_buffer("load_bus_idx", meta.load_bus_idx.clone())

    def forward(self, data) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sequence = data.sequence  # [B, T, n_bus, 2]
        B, T, N, _ = sequence.shape
        seq = sequence.permute(0, 2, 3, 1).reshape(B * N, 2, T)
        tcn_out = self.tcn(seq)
        latest = tcn_out[:, :, -1]  # [B*N, channels]
        node_features = latest.view(B, N, -1).reshape(-1, latest.size(1))
        h = F.relu(node_features)
        h = self.graph_conv(h, data.edge_index)
        pooled = global_mean_pool(h, data.batch)
        latent = F.relu(self.pool_proj(pooled))
        features = self.backbone(latent)
        pg = self.pg_head(features)
        qg = self.qg_head(features)
        va = self.va_head(features)
        vm = self.vm_head(features)
        outputs = torch.cat([pg, qg, va, vm], dim=1)
        corrected = self.physics(outputs, data.pdqd)
        return corrected, {"latent": latent, "features": features}

