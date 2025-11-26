from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from Autoencoder.common.core import ADPBackbone, PhysicsCorrection
from Autoencoder.common.data_utils import CaseMetadata


class MaskedEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden: int, layers: int):
        super().__init__()
        convs = []
        last = in_channels
        for _ in range(layers):
            convs.append(GCNConv(last, hidden))
            last = hidden
        self.convs = nn.ModuleList(convs)

    def forward(self, x, edge_index):
        h = x
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
        return h


class AEMaskedModel(nn.Module):
    def __init__(
        self,
        meta: CaseMetadata,
        *,
        latent_dim: int,
        hidden: int,
        layers: int,
        proj_newton_steps: int,
    ):
        super().__init__()
        self.meta = meta
        self.encoder = MaskedEncoder(2, hidden, layers)
        self.pool_proj = nn.Linear(hidden, latent_dim)
        self.backbone = ADPBackbone(latent_dim, hidden, depth=2, out_dim=hidden)
        self.pg_head = nn.Linear(hidden, meta.n_gen)
        self.qg_head = nn.Linear(hidden, meta.n_gen)
        self.va_head = nn.Linear(hidden, meta.n_bus)
        self.vm_head = nn.Linear(hidden, meta.n_bus)
        self.bus_recon_head = nn.Linear(hidden, meta.n_bus * 2)
        self.physics = PhysicsCorrection(meta.case_name, mask=None, steps=proj_newton_steps)
        self.register_buffer("bounds_lo", self.physics.bound_layer.lo.squeeze(0))
        self.register_buffer("bounds_hi", self.physics.bound_layer.hi.squeeze(0))
        self.register_buffer("y_bus_real", meta.y_bus.real)
        self.register_buffer("y_bus_imag", meta.y_bus.imag)
        self.register_buffer("gen_bus_idx", meta.gen_bus_idx.clone())
        self.register_buffer("load_bus_idx", meta.load_bus_idx.clone())

    def forward(self, data) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = data.x
        h = self.encoder(x, data.edge_index)
        pooled = global_mean_pool(h, data.batch)
        latent = F.relu(self.pool_proj(pooled))
        features = self.backbone(latent)
        pg = self.pg_head(features)
        qg = self.qg_head(features)
        va = self.va_head(features)
        vm = self.vm_head(features)
        outputs = torch.cat([pg, qg, va, vm], dim=1)
        corrected = self.physics(outputs, data.pdqd)
        bus_recon = self.bus_recon_head(features).view(-1, self.meta.n_bus, 2)
        return corrected, {"latent": latent, "features": features, "bus_recon": bus_recon}

