from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from Autoencoder.common.core import ADPBackbone, PhysicsCorrection
from Autoencoder.common.data_utils import CaseMetadata


class LowRankEncoder(nn.Module):
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


class AELowRankModel(nn.Module):
    def __init__(
        self,
        meta: CaseMetadata,
        *,
        latent_dim: int,
        hidden: int,
        layers: int,
        rank_k: int,
        proj_newton_steps: int,
        nuclear_lambda: float,
    ):
        super().__init__()
        self.meta = meta
        self.rank_k = rank_k
        self.nuclear_lambda = nuclear_lambda
        self.encoder = LowRankEncoder(2, hidden, layers)
        self.u_proj = nn.Linear(hidden, rank_k)
        self.latent_factor = nn.Parameter(torch.randn(rank_k, latent_dim) * 0.01)
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

    def lowrank_penalty(self, u_batch: torch.Tensor) -> torch.Tensor:
        if self.nuclear_lambda <= 0:
            return torch.tensor(0.0, device=u_batch.device)
        return self.nuclear_lambda * (
            u_batch.pow(2).mean() + self.latent_factor.pow(2).mean()
        )

    def forward(self, data) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        h = self.encoder(data.x, data.edge_index)
        pooled = global_mean_pool(h, data.batch)
        u_batch = self.u_proj(pooled)
        latent = torch.matmul(u_batch, self.latent_factor)
        features = self.backbone(F.relu(latent))
        pg = self.pg_head(features)
        qg = self.qg_head(features)
        va = self.va_head(features)
        vm = self.vm_head(features)
        outputs = torch.cat([pg, qg, va, vm], dim=1)
        corrected = self.physics(outputs, data.pdqd)
        return corrected, {"latent": latent, "features": features, "u_batch": u_batch}

