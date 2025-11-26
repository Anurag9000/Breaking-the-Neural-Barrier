from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from Autoencoder.common.core import ADPBackbone, PhysicsCorrection
from Autoencoder.common.data_utils import CaseMetadata


class AEMultiScaleModel(nn.Module):
    def __init__(
        self,
        meta: CaseMetadata,
        *,
        levels: int,
        hidden: int,
        latent_dim: int,
        proj_newton_steps: int,
    ):
        super().__init__()
        self.meta = meta
        self.levels = max(1, levels)
        self.input_mlp = nn.Linear(2, hidden)
        self.backbone = ADPBackbone(hidden * self.levels, hidden, depth=2, out_dim=hidden)
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
        self.cluster_maps = nn.ParameterList()
        for level in range(self.levels):
            cluster_ids = self._build_clusters(meta.n_bus, level)
            self.cluster_maps.append(nn.Parameter(cluster_ids, requires_grad=False))

    def _build_clusters(self, n_bus: int, level: int) -> torch.Tensor:
        if level == 0:
            return torch.arange(n_bus)
        cluster_count = max(1, n_bus // (2 ** level))
        cluster_size = math.ceil(n_bus / cluster_count)
        cluster_ids = torch.arange(n_bus) // cluster_size
        cluster_ids = torch.clamp(cluster_ids, max=cluster_count - 1)
        return cluster_ids

    def forward(self, data) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = data.pdqd.size(0)
        device = data.pdqd.device
        node_feat = F.relu(self.input_mlp(data.pdqd.view(B, self.meta.n_bus, 2)))
        fine_level = node_feat
        pooled_features: List[torch.Tensor] = []
        coarse_info: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for level, cluster_param in enumerate(self.cluster_maps):
            clusters = cluster_param.to(device)
            idx = clusters.unsqueeze(0).repeat(B, 1)
            offset = (torch.arange(B, device=device) * (clusters.max().item() + 1)).unsqueeze(1)
            flat_idx = (idx + offset).reshape(-1)
            flat_feat = node_feat.reshape(B * self.meta.n_bus, -1)
            coarse = scatter_mean(flat_feat, flat_idx, dim=0)
            cluster_count = int(clusters.max().item() + 1)
            coarse = coarse.view(B, cluster_count, -1)
            pooled_features.append(coarse.mean(dim=1))
            if level < self.levels - 1:
                upsample = coarse[:, clusters.long(), :]
                coarse_info.append((coarse, clusters.to(torch.long), fine_level))
                fine_level = coarse
            node_feat = coarse
        concat = torch.cat(pooled_features, dim=1)
        latent = self.backbone(concat)
        pg = self.pg_head(latent)
        qg = self.qg_head(latent)
        va = self.va_head(latent)
        vm = self.vm_head(latent)
        outputs = torch.cat([pg, qg, va, vm], dim=1)
        corrected = self.physics(outputs, data.pdqd)
        return corrected, {"coarse_info": coarse_info, "fine_features": data.pdqd.view(B, self.meta.n_bus, 2)}

