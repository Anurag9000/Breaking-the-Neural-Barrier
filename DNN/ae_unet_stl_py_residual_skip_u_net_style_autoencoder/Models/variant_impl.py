from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import GraphUNet

from Autoencoder.common.core import ADPBackbone, PhysicsCorrection
from Autoencoder.common.data_utils import CaseMetadata


class AEUNetModel(nn.Module):
    def __init__(
        self,
        meta: CaseMetadata,
        *,
        depth: int,
        hidden: int,
        latent_dim: int,
        graph_unet: bool,
        proj_newton_steps: int,
    ):
        super().__init__()
        self.meta = meta
        self.graph_unet = graph_unet
        if graph_unet:
            self.encoder = GraphUNet(in_channels=2, hidden_channels=hidden, out_channels=hidden, depth=depth)
        else:
            self.encoder = None
            self.input_lin = nn.Linear(2, hidden)
        self.backbone = ADPBackbone(hidden, hidden, depth=2, out_dim=hidden)
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
        x = data.x
        if self.graph_unet:
            features = self.encoder(x, data.edge_index)
        else:
            features = F.relu(self.input_lin(x))
        pooled = scatter_mean(features, data.batch, dim=0)
        latent = self.backbone(pooled)
        pg = self.pg_head(latent)
        qg = self.qg_head(latent)
        va = self.va_head(latent)
        vm = self.vm_head(latent)
        outputs = torch.cat([pg, qg, va, vm], dim=1)
        corrected = self.physics(outputs, data.pdqd)
        return corrected, {"node_features": features}

