from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Autoencoder.common.core import ADPBackbone, PhysicsCorrection
from Autoencoder.common.data_utils import CaseMetadata


class PositionalEncoding(nn.Module):
    def __init__(self, n_tokens: int, dim: int):
        super().__init__()
        self.embedding = nn.Parameter(torch.randn(n_tokens, dim) * 0.01)

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.embedding.unsqueeze(0).repeat(batch_size, 1, 1)


class AETransformerModel(nn.Module):
    def __init__(
        self,
        meta: CaseMetadata,
        *,
        latent_dim: int,
        tx_dim: int,
        tx_layers: int,
        tx_heads: int,
        hidden: int,
        proj_newton_steps: int,
    ):
        super().__init__()
        self.meta = meta
        self.input_proj = nn.Linear(2, tx_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=tx_dim, nhead=tx_heads, dim_feedforward=hidden, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=tx_layers)
        self.pos_enc = PositionalEncoding(meta.n_bus, tx_dim)
        self.global_proj = nn.Linear(tx_dim, latent_dim)
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
        B = data.pdqd.size(0)
        x = data.pdqd.view(B, self.meta.n_bus, 2)
        tokens = self.input_proj(x)
        tokens = tokens + self.pos_enc(B).to(tokens.device)
        encoded = self.transformer(tokens)
        pooled = encoded.mean(dim=1)
        latent = F.relu(self.global_proj(pooled))
        features = self.backbone(latent)
        pg = self.pg_head(features)
        qg = self.qg_head(features)
        va = self.va_head(features)
        vm = self.vm_head(features)
        outputs = torch.cat([pg, qg, va, vm], dim=1)
        corrected = self.physics(outputs, data.pdqd)
        return corrected, {"latent": latent, "features": features}

