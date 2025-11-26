"""Data helper utilities for autoencoder variants."""
from __future__ import annotations

from typing import Dict, Optional, List
from collections import deque

import torch
from torch_geometric.data import Data

from pathlib import Path
import json

from Dyn_DNN4OPF.data.opf_loader import load_case_bounds, load_cost_coeff

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "data"


class CaseMetadata:
    def __init__(self, case_name: str):
        raw = load_case_bounds(case_name)
        self.case_name = case_name
        self.from_bus = torch.tensor(raw["from_bus"], dtype=torch.long)
        self.to_bus = torch.tensor(raw["to_bus"], dtype=torch.long)
        self.n_bus = len(raw["v_min"])
        self.n_gen = len(raw["p_min"])
        self.v_min = raw["v_min"]
        self.v_max = raw["v_max"]
        self.p_min = raw["p_min"]
        self.p_max = raw["p_max"]
        self.q_min = raw["q_min"]
        self.q_max = raw["q_max"]
        self.gen_bus_idx = torch.tensor(raw["gen_buses"], dtype=torch.long)
        self.load_bus_idx = torch.tensor(raw["load_buses"], dtype=torch.long)
        self.y_bus = raw["y_bus"]
        self.bus_types = raw.get("bus_types", torch.ones(self.n_bus, dtype=torch.long))
        self.shunt_b = raw.get("shunt_b", torch.zeros(self.n_bus, dtype=torch.float32))
        self.shunt_g = raw.get("shunt_g", torch.zeros(self.n_bus, dtype=torch.float32))
        self.line_features = raw.get("line_features", torch.zeros((len(self.from_bus), 1), dtype=torch.float32)).float()
        self.transformer_features = raw.get("transformer_features", torch.zeros((0,), dtype=torch.float32)).float()
        if self.line_features.ndim == 1:
            self.line_features = self.line_features.unsqueeze(1)
        self.n_line = self.line_features.size(0)

        self._edge_index = self._build_edge_index()
        self.adjacency = self._build_adjacency_list()
        degree = torch.tensor([len(neigh) for neigh in self.adjacency], dtype=torch.float32)
        self.degree = degree
        max_deg = max(degree.max().item(), 1.0)
        self.degree_norm = degree / max_deg
        self.bfs_order = self._compute_bfs_order()
        self.inverse_bfs = torch.empty(self.n_bus, dtype=torch.long)
        self.inverse_bfs[self.bfs_order] = torch.arange(self.n_bus, dtype=torch.long)

        sample_file = DATASET_ROOT / f"sample_{case_name.split('_')[2]}.json"
        with open(sample_file, "r", encoding="utf-8") as f:
            json_data = json.load(f)
        self.cost_coeffs = load_cost_coeff(json_data)

    @property
    def edge_index(self) -> torch.Tensor:
        return self._edge_index

    def _build_edge_index(self) -> torch.Tensor:
        edges = torch.stack([self.from_bus, self.to_bus], dim=0)
        return torch.cat([edges, edges.flip(0)], dim=1)

    def _build_adjacency_list(self) -> List[List[int]]:
        neighbors: List[List[int]] = [[] for _ in range(self.n_bus)]
        for u, v in zip(self.from_bus.tolist(), self.to_bus.tolist()):
            neighbors[u].append(v)
            neighbors[v].append(u)
        for idx in range(self.n_bus):
            neighbors[idx] = sorted(set(neighbors[idx]))
        return neighbors

    def _compute_bfs_order(self) -> torch.Tensor:
        if (self.bus_types == 3).any():
            start = int(torch.nonzero(self.bus_types == 3, as_tuple=True)[0][0].item())
        else:
            start = 0
        visited = [False] * self.n_bus
        order: List[int] = []
        queue: deque[int] = deque()
        queue.append(start)
        visited[start] = True
        while queue:
            u = queue.popleft()
            order.append(u)
            for v in self.adjacency[u]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        for node in range(self.n_bus):
            if not visited[node]:
                queue.append(node)
                visited[node] = True
                while queue:
                    u = queue.popleft()
                    order.append(u)
                    for v in self.adjacency[u]:
                        if not visited[v]:
                            visited[v] = True
                            queue.append(v)
        return torch.tensor(order, dtype=torch.long)


def build_bus_features(x_flat: torch.Tensor, n_bus: int) -> torch.Tensor:
    return x_flat.view(x_flat.size(0), 2, n_bus).transpose(1, 2).reshape(-1, 2)


def collate_graph_batch(xb: torch.Tensor, meta: CaseMetadata) -> Data:
    B = xb.size(0)
    device = xb.device
    bus_x = build_bus_features(xb, meta.n_bus)
    base_edge = meta.edge_index.to(device)
    edges = []
    offset = meta.n_bus
    for b in range(B):
        edges.append(base_edge + b * offset)
    edge_index = torch.cat(edges, dim=1)
    batch = torch.arange(B, device=device).repeat_interleave(meta.n_bus)
    data = Data(x=bus_x, edge_index=edge_index)
    data.batch = batch
    data.pdqd = xb
    data.edge_index = edge_index
    return data

