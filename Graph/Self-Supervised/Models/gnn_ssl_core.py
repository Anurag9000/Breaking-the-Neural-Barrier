
import math, random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================
# Optional PyG dependencies
# =============================
try:
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import SAGEConv, GCNConv, GATConv, GraphNorm, global_mean_pool, TopKPooling
    from torch_geometric.datasets import TUDataset
    PYG_OK = True
except Exception as e:
    PYG_OK = False
    PYG_ERR = e

# =============================
# SSL objectives
# =============================

def nt_xent(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    z1 = F.normalize(z1, dim=1); z2 = F.normalize(z2, dim=1)
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = torch.mm(z, z.t())  # cosine sims since normalized
    mask = torch.eye(2*B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -9e15)
    positives = torch.sum(z1 * z2, dim=1)  # (B,)
    pos = torch.cat([positives, positives], dim=0)  # (2B,)
    logits = sim / temperature
    labels = torch.arange(2*B, device=z.device)
    labels = (labels + B) % (2*B)  # each i matches i+B
    loss = F.cross_entropy(logits, labels)
    return loss

def barlow_twins(z1: torch.Tensor, z2: torch.Tensor, lambd: float = 5e-3) -> torch.Tensor:
    z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-9)
    z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-9)
    N, D = z1.shape
    c = (z1.T @ z2) / N  # D x D
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = (c - torch.diag(torch.diagonal(c))).pow_(2).sum()
    return on_diag + lambd * off_diag

# =============================
# Graph augmentations
# =============================

def dropout_edges(edge_index: torch.Tensor, drop_prob: float) -> torch.Tensor:
    E = edge_index.size(1)
    keep = torch.rand(E, device=edge_index.device) > drop_prob
    return edge_index[:, keep]

def dropout_features(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    mask = torch.rand_like(x) > drop_prob
    return x * mask

def random_node_drop(x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor, drop_prob: float):
    # Drop a fraction of nodes independently per graph in the batch
    device = x.device
    keep_mask = torch.ones(x.size(0), device=device, dtype=torch.bool)
    # For each graph id in batch:
    for gid in batch.unique():
        idx = (batch == gid).nonzero(as_tuple=False).flatten()
        n = idx.numel()
        if n <= 1: 
            continue
        k = int(n * (1 - drop_prob))
        if k < 1: k = 1
        kept = idx[torch.randperm(n, device=device)[:k]]
        keep_mask[idx] = False
        keep_mask[kept] = True
    # Reindex nodes
    new_x = x[keep_mask]
    new_batch = batch[keep_mask]
    # mapping old->new
    mapping = -torch.ones(x.size(0), device=device, dtype=torch.long)
    mapping[keep_mask] = torch.arange(new_x.size(0), device=device)
    src = mapping[edge_index[0]]; dst = mapping[edge_index[1]]
    valid = (src >= 0) & (dst >= 0)
    new_edge_index = torch.stack([src[valid], dst[valid]], dim=0)
    return new_x, new_edge_index, new_batch

def two_view_augment(data, feat_drop=0.2, edge_drop=0.2, node_drop=0.0):
    # data: Data(x, edge_index, batch, y?)
    def aug_once(d):
        x = d.x
        ei = d.edge_index
        batch = d.batch if hasattr(d, "batch") and d.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x1 = dropout_features(x, feat_drop) if feat_drop > 0 else x
        ei1 = dropout_edges(ei, edge_drop) if edge_drop > 0 else ei
        if node_drop > 0:
            x1, ei1, batch1 = random_node_drop(x1, ei1, batch, node_drop)
        else:
            batch1 = batch
        return type(d)(x=x1, edge_index=ei1, batch=batch1, y=getattr(d, "y", None))
    return aug_once(data), aug_once(data)

# =============================
# GNN blocks
# =============================

class GNNBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, conv_type: str = "sage", heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if not PYG_OK:
            raise RuntimeError(f"torch_geometric is required for GNNBlock. Import error: {PYG_ERR}")
        self.conv_type = conv_type
        self.heads = heads
        if conv_type == "sage":
            self.conv = SAGEConv(in_dim, out_dim)
            self.out_dim = out_dim
        elif conv_type == "gcn":
            self.conv = GCNConv(in_dim, out_dim)
            self.out_dim = out_dim
        elif conv_type == "gat":
            # For GAT, use multi-head and then project back to out_dim
            per_head = max(1, out_dim // heads)
            self.conv = GATConv(in_dim, per_head, heads=heads, concat=True, dropout=dropout)
            self.out_dim = per_head * heads
            if self.out_dim != out_dim:
                self.proj = nn.Linear(self.out_dim, out_dim)
            else:
                self.proj = nn.Identity()
        else:
            raise ValueError(f"Unknown conv_type {conv_type}")
        self.norm = GraphNorm(out_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        h = self.conv(x, edge_index)
        if hasattr(self, "proj"):
            h = self.proj(h)
        h = self.norm(h, batch)
        h = self.act(h)
        h = self.drop(h)
        return h

# =============================
# Adaptive GNN Encoder
# =============================

class AdaptiveGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], conv_type: str = "sage", heads: int = 4,
                 pooling_indices: List[int] = (), dropout: float = 0.0, proj_dim: int = 128):
        super().__init__()
        if not PYG_OK:
            raise RuntimeError(f"torch_geometric is required. Import error: {PYG_ERR}")
        assert len(hidden_dims) >= 1
        self.in_dim = in_dim
        self.hidden_dims = list(hidden_dims)
        self.conv_type = conv_type
        self.heads = heads
        self.pooling_indices = sorted(set(pooling_indices))
        self.dropout = dropout

        layers = nn.ModuleList()
        pools = nn.ModuleList()
        c = in_dim
        for i, h in enumerate(hidden_dims):
            layers.append(GNNBlock(c, h, conv_type=conv_type, heads=heads, dropout=dropout))
            c = h
            if i in self.pooling_indices:
                pools.append(TopKPooling(c, ratio=0.5))
            else:
                pools.append(nn.Identity())
        self.layers = layers
        self.pools = pools
        self.proj = nn.Linear(c, proj_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, "batch") and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        for i, (layer, pool) in enumerate(zip(self.layers, self.pools)):
            x = layer(x, edge_index, batch)
            if isinstance(pool, TopKPooling):
                x, edge_index, _, batch, _, _ = pool(x, edge_index, None, batch)
        # global pooling
        g = global_mean_pool(x, batch)
        z = self.proj(g)
        return x, z  # node features (last) and graph embedding

    @property
    def widths(self) -> List[int]:
        return self.hidden_dims

    def total_neurons(self) -> int:
        return sum(self.hidden_dims) + self.proj.in_features

    # ---------- mutations -----------
    def append_depth(self):
        last = self.hidden_dims[-1]
        self.layers.append(GNNBlock(last, last, conv_type=self.conv_type, heads=self.heads, dropout=self.dropout))
        self.pools.append(nn.Identity())
        self.hidden_dims.append(last)
        # projection input remains same (last)

    def widen_all(self, ex_k: int):
        if ex_k <= 0: return
        new_dims = [h + ex_k for h in self.hidden_dims]
        self._rebuild_layers(new_dims)

    def _rebuild_layers(self, new_dims: List[int]):
        # Build new stack and transplant overlapping parameters via state_dict key intersection
        new_layers = nn.ModuleList()
        new_pools = nn.ModuleList()
        c = self.in_dim
        for i, h in enumerate(new_dims):
            nb = GNNBlock(c, h, conv_type=self.conv_type, heads=self.heads, dropout=self.dropout)
            if i < len(self.layers):
                sb = self.layers[i]
                _safe_overlap_load(nb, sb)
            new_layers.append(nb)
            c = h
            if i < len(self.pools) and isinstance(self.pools[i], TopKPooling):
                new_pools.append(TopKPooling(c, ratio=0.5))
            else:
                new_pools.append(nn.Identity())
        # Resize projection
        new_proj = nn.Linear(c, self.proj.out_features)
        _safe_overlap_linear(new_proj, self.proj)
        self.layers = new_layers
        self.pools = new_pools
        self.hidden_dims = new_dims
        self.proj = new_proj

def _safe_overlap_load(dst: nn.Module, src: nn.Module):
    sd_dst = dst.state_dict()
    sd_src = src.state_dict()
    common = {k: v for k, v in sd_src.items() if k in sd_dst and sd_dst[k].shape == v.shape}
    sd_dst.update(common)
    dst.load_state_dict(sd_dst)

def _safe_overlap_linear(dst: nn.Linear, src: nn.Linear):
    with torch.no_grad():
        h = min(dst.weight.shape[0], src.weight.shape[0])
        w = min(dst.weight.shape[1], src.weight.shape[1])
        dst.weight[:h, :w].copy_(src.weight[:h, :w])
        if dst.bias is not None and src.bias is not None:
            b = min(dst.bias.shape[0], src.bias.shape[0])
            dst.bias[:b].copy_(src.bias[:b])

# =============================
# Data utilities
# =============================

def make_tu_loader(name: str, batch_size: int, num_workers: int = 0, val_split: float = 0.1, root: str = "./data"):
    if not PYG_OK:
        raise RuntimeError(f"torch_geometric is required. Import error: {PYG_ERR}")
    ds = TUDataset(root=root, name=name)
    # Ensure features exist
    if ds.num_features == 0:
        # Create degree features if none
        xs = []
        for g in ds:
            deg = torch.bincount(g.edge_index[0], minlength=g.num_nodes).float().unsqueeze(1)
            xs.append(deg)
        for i, g in enumerate(ds):
            g.x = xs[i]
    n = len(ds); n_val = int(n*val_split); n_train = n - n_val
    # simple split
    train_ds = ds[:n_train]; val_ds = ds[n_train:]; test_ds = ds[n_train:]  # use same as val for quick sanity
    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dl_test = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    in_dim = ds.num_features if ds.num_features > 0 else 1
    return dl_train, dl_val, dl_test, in_dim

# =============================
# Training wrapper
# =============================

@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0
    es_patience: int = 20
    grad_clip: Optional[float] = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    temperature: float = 0.2
    lambda_ntx: float = 1.0
    lambda_barlow: float = 0.0
    proj_dim: int = 128
    feat_drop: float = 0.2
    edge_drop: float = 0.2
    node_drop: float = 0.0

@dataclass
class SearchConfig:
    delta: float = 1e-3
    patience_width: int = 2
    patience_depth: int = 2
    ex_k: int = 16
    max_neurons: int = 1_200_000
    max_depth: int = 32
    max_width: int = 2048
    max_total_epochs: Optional[int] = None
    pooling_indices: Tuple[int, ...] = ()

class InnerTrainer:
    def __init__(self, net: AdaptiveGNN, tcfg: TrainConfig):
        self.net = net; self.tcfg = tcfg
        self.net.to(tcfg.device)
        self.optim = torch.optim.AdamW(net.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
        self.best = float("inf"); self.best_state = None; self.epochs_done = 0

    def _compute_loss_batch(self, batch):
        batch = batch.to(self.tcfg.device)
        d1, d2 = two_view_augment(batch, feat_drop=self.tcfg.feat_drop, edge_drop=self.tcfg.edge_drop, node_drop=self.tcfg.node_drop)
        d1 = d1.to(self.tcfg.device); d2 = d2.to(self.tcfg.device)
        _, z1 = self.net(d1); _, z2 = self.net(d2)
        loss = 0.0
        if self.tcfg.lambda_ntx > 0:
            loss = loss + self.tcfg.lambda_ntx * nt_xent(z1, z2, temperature=self.tcfg.temperature)
        if self.tcfg.lambda_barlow > 0:
            loss = loss + self.tcfg.lambda_barlow * barlow_twins(z1, z2)
        return loss

    @torch.no_grad()
    def _eval_epoch(self, loader):
        self.net.eval(); tot, n = 0.0, 0
        for batch in loader:
            l = self._compute_loss_batch(batch); bsz = batch.num_graphs
            tot += float(l.item()) * bsz; n += bsz
        return tot / max(n, 1)

    def fit(self, dl_train, dl_val, max_epochs=50):
        es = 0
        for _ in range(max_epochs):
            self.net.train()
            for batch in dl_train:
                self.optim.zero_grad(set_to_none=True)
                loss = self._compute_loss_batch(batch)
                loss.backward()
                if self.tcfg.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.net.parameters(), self.tcfg.grad_clip)
                self.optim.step()
            val = self._eval_epoch(dl_val); self.epochs_done += 1
            if val + 1e-12 < self.best:
                self.best = val
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                es = 0
            else:
                es += 1
            if es >= self.tcfg.es_patience:
                break
        if self.best_state is not None:
            self.net.load_state_dict(self.best_state)
        return self.best

# =============================
# Snapshot / restore / guards
# =============================

def snapshot(net: AdaptiveGNN):
    return {"state": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
            "hidden": net.hidden_dims.copy(),
            "pools": [isinstance(p, TopKPooling) for p in net.pools]}

def restore(net: AdaptiveGNN, snap):
    target_hidden = snap["hidden"]
    if net.hidden_dims != target_hidden:
        net._rebuild_layers(target_hidden)
    net.load_state_dict(snap["state"])

def can_widen(net: AdaptiveGNN, ex_k: int, scfg: SearchConfig) -> bool:
    if ex_k <= 0: return False
    if any(h + ex_k > scfg.max_width for h in net.hidden_dims): return False
    projected = net.total_neurons() + ex_k * (len(net.hidden_dims) + 1)
    return projected <= scfg.max_neurons

def can_deepen(net: AdaptiveGNN, scfg: SearchConfig) -> bool:
    if len(net.hidden_dims) + 1 > scfg.max_depth: return False
    projected = net.total_neurons() + net.hidden_dims[-1]
    return projected <= scfg.max_neurons

# =============================
# Six ADP searchers
# =============================

def _train_eval_val(net: AdaptiveGNN, dl_train, dl_val, tcfg: TrainConfig, max_epochs: int) -> float:
    return InnerTrainer(net, tcfg).fit(dl_train, dl_val, max_epochs=max_epochs)

def gnn_width_to_depth(net, dl_train, dl_val, tcfg, scfg, max_epochs=30):
    best_snap = snapshot(net); best_val = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs)
    width_fails = 0
    while width_fails < scfg.patience_width and can_widen(net, scfg.ex_k, scfg):
        pre = snapshot(net); net.widen_all(scfg.ex_k)
        v = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs)
        if v < best_val - scfg.delta:
            best_val = v; best_snap = snapshot(net)
            depth_fails = 0
            while depth_fails < scfg.patience_depth and can_deepen(net, scfg):
                pre2 = snapshot(net); net.append_depth()
                v2 = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs)
                if v2 < best_val - scfg.delta: best_val = v2; best_snap = snapshot(net)
                else: depth_fails += 1; restore(net, pre2)
        else:
            width_fails += 1; restore(net, pre)
    restore(net, best_snap); return net

def gnn_depth_to_width(net, dl_train, dl_val, tcfg, scfg, max_epochs=30):
    best_snap = snapshot(net); best_val = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs)
    depth_fails = 0
    while depth_fails < scfg.patience_depth and can_deepen(net, scfg):
        pre = snapshot(net); net.append_depth()
        v = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs)
        if v < best_val - scfg.delta:
            best_val = v; best_snap = snapshot(net)
            width_fails = 0
            while width_fails < scfg.patience_width and can_widen(net, scfg.ex_k, scfg):
                pre2 = snapshot(net); net.widen_all(scfg.ex_k)
                v2 = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs)
                if v2 < best_val - scfg.delta: best_val = v2; best_snap = snapshot(net)
                else: width_fails += 1; restore(net, pre2)
        else:
            depth_fails += 1; restore(net, pre)
    restore(net, best_snap); return net

def gnn_alt_depth_first(net, dl_train, dl_val, tcfg, scfg, max_epochs=30):
    best_snap = snapshot(net); best_val = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs)
    total = 0; ok = lambda e: scfg.max_total_epochs is None or e < scfg.max_total_epochs
    improved = True
    while improved and ok(total):
        improved = False
        depth_fails = 0
        while depth_fails < scfg.patience_depth and can_deepen(net, scfg) and ok(total):
            pre = snapshot(net); net.append_depth()
            v = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs); total += max_epochs
            if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(net); improved = True
            else: depth_fails += 1; restore(net, pre)
        width_fails = 0
        while width_fails < scfg.patience_width and can_widen(net, scfg.ex_k, scfg) and ok(total):
            pre = snapshot(net); net.widen_all(scfg.ex_k)
            v = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs); total += max_epochs
            if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(net); improved = True
            else: width_fails += 1; restore(net, pre)
    restore(net, best_snap); return net

def gnn_alt_width_first(net, dl_train, dl_val, tcfg, scfg, max_epochs=30):
    best_snap = snapshot(net); best_val = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs)
    total = 0; ok = lambda e: scfg.max_total_epochs is None or e < scfg.max_total_epochs
    improved = True
    while improved and ok(total):
        improved = False
        width_fails = 0
        while width_fails < scfg.patience_width and can_widen(net, scfg.ex_k, scfg) and ok(total):
            pre = snapshot(net); net.widen_all(scfg.ex_k)
            v = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs); total += max_epochs
            if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(net); improved = True
            else: width_fails += 1; restore(net, pre)
        depth_fails = 0
        while depth_fails < scfg.patience_depth and can_deepen(net, scfg) and ok(total):
            pre = snapshot(net); net.append_depth()
            v = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs); total += max_epochs
            if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(net); improved = True
            else: depth_fails += 1; restore(net, pre)
    restore(net, best_snap); return net

def gnn_depth_only(net, dl_train, dl_val, tcfg, scfg, max_epochs=30):
    best_snap = snapshot(net); best_val = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs); fails = 0
    while fails < scfg.patience_depth and can_deepen(net, scfg):
        pre = snapshot(net); net.append_depth()
        v = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs)
        if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(net)
        else: fails += 1; restore(net, pre)
    restore(net, best_snap); return net

def gnn_width_only(net, dl_train, dl_val, tcfg, scfg, max_epochs=30):
    best_snap = snapshot(net); best_val = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs); fails = 0
    while fails < scfg.patience_width and can_widen(net, scfg.ex_k, scfg):
        pre = snapshot(net); net.widen_all(scfg.ex_k)
        v = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs)
        if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(net)
        else: fails += 1; restore(net, pre)
    restore(net, best_snap); return net
