
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================
# Optional PyG dependencies
# =============================
try:
    from torch_geometric.nn import SAGEConv, GCNConv, GATConv, GraphNorm
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T
    PYG_OK = True
except Exception as e:
    PYG_OK = False
    PYG_ERR = e

# =============================
# GNN block
# =============================

class GNNBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, conv_type: str = "sage", heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if not PYG_OK:
            raise RuntimeError(f"torch_geometric is required for GNNBlock. Import error: {PYG_ERR}")
        self.conv_type = conv_type
        self.heads = heads
        self.dropout = dropout
        if conv_type == "sage":
            self.conv = SAGEConv(in_dim, out_dim)
            real_out = out_dim
            self.post = nn.Identity()
        elif conv_type == "gcn":
            self.conv = GCNConv(in_dim, out_dim)
            real_out = out_dim
            self.post = nn.Identity()
        elif conv_type == "gat":
            per_head = max(1, out_dim // heads)
            self.conv = GATConv(in_dim, per_head, heads=heads, concat=True, dropout=dropout)
            real_out = per_head * heads
            self.post = nn.Linear(real_out, out_dim) if real_out != out_dim else nn.Identity()
        else:
            raise ValueError(f"Unknown conv_type {conv_type}")
        self.norm = GraphNorm(out_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch=None):
        h = self.conv(x, edge_index)
        h = self.post(h)
        # For node classification with one large graph, batch is None; GraphNorm expects batch,
        # so fall back to LayerNorm in that case.
        if batch is None:
            h = F.layer_norm(h, h.shape[-1:])
        else:
            h = self.norm(h, batch)
        h = self.act(h)
        h = self.drop(h)
        return h

# =============================
# Adaptive GNN for NODE classification
# =============================

class AdaptiveGNNNode(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], num_classes: int,
                 conv_type: str = "sage", heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if not PYG_OK:
            raise RuntimeError(f"torch_geometric is required. Import error: {PYG_ERR}")
        assert len(hidden_dims) >= 1
        self.in_dim = in_dim
        self.hidden_dims = list(hidden_dims)
        self.num_classes = num_classes
        self.conv_type = conv_type
        self.heads = heads
        self.dropout = dropout

        layers = nn.ModuleList()
        c = in_dim
        for h in self.hidden_dims:
            layers.append(GNNBlock(c, h, conv_type=self.conv_type, heads=self.heads, dropout=self.dropout))
            c = h
        self.layers = layers
        self.cls = nn.Linear(c, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        h = x
        for layer in self.layers:
            h = layer(h, edge_index, batch=None)
        logits = self.cls(h)
        return logits

    @property
    def widths(self) -> List[int]:
        return self.hidden_dims

    def total_neurons(self) -> int:
        return sum(self.hidden_dims) + self.cls.in_features + self.cls.out_features

    # ---------- mutations -----------
    def append_depth(self):
        last = self.hidden_dims[-1]
        self.layers.append(GNNBlock(last, last, conv_type=self.conv_type, heads=self.heads, dropout=self.dropout))
        self.hidden_dims.append(last)

    def widen_all(self, ex_k: int):
        if ex_k <= 0: return
        new_dims = [h + ex_k for h in self.hidden_dims]
        self._rebuild_layers(new_dims)

    def _rebuild_layers(self, new_dims: List[int]):
        new_layers = nn.ModuleList()
        c = self.in_dim
        for i, h in enumerate(new_dims):
            nb = GNNBlock(c, h, conv_type=self.conv_type, heads=self.heads, dropout=self.dropout)
            if i < len(self.layers):
                _safe_overlap_load(nb, self.layers[i])
            new_layers.append(nb)
            c = h
        new_cls = nn.Linear(c, self.cls.out_features)
        _safe_overlap_linear(new_cls, self.cls)
        self.layers = new_layers
        self.hidden_dims = new_dims
        self.cls = new_cls

# =============================
# Overlap helpers
# =============================

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
# Data utilities (Planetoid)
# =============================

def load_planetoid(name: str, root: str = "./data"):
    if not PYG_OK:
        raise RuntimeError(f"torch_geometric is required. Import error: {PYG_ERR}")
    dataset = Planetoid(root=root, name=name, transform=T.NormalizeFeatures())
    data = dataset[0]
    in_dim = dataset.num_features
    num_classes = dataset.num_classes
    return data, in_dim, num_classes

# =============================
# Training & Search configs
# =============================

@dataclass
class TrainConfig:
    lr: float = 1e-2
    weight_decay: float = 5e-4
    es_patience: int = 200
    grad_clip: Optional[float] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class SearchConfig:
    delta: float = 1e-3
    patience_width: int = 2
    patience_depth: int = 2
    ex_k: int = 16
    max_neurons: int = 1_200_000
    max_depth: int = 16
    max_width: int = 4096
    max_total_epochs: Optional[int] = None

# =============================
# Trainer (full-batch)
# =============================

class NodeTrainer:
    def __init__(self, net: AdaptiveGNNNode, tcfg: TrainConfig, data):
        self.net = net; self.tcfg = tcfg; self.data = data.to(tcfg.device)
        self.net.to(tcfg.device)
        self.optim = torch.optim.Adam(self.net.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
        self.best = float("inf"); self.best_state = None; self.epochs_done = 0

    def _compute_losses(self):
        self.net.train()
        logits = self.net(self.data)
        loss = F.cross_entropy(logits[self.data.train_mask], self.data.y[self.data.train_mask])
        with torch.no_grad():
            self.net.eval()
            logits_val = self.net(self.data)
            val_loss = F.cross_entropy(logits_val[self.data.val_mask], self.data.y[self.data.val_mask])
        return loss, float(val_loss.item())

    def _eval_test(self):
        self.net.eval()
        logits = self.net(self.data)
        y = self.data.y
        pred = logits.argmax(dim=1)
        def acc(mask): 
            m = mask.bool()
            return float((pred[m] == y[m]).sum().item()) / max(int(m.sum().item()), 1)
        return acc(self.data.train_mask), acc(self.data.val_mask), acc(self.data.test_mask)

    def fit(self, max_epochs=2000):
        es = 0
        for _ in range(max_epochs):
            loss, val = self._compute_losses()
            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            if self.tcfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.net.parameters(), self.tcfg.grad_clip)
            self.optim.step()
            self.epochs_done += 1
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

def snapshot(net: AdaptiveGNNNode):
    return {"state": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
            "hidden": net.hidden_dims.copy()}

def restore(net: AdaptiveGNNNode, snap):
    if net.hidden_dims != snap["hidden"]:
        net._rebuild_layers(snap["hidden"])
    net.load_state_dict(snap["state"])

def can_widen(net: AdaptiveGNNNode, ex_k: int, scfg: SearchConfig) -> bool:
    if ex_k <= 0: return False
    if any(h + ex_k > scfg.max_width for h in net.hidden_dims): return False
    projected = net.total_neurons() + ex_k * (len(net.hidden_dims) + 1)
    return projected <= scfg.max_neurons

def can_deepen(net: AdaptiveGNNNode, scfg: SearchConfig) -> bool:
    if len(net.hidden_dims) + 1 > scfg.max_depth: return False
    projected = net.total_neurons() + net.hidden_dims[-1]
    return projected <= scfg.max_neurons

# =============================
# Six ADP searchers
# =============================

def _train_eval_val(net: AdaptiveGNNNode, data, tcfg: TrainConfig, max_epochs: int) -> float:
    return NodeTrainer(net, tcfg, data).fit(max_epochs=max_epochs)

def gnn_node_width_to_depth(net, data, tcfg, scfg, max_epochs=400):
    best_snap = snapshot(net); best_val = _train_eval_val(net, data, tcfg, max_epochs)
    width_fails = 0
    while width_fails < scfg.patience_width and can_widen(net, scfg.ex_k, scfg):
        pre = snapshot(net); net.widen_all(scfg.ex_k)
        v = _train_eval_val(net, data, tcfg, max_epochs)
        if v < best_val - scfg.delta:
            best_val = v; best_snap = snapshot(net)
            depth_fails = 0
            while depth_fails < scfg.patience_depth and can_deepen(net, scfg):
                pre2 = snapshot(net); net.append_depth()
                v2 = _train_eval_val(net, data, tcfg, max_epochs)
                if v2 < best_val - scfg.delta: best_val = v2; best_snap = snapshot(net)
                else: depth_fails += 1; restore(net, pre2)
        else:
            width_fails += 1; restore(net, pre)
    restore(net, best_snap); return net

def gnn_node_depth_to_width(net, data, tcfg, scfg, max_epochs=400):
    best_snap = snapshot(net); best_val = _train_eval_val(net, data, tcfg, max_epochs)
    depth_fails = 0
    while depth_fails < scfg.patience_depth and can_deepen(net, scfg):
        pre = snapshot(net); net.append_depth()
        v = _train_eval_val(net, data, tcfg, max_epochs)
        if v < best_val - scfg.delta:
            best_val = v; best_snap = snapshot(net)
            width_fails = 0
            while width_fails < scfg.patience_width and can_widen(net, scfg.ex_k, scfg):
                pre2 = snapshot(net); net.widen_all(scfg.ex_k)
                v2 = _train_eval_val(net, data, tcfg, max_epochs)
                if v2 < best_val - scfg.delta: best_val = v2; best_snap = snapshot(net)
                else: width_fails += 1; restore(net, pre2)
        else:
            depth_fails += 1; restore(net, pre)
    restore(net, best_snap); return net

def gnn_node_alt_depth_first(net, data, tcfg, scfg, max_epochs=400):
    best_snap = snapshot(net); best_val = _train_eval_val(net, data, tcfg, max_epochs)
    total = 0; ok = lambda e: scfg.max_total_epochs is None or e < scfg.max_total_epochs
    improved = True
    while improved and ok(total):
        improved = False
        depth_fails = 0
        while depth_fails < scfg.patience_depth and can_deepen(net, scfg) and ok(total):
            pre = snapshot(net); net.append_depth()
            v = _train_eval_val(net, data, tcfg, max_epochs); total += max_epochs
            if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(net); improved = True
            else: depth_fails += 1; restore(net, pre)
        width_fails = 0
        while width_fails < scfg.patience_width and can_widen(net, scfg.ex_k, scfg) and ok(total):
            pre = snapshot(net); net.widen_all(scfg.ex_k)
            v = _train_eval_val(net, data, tcfg, max_epochs); total += max_epochs
            if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(net); improved = True
            else: width_fails += 1; restore(net, pre)
    restore(net, best_snap); return net

def gnn_node_alt_width_first(net, data, tcfg, scfg, max_epochs=400):
    best_snap = snapshot(net); best_val = _train_eval_val(net, data, tcfg, max_epochs)
    total = 0; ok = lambda e: scfg.max_total_epochs is None or e < scfg.max_total_epochs
    improved = True
    while improved and ok(total):
        improved = False
        width_fails = 0
        while width_fails < scfg.patience_width and can_widen(net, scfg.ex_k, scfg) and ok(total):
            pre = snapshot(net); net.widen_all(scfg.ex_k)
            v = _train_eval_val(net, data, tcfg, max_epochs); total += max_epochs
            if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(net); improved = True
            else: width_fails += 1; restore(net, pre)
        depth_fails = 0
        while depth_fails < scfg.patience_depth and can_deepen(net, scfg) and ok(total):
            pre = snapshot(net); net.append_depth()
            v = _train_eval_val(net, data, tcfg, max_epochs); total += max_epochs
            if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(net); improved = True
            else: depth_fails += 1; restore(net, pre)
    restore(net, best_snap); return net

def gnn_node_depth_only(net, data, tcfg, scfg, max_epochs=400):
    best_snap = snapshot(net); best_val = _train_eval_val(net, data, tcfg, max_epochs); fails = 0
    while fails < scfg.patience_depth and can_deepen(net, scfg):
        pre = snapshot(net); net.append_depth()
        v = _train_eval_val(net, data, tcfg, max_epochs)
        if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(net)
        else: fails += 1; restore(net, pre)
    restore(net, best_snap); return net

def gnn_node_width_only(net, data, tcfg, scfg, max_epochs=400):
    best_snap = snapshot(net); best_val = _train_eval_val(net, data, tcfg, max_epochs); fails = 0
    while fails < scfg.patience_width and can_widen(net, scfg.ex_k, scfg):
        pre = snapshot(net); net.widen_all(scfg.ex_k)
        v = _train_eval_val(net, data, tcfg, max_epochs)
        if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(net)
        else: fails += 1; restore(net, pre)
    restore(net, best_snap); return net
