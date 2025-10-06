
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Adaptive CNN with Alternating Width-Depth Growth (W<->D)
# Mechanism:
#   Start depth=1, width=1.
#   Phase W: keep depth fixed, increment width by +1 across all layers; early-stop; accept if val_loss improves by > delta.
#   Phase D: keep width fixed, append one block (width stays same as previous); early-stop; accept if improves by > delta.
#   Alternate W then D; stop when an entire cycle has no accepted changes (both phases plateau).

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class AdaptiveCNN(nn.Module):
    # A simple stack of Conv-BN-ReLU blocks followed by GAP and a Linear classifier.
    # widths: list[int] of out_channels for each block (depth = len(widths))
    # pooling_indices: indices (0-based) after which we apply a 2x2 max-pool
    def __init__(self, in_ch: int, num_classes: int, widths: List[int], pooling_indices: Optional[List[int]] = None):
        super().__init__()
        assert len(widths) >= 1, "Depth must be at least 1"
        self.in_ch = in_ch
        self.num_classes = num_classes
        self.pooling_indices = sorted(pooling_indices or [])
        self.widths = list(widths)
        self.blocks = nn.ModuleList()
        last = in_ch
        for w in self.widths:
            self.blocks.append(ConvBNReLU(last, w, 3, 1, 1))
            last = w
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.widths[-1], num_classes)

    def forward(self, x):
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.pooling_indices:
                x = F.max_pool2d(x, 2)
        x = self.gap(x).flatten(1)
        return self.head(x)

    def _rebuild_with_widths(self, new_widths: List[int]):
        # Rebuild blocks/head to match new_widths. Attempt to copy overlapping weights.
        old_blocks = self.blocks
        self.widths = list(new_widths)
        self.blocks = nn.ModuleList()
        last = self.in_ch
        for i, w in enumerate(self.widths):
            new_blk = ConvBNReLU(last, w, 3, 1, 1)
            if i < len(old_blocks):
                old_blk = old_blocks[i]
                with torch.no_grad():
                    # conv overlap copy
                    k = new_blk.conv.weight
                    k_old = old_blk.conv.weight
                    c_out = min(k.shape[0], k_old.shape[0])
                    c_in = min(k.shape[1], k_old.shape[1])
                    k[:c_out, :c_in].copy_(k_old[:c_out, :c_in])
                    # bn stats/affine
                    new_bn, old_bn = new_blk.bn, old_blk.bn
                    c = min(new_bn.num_features, old_bn.num_features)
                    new_bn.running_mean[:c].copy_(old_bn.running_mean[:c])
                    new_bn.running_var[:c].copy_(old_bn.running_var[:c])
                    if new_bn.affine and old_bn.affine:
                        new_bn.weight.data[:c].copy_(old_bn.weight.data[:c])
                        new_bn.bias.data[:c].copy_(old_bn.bias.data[:c])
            self.blocks.append(new_blk)
            last = w
        # head (overlap copy)
        old_head = self.head
        self.head = nn.Linear(self.widths[-1], self.num_classes)
        with torch.no_grad():
            w = self.head.weight
            b = self.head.bias
            if isinstance(old_head, nn.Linear):
                w_old = old_head.weight
                b_old = old_head.bias
                c_out = min(w.shape[0], w_old.shape[0])
                c_in = min(w.shape[1], w_old.shape[1])
                w[:c_out, :c_in].copy_(w_old[:c_out, :c_in])
                b[:c_out].copy_(b_old[:c_out])

    def append_depth(self):
        # Append one block with width equal to the current last width.
        new_widths = self.widths + [self.widths[-1]]
        self._rebuild_with_widths(new_widths)

    def widen_all(self, delta_w: int = 1):
        # Increase all block widths by +delta_w uniformly.
        assert delta_w >= 1
        new_widths = [w + delta_w for w in self.widths]
        self._rebuild_with_widths(new_widths)

    def snapshot(self) -> Dict:
        return {"state_dict": deepcopy(self.state_dict()), "widths": list(self.widths)}

    def restore(self, snap: Dict):
        target_widths = list(snap["widths"])
        self._rebuild_with_widths(target_widths)
        self.load_state_dict(deepcopy(snap["state_dict"]), strict=True)

    @property
    def depth(self) -> int:
        return len(self.widths)

    @property
    def width(self) -> int:
        return self.widths[-1]

@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    lr: float = 3e-4
    weight_decay: float = 5e-4
    max_epochs_inner: int = 200          # per proposal cap (early-stopping usually ends earlier)
    es_patience: int = 10
    grad_clip: float = 1.0

@dataclass
class SearchConfig:
    delta: float = 1e-3                  # min val-loss improvement to accept
    patience_width: int = 3              # plateau detector for width (+1 each try)
    patience_depth: int = 3              # plateau detector for depth (+1 each try)
    max_total_epochs: int = 5000         # global budget across all proposals
    pooling_indices: Tuple[int, ...] = (0,)  # default pool schedule

def _run_inner_train(model: AdaptiveCNN,
                     train_loader: DataLoader,
                     val_loader: DataLoader,
                     tcfg: TrainConfig,
                     remaining_budget: int) -> Tuple[float, Dict, int]:
    # Train with early-stopping on val loss. Returns (best_val_loss, best_snapshot, epochs_spent).
    device = tcfg.device
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    best = math.inf
    bad = 0
    epochs = 0
    best_snap = model.snapshot()

    max_epochs = min(tcfg.max_epochs_inner, remaining_budget)
    if max_epochs <= 0:
        return best, best_snap, 0

    for _ in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            if tcfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()

        # val
        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                val_loss += loss.item() * xb.size(0)
                n += xb.size(0)
        val_loss /= max(1, n)

        epochs += 1
        if val_loss + 1e-12 < best:
            best = val_loss
            bad = 0
            best_snap = model.snapshot()
        else:
            bad += 1
            if bad >= tcfg.es_patience:
                break

    # restore best
    model.restore(best_snap)
    return best, best_snap, epochs

def alternating_adp_search(
    in_ch: int,
    num_classes: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tcfg: TrainConfig,
    scfg: SearchConfig,
    seed_width: int = 1,
    seed_depth: int = 1,
):
    # Execute W->D->W->... alternating growth until a full cycle yields no accepted changes.
    # Both width and depth start at 1; increments are +1.
    torch.backends.cudnn.benchmark = True

    widths = [seed_width for _ in range(seed_depth)]
    model = AdaptiveCNN(in_ch=in_ch, num_classes=num_classes, widths=widths, pooling_indices=list(scfg.pooling_indices))
    device = tcfg.device
    model.to(device)

    total_spent = 0

    # Baseline train
    best_val, best_snap, spent = _run_inner_train(model, train_loader, val_loader, tcfg, scfg.max_total_epochs - total_spent)
    total_spent += spent
    baseline_loss = best_val

    while total_spent < scfg.max_total_epochs:
        accepted_in_cycle = False

        # Width phase (depth fixed)
        no_accept_w = 0
        while no_accept_w < scfg.patience_width and total_spent < scfg.max_total_epochs:
            prop_snap = model.snapshot()
            model.widen_all(delta_w=1)

            prop_val, _, spent = _run_inner_train(model, train_loader, val_loader, tcfg, scfg.max_total_epochs - total_spent)
            total_spent += spent

            if prop_val < baseline_loss - scfg.delta:
                baseline_loss = prop_val
                best_snap = model.snapshot()
                no_accept_w = 0
                accepted_in_cycle = True
            else:
                model.restore(prop_snap)
                no_accept_w += 1

        # Depth phase (width fixed)
        no_accept_d = 0
        while no_accept_d < scfg.patience_depth and total_spent < scfg.max_total_epochs:
            prop_snap = model.snapshot()
            model.append_depth()

            prop_val, _, spent = _run_inner_train(model, train_loader, val_loader, tcfg, scfg.max_total_epochs - total_spent)
            total_spent += spent

            if prop_val < baseline_loss - scfg.delta:
                baseline_loss = prop_val
                best_snap = model.snapshot()
                no_accept_d = 0
                accepted_in_cycle = True
            else:
                model.restore(prop_snap)
                no_accept_d += 1

        if not accepted_in_cycle:
            break

    model.restore(best_snap)
    return model, baseline_loss, total_spent
