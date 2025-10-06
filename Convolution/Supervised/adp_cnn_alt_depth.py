from __future__ import annotations
import math, os, copy
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# -----------------------------
# Model
# -----------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class AdaptiveCNN(nn.Module):
    def __init__(
        self,
        in_ch: int,
        num_classes: int,
        widths: List[int],
        pooling_indices: Iterable[int] | None = None,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.num_classes = num_classes
        self.widths = list(widths)
        self.pooling_indices = sorted(list(pooling_indices) if pooling_indices is not None else [])
        self.blocks = nn.ModuleList()

        ch = in_ch
        for i, w in enumerate(self.widths):
            self.blocks.append(ConvBlock(ch, w))
            ch = w
        self.head_gap = nn.AdaptiveAvgPool2d(1)
        self.head_fc = nn.Linear(ch, num_classes)

    # ---------- utility ----------
    def _make_from_widths(self, widths: List[int]):
        new = AdaptiveCNN(self.in_ch, self.num_classes, widths, self.pooling_indices)
        return new

    def _transplant_(self, other: "AdaptiveCNN"):
        """In-place copy overlapping weights from `other` into `self`.
        Shapes may differ due to width/depth changes; copy overlapping slices.
        """
        with torch.no_grad():
            for (b_new, b_old) in zip(self.blocks, other.blocks):
                # conv
                Wn, Wo = b_new.conv.weight, b_old.conv.weight
                oc = min(Wn.shape[0], Wo.shape[0])
                ic = min(Wn.shape[1], Wo.shape[1])
                ks = min(Wn.shape[2], Wo.shape[2])  # kernel is same (3), but keep general
                Wn[:oc, :ic, :ks, :ks].copy_(Wo[:oc, :ic, :ks, :ks])
                # bn
                for attr in ["weight", "bias", "running_mean", "running_var"]:
                    tn = getattr(b_new.bn, attr)
                    to = getattr(b_old.bn, attr)
                    c = min(tn.shape[0], to.shape[0])
                    tn[:c].copy_(to[:c])
            # head
            Wn, Wo = self.head_fc.weight, other.head_fc.weight
            bn, bo = self.head_fc.bias, other.head_fc.bias
            oc = min(Wn.shape[0], Wo.shape[0])
            ic = min(Wn.shape[1], Wo.shape[1])
            Wn[:oc, :ic].copy_(Wo[:oc, :ic])
            bn[:oc].copy_(bo[:oc])

    def snapshot(self):
        return {
            "state_dict": copy.deepcopy(self.state_dict()),
            "widths": copy.deepcopy(self.widths),
        }

    def restore(self, snap):
        widths = snap["widths"]
        rebuilt = self._make_from_widths(widths)
        rebuilt.load_state_dict(snap["state_dict"], strict=True)
        # swap all fields into self
        self.__dict__.update(rebuilt.__dict__)

    # ---------- growth ops ----------
    def widen_all(self, delta: int = 1):
        assert delta >= 1
        new_widths = [w + delta for w in self.widths]
        new_model = self._make_from_widths(new_widths)
        new_model._transplant_(self)
        self.__dict__.update(new_model.__dict__)

    def append_depth(self):
        last_w = self.widths[-1]
        new_widths = self.widths + [last_w]
        new_model = self._make_from_widths(new_widths)
        new_model._transplant_(self)
        self.__dict__.update(new_model.__dict__)

    def forward(self, x):
        h = x
        for i, blk in enumerate(self.blocks):
            h = blk(h)
            if i in self.pooling_indices:
                h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = self.head_gap(h).squeeze(-1).squeeze(-1)
        logits = self.head_fc(h)
        return logits

# -----------------------------
# Training / evaluation
# -----------------------------

def eval_loss(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, reduction="sum")
            total += loss.item()
            count += y.size(0)
    return total / max(1, count)

@dataclass
class InnerCfg:
    lr: float = 3e-4
    weight_decay: float = 5e-4
    max_epochs_inner: int = 1000000
    es_patience: int = 100
    grad_clip: Optional[float] = 1.0

@dataclass
class SearchCfg:
    delta: float = 0
    patience_width: int = 100
    patience_depth: int = 100
    max_total_epochs: int = 5000000

class InnerTrainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, cfg: InnerCfg):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cfg = cfg

    def run(self, remaining_budget: int) -> Tuple[float, dict, int]:
        """Train with early stopping on validation loss. Returns (best_val, best_snap, epochs_spent)."""
        if remaining_budget <= 0:
            return float("inf"), self.model.snapshot(), 0
        max_epochs = min(self.cfg.max_epochs_inner, remaining_budget)

        model = self.model
        device = self.device
        model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        best = float("inf")
        bad = 0
        epochs = 0
        best_snap = model.snapshot()

        for _ in range(max_epochs):
            model.train()
            for x, y in self.train_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                if self.cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
                opt.step()
            val = eval_loss(model, self.val_loader, device)
            epochs += 1
            if val + 1e-12 < best:
                best = val
                bad = 0
                best_snap = model.snapshot()
            else:
                bad += 1
                if bad >= self.cfg.es_patience:
                    break
        model.restore(best_snap)
        return best, best_snap, epochs

# -----------------------------
# Alternating search (DEPTH first in each cycle)
# -----------------------------

def alternating_adp_search_depth_first(
    model: AdaptiveCNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    inner_cfg: InnerCfg,
    search_cfg: SearchCfg,
):
    torch.backends.cudnn.benchmark = True

    trainer = InnerTrainer(model, train_loader, val_loader, device, inner_cfg)

    total_spent = 0
    best_val, best_snap, spent = trainer.run(remaining_budget=search_cfg.max_total_epochs)
    total_spent += spent
    baseline = best_val

    while total_spent < search_cfg.max_total_epochs:
        accepted_cycle = False

        # ---- DEPTH PHASE FIRST ----
        no_accept_d = 0
        while no_accept_d < search_cfg.patience_depth and total_spent < search_cfg.max_total_epochs:
            prop_snap = model.snapshot()
            model.append_depth()
            val, _, e = trainer.run(remaining_budget=search_cfg.max_total_epochs - total_spent)
            total_spent += e
            if val < baseline - search_cfg.delta:
                baseline = val
                best_snap = model.snapshot()
                no_accept_d = 0
                accepted_cycle = True
            else:
                model.restore(prop_snap)
                no_accept_d += 1

        # ---- WIDTH PHASE SECOND ----
        no_accept_w = 0
        while no_accept_w < search_cfg.patience_width and total_spent < search_cfg.max_total_epochs:
            prop_snap = model.snapshot()
            model.widen_all(+1)
            val, _, e = trainer.run(remaining_budget=search_cfg.max_total_epochs - total_spent)
            total_spent += e
            if val < baseline - search_cfg.delta:
                baseline = val
                best_snap = model.snapshot()
                no_accept_w = 0
                accepted_cycle = True
            else:
                model.restore(prop_snap)
                no_accept_w += 1

        if not accepted_cycle:
            break

    model.restore(best_snap)
    return model, baseline, total_spent
