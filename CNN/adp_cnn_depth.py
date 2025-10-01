"""
ADP-CNN Depth→Width (mask-free) for CIFAR-100
──────────────────────────────────────────────
Faithful CNN port of your ADPDepth MLP algorithm:
• Inner loop  : early-stopping on validation loss (CrossEntropy)
• Outer loops :
    1) Depth series  — append one ConvBNReLU (same width as last) per try, with patience `trials_depth` on acceptance.
    2) Width series  — after an accepted depth, widen ALL conv blocks by `ex_k` channels per try, with patience `trials_width`.
   Acceptance rule for either series: best_val_loss < pre_series_val_loss - delta.
   On exceeding the patience, rollback to the pre-series baseline (arch + weights).
• Final model : best-by-validation-loss snapshot (weights + architecture).

Extras:
• Plot auto-saves/updates every ~60s to results_adp_cnn/<ClassName>_neuron_loss_plot.png
  X = total "neurons" (sum of conv out_channels + head fan-in), Y = best val loss.
• CIFAR-100 defaults. No constraints and no bounded-activation logic.
"""

from __future__ import annotations
import os
import time
import copy
import math
import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    _ch = logging.StreamHandler()
    _ch.setLevel(logging.INFO)
    _ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(_ch)

# ════════════════════════════════════════════════════════════════════════
# Resizing helpers (overlap-copy) — conv, batchnorm, linear
# ════════════════════════════════════════════════════════════════════════

def _overlap_copy_(dst: torch.Tensor, src: torch.Tensor):
    if dst is None or src is None:
        return
    common = tuple(min(d, s) for d, s in zip(dst.shape, src.shape))
    if not common:
        return
    ds = tuple(slice(0, c) for c in common)
    ss = tuple(slice(0, c) for c in common)
    with torch.no_grad():
        dst[ds] = src[ss]

def _resize_conv2d(old: nn.Conv2d, new_out: int, new_in: Optional[int] = None) -> nn.Conv2d:
    if new_in is None:
        new_in = old.in_channels
    new = nn.Conv2d(
        in_channels=new_in, out_channels=new_out, kernel_size=old.kernel_size,
        stride=old.stride, padding=old.padding, dilation=old.dilation,
        groups=1, bias=True, padding_mode=old.padding_mode
    ).to(old.weight.device)
    _overlap_copy_(new.weight, old.weight)
    if old.bias is not None and new.bias is not None:
        _overlap_copy_(new.bias, old.bias)
    return new

def _resize_bn2d(old: nn.BatchNorm2d, new_features: int) -> nn.BatchNorm2d:
    new = nn.BatchNorm2d(new_features).to(next(old.parameters()).device)
    if old.affine:
        _overlap_copy_(new.weight.data, old.weight.data)
        _overlap_copy_(new.bias.data,   old.bias.data)
    _overlap_copy_(new.running_mean, old.running_mean)
    _overlap_copy_(new.running_var,  old.running_var)
    return new

def _resize_linear(old: nn.Linear, new_out: int, new_in: Optional[int] = None) -> nn.Linear:
    if new_in is None:
        new_in = old.in_features
    new = nn.Linear(new_in, new_out, bias=True).to(old.weight.device)
    _overlap_copy_(new.weight, old.weight)
    if old.bias is not None and new.bias is not None:
        _overlap_copy_(new.bias, old.bias)
    return new

def _resize_head(old: nn.Linear, new_in: int) -> nn.Linear:
    return _resize_linear(old, old.out_features, new_in)

# ════════════════════════════════════════════════════════════════════════
# Utilities (logging helpers)
# ════════════════════════════════════════════════════════════════════════

def _total_neurons(m: "AdaptiveCNN") -> int:
    s = sum(b.conv.out_channels for b in m.convs)
    s += m.head.in_features
    return int(s)

def _depth(m: "AdaptiveCNN") -> int:
    return len(m.convs)

def _last_width(m: "AdaptiveCNN") -> int:
    return m.convs[-1].conv.out_channels

def _widths_list(m: "AdaptiveCNN") -> List[int]:
    return [b.conv.out_channels for b in m.convs]

# ════════════════════════════════════════════════════════════════════════
# Blocks & Adaptive CNN
# ════════════════════════════════════════════════════════════════════════

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=True)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class AdaptiveCNN(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        widths: List[int] = [64, 64, 128, 128],
        num_classes: int = 100,
        pooling_indices: Optional[List[int]] = None
    ):
        super().__init__()
        self.pooling_indices = set(pooling_indices or [1, 3])
        self.convs = nn.ModuleList()
        prev = in_ch
        for w in widths:
            self.convs.append(ConvBNReLU(prev, w, 3, 1, 1))
            prev = w
        self.gap  = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(prev, num_classes)

    @property
    def last_width(self) -> int:
        return self.convs[-1].bn.num_features

    def forward(self, x):
        for i, block in enumerate(self.convs):
            x = block(x)
            if i in self.pooling_indices:
                x = F.max_pool2d(x, 2)
        x = self.gap(x).flatten(1)
        return self.head(x)

    # -------- expansions --------
    def append_depth(self, device: Optional[torch.device] = None):
        w = self.last_width
        block = ConvBNReLU(w, w, 3, 1, 1)
        if device is None:
            device = next(self.parameters()).device
        block.to(device)
        self.convs.append(block)

    def widen_all(self, ex_k: int):
        new_convs = nn.ModuleList()
        prev_out = None
        for blk in self.convs:
            old_in  = blk.conv.in_channels
            old_out = blk.conv.out_channels
            new_in  = prev_out if prev_out is not None else old_in
            new_out = old_out + ex_k
            new_blk = ConvBNReLU(new_in, new_out, 3, 1, 1)
            new_blk.conv = _resize_conv2d(blk.conv, new_out, new_in)
            new_blk.bn   = _resize_bn2d(blk.bn, new_out)
            new_blk.to(next(self.parameters()).device)
            new_convs.append(new_blk)
            prev_out = new_out
        self.convs = new_convs
        self.head  = _resize_head(self.head, prev_out)

    # -------- snapshots --------
    def snapshot_state(self) -> Dict[str, object]:
        return dict(
            state_dict = copy.deepcopy(self.state_dict()),
            widths     = [b.bn.num_features for b in self.convs]
        )

    def restore_state(self, snap: Dict[str, object]):
        dev = next(self.parameters()).device
        tgt = snap["widths"]
        # Rebuild depth if needed
        if len(tgt) != len(self.convs):
            new_convs = nn.ModuleList()
            prev_in = self.convs[0].conv.in_channels if self.convs else 3
            for w in tgt:
                blk = ConvBNReLU(prev_in, w, 3, 1, 1).to(dev)
                new_convs.append(blk)
                prev_in = w
            self.convs = new_convs
        else:
            prev_out = None
            for blk, w in zip(self.convs, tgt):
                in_ch = blk.conv.in_channels if prev_out is None else prev_out
                if blk.conv.in_channels != in_ch or blk.conv.out_channels != w:
                    blk.conv = _resize_conv2d(blk.conv, w, in_ch)
                    blk.bn   = _resize_bn2d(blk.bn, w)
                prev_out = w
        last = tgt[-1]
        if self.head.in_features != last:
            self.head = _resize_head(self.head, last).to(dev)
        self.load_state_dict(snap["state_dict"], strict=True)

# ════════════════════════════════════════════════════════════════════════
# Training utilities
# ════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    ce = nn.CrossEntropyLoss()
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = ce(logits, y)
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, total_correct / total

def _save_plot(path: str, hist: List[Tuple[int, float]]):
    if not hist:
        return
    xs, ys = zip(*hist)
    xs = [max(int(x), 1) for x in xs]
    ys = [max(float(y), 1e-12) for y in ys]
    plt.figure(figsize=(6, 4))
    plt.semilogy(xs, ys, marker="o")
    plt.xlabel("Total neurons (channels sum + head fan-in)")
    plt.ylabel("Best validation loss (log scale)")
    plt.title("Architecture vs Validation Loss (updated)")
    plt.grid(True, ls="--", alpha=0.5)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def train_with_early_stop(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    max_epochs: int,
    patience: int,
    lr: float,
    weight_decay: float,
    global_epoch_ref: Optional[List[int]] = None,
    plot_path: Optional[str] = None,
    plot_history: Optional[List[Tuple[int, float]]] = None,
    plot_interval_s: float = 60.0,
):
    logger.info(f"[Inner] start early-stop | budget_epochs={max_epochs} | patience(inner)={patience} | lr={lr} | wd={weight_decay}")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce  = nn.CrossEntropyLoss()
    best_state = copy.deepcopy(model.state_dict())
    best_val   = math.inf
    best_acc   = 0.0
    bad = 0
    ran = 0
    last_plot_t = time.time()

    for _ in range(max_epochs):
        # train one epoch
        model.train()
        train_loss_sum = 0.0
        train_correct  = 0
        train_total    = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = x.size(0)
            train_loss_sum += loss.item() * bs
            train_correct  += (logits.argmax(1) == y).sum().item()
            train_total    += bs

        val_loss, val_acc = evaluate(model, val_loader, device)
        ran += 1
        if global_epoch_ref is not None:
            global_epoch_ref[0] += 1

        train_loss = train_loss_sum / max(train_total, 1)
        improved = val_loss + 1e-12 < best_val
        if improved:
            best_val = val_loss
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1

        # per-epoch log with architecture stats
        logger.info(
            f"[Inner][epoch {ran}/{max_epochs} | global_epoch={global_epoch_ref[0] if global_epoch_ref else 'NA'}] "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
            f"best_val={best_val:.4f} | best_acc={best_acc:.4f} | bad={bad}/{patience} | "
            f"neurons={_total_neurons(model)} | depth={_depth(model)} | last_width={_last_width(model)} | widths={_widths_list(model)}"
        )

        # plot update
        if plot_path is not None and plot_history is not None:
            now = time.time()
            if now - last_plot_t >= plot_interval_s:
                plot_history.append((_total_neurons(model), best_val))
                _save_plot(plot_path, plot_history)
                logger.info(f"[Plot] updated '{plot_path}' (points={len(plot_history)})")
                last_plot_t = now

        if bad >= patience:
            logger.info(f"[Inner] early-stop at epoch={ran} (bad={bad} >= patience={patience})")
            break

    model.load_state_dict(best_state)
    logger.info(f"[Inner] done | epochs_run={ran} | best_val={best_val:.4f} | best_acc={best_acc:.4f}")
    return best_state, best_val, best_acc

# ════════════════════════════════════════════════════════════════════════
# Algorithm: ADP_CNN_Depth (Depth series → Width series)
# ════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    delta: float = 0.0
    trials_depth: int = 2
    trials_width: int = 2
    patience: int = 15
    max_epochs: int = 140
    init_widths: List[int] = None
    num_classes: int = 100
    pooling_indices: Optional[List[int]] = None
    lr: float = 1e-3
    weight_decay: float = 1e-4
    ex_k: int = 16
    max_neurons: int = 1_000_000
    max_depth: int = 128
    max_width: int = 10_000

class ADP_CNN_Depth:
    def __init__(self, config: Optional[Config] = None, device: Optional[torch.device] = None, plot_path: Optional[str] = None):
        cfg = config or Config(init_widths=[64, 64, 128, 128])
        self.config = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = AdaptiveCNN(3, cfg.init_widths or [64,64,128,128], cfg.num_classes, cfg.pooling_indices).to(self.device)
        self.delta = float(cfg.delta)
        self.trials_depth = int(cfg.trials_depth)
        self.trials_width = int(cfg.trials_width)
        self.patience = int(cfg.patience)
        self.max_epochs = int(cfg.max_epochs)
        self.lr = float(cfg.lr)
        self.weight_decay = float(cfg.weight_decay)
        self.ex_k = int(cfg.ex_k)
        self.max_neurons = int(cfg.max_neurons)
        self.max_depth = int(cfg.max_depth)
        self.max_width = int(cfg.max_width)
        self.global_epoch = [0]
        self.plot_path = plot_path or os.path.join("results_adp_cnn", f"{self.__class__.__name__}_neuron_loss_plot.png")
        self.plot_history: List[Tuple[int, float]] = []

        logger.info(
            f"[Init] widths={_widths_list(self.model)} | depth={_depth(self.model)} | "
            f"pooling_indices={sorted(self.model.pooling_indices)} | num_classes={cfg.num_classes} | "
            f"neurons={_total_neurons(self.model)}"
        )
        logger.info(
            f"[Config] delta={self.delta} | patience(inner)={self.patience} | "
            f"trials_depth={self.trials_depth} | trials_width={self.trials_width} | "
            f"max_epochs={self.max_epochs} | lr={self.lr} | wd={self.weight_decay} | "
            f"ex_k={self.ex_k} | max_neurons={self.max_neurons} | max_depth={self.max_depth} | max_width={self.max_width}"
        )

    def fit(self, train_loader, val_loader):
        device = self.device

        # ── Baseline training ──
        logger.info("[Outer] baseline training start")
        _, base_loss, _ = train_with_early_stop(
            self.model, train_loader, val_loader, device,
            max_epochs=self.max_epochs, patience=self.patience,
            lr=self.lr, weight_decay=self.weight_decay,
            global_epoch_ref=self.global_epoch,
            plot_path=self.plot_path, plot_history=self.plot_history, plot_interval_s=60.0,
        )
        baseline_snapshot = self.model.snapshot_state()
        baseline_loss     = base_loss
        logger.info(
            f"[Outer] baseline done | baseline_val_loss={baseline_loss:.4f} | "
            f"neurons={_total_neurons(self.model)} | depth={_depth(self.model)} | widths={_widths_list(self.model)}"
        )

        # ───────────────────────────── DEPTH SERIES ─────────────────────────────
        depth_failures = 0
        while True:
            if self.global_epoch[0] >= self.max_epochs:
                logger.info("[Outer] global epoch budget exhausted; stopping")
                break

            # Guards
            if _depth(self.model) >= self.max_depth:
                logger.info(f"[Depth] max_depth reached ({_depth(self.model)} >= {self.max_depth}); stop depth series.")
                break
            if _total_neurons(self.model) >= self.max_neurons:
                logger.info(f"[Depth] max_neurons reached ({_total_neurons(self.model)} >= {self.max_neurons}); stop depth series.")
                break

            # Pre-depth baseline
            pre_depth_snapshot = baseline_snapshot
            pre_depth_loss     = baseline_loss
            logger.info(
                f"[Depth] PROPOSE depth+1 | depth_before={_depth(self.model)} -> {_depth(self.model)+1} | "
                f"last_width={_last_width(self.model)} | widths_before={_widths_list(self.model)} | "
                f"neurons_before={_total_neurons(self.model)} | patience_depth={self.trials_depth}"
            )

            # Proposal: append one block
            arch_snapshot_before = copy.deepcopy(self.model)
            self.model.append_depth(device=device)

            # Train proposal
            remaining_epochs = max(self.max_epochs - self.global_epoch[0], 0)
            logger.info(f"[Depth] train proposal | remaining_epochs_budget={remaining_epochs}")
            _, depth_loss, _ = train_with_early_stop(
                self.model, train_loader, val_loader, device,
                max_epochs=remaining_epochs, patience=self.patience,
                lr=self.lr, weight_decay=self.weight_decay,
                global_epoch_ref=self.global_epoch,
                plot_path=self.plot_path, plot_history=self.plot_history, plot_interval_s=60.0,
            )
            logger.info(
                f"[Depth] proposal result | prop_val_loss={depth_loss:.6f} | "
                f"pre_depth_val_loss={pre_depth_loss:.6f} | delta={self.delta:.6f} | "
                f"neurons_after={_total_neurons(self.model)} | depth_after={_depth(self.model)} | widths_after={_widths_list(self.model)}"
            )

            # Decide
            if depth_loss + 1e-12 < pre_depth_loss - self.delta:
                logger.info("[Depth] ACCEPT proposal (improved by delta); reset depth_failures")
                baseline_snapshot = self.model.snapshot_state()
                baseline_loss     = depth_loss
                depth_failures    = 0

                # ────────────────────── WIDTH SERIES (after accepted depth) ──────────────────────
                width_failures = 0
                pre_width_snapshot = baseline_snapshot
                pre_width_loss     = baseline_loss
                while True:
                    if self.global_epoch[0] >= self.max_epochs:
                        logger.info("[Width] global epoch budget exhausted; stop width series.")
                        break

                    # Capacity guards for width
                    neurons_now = _total_neurons(self.model)
                    depth_now   = _depth(self.model)
                    last_w      = _last_width(self.model)
                    projected_neurons = neurons_now + self.ex_k * depth_now
                    if projected_neurons > self.max_neurons or (last_w + self.ex_k) > self.max_width:
                        logger.info(
                            f"[Width] capacity guard hit | projected_neurons={projected_neurons} (limit {self.max_neurons}), "
                            f"next_last_width={last_w + self.ex_k} (limit {self.max_width})"
                        )
                        break

                    logger.info(
                        f"[Width] PROPOSE width+w | ex_k={self.ex_k} | widths_before={_widths_list(self.model)} | "
                        f"depth={depth_now} | neurons_before={neurons_now} | patience_width={self.trials_width}"
                    )
                    arch_snapshot_before_w = copy.deepcopy(self.model)
                    self.model.widen_all(self.ex_k)

                    # Train width proposal
                    remaining_epochs = max(self.max_epochs - self.global_epoch[0], 0)
                    logger.info(f"[Width] train proposal | remaining_epochs_budget={remaining_epochs}")
                    _, width_loss, _ = train_with_early_stop(
                        self.model, train_loader, val_loader, device,
                        max_epochs=remaining_epochs, patience=self.patience,
                        lr=self.lr, weight_decay=self.weight_decay,
                        global_epoch_ref=self.global_epoch,
                        plot_path=self.plot_path, plot_history=self.plot_history, plot_interval_s=60.0,
                    )
                    logger.info(
                        f"[Width] proposal result | prop_val_loss={width_loss:.6f} | "
                        f"pre_width_val_loss={pre_width_loss:.6f} | delta={self.delta:.6f} | "
                        f"neurons_after={_total_neurons(self.model)} | widths_after={_widths_list(self.model)}"
                    )

                    if width_loss + 1e-12 < pre_width_loss - self.delta:
                        logger.info("[Width] ACCEPT proposal (improved by delta); reset width_failures")
                        baseline_snapshot = self.model.snapshot_state()
                        baseline_loss     = width_loss
                        pre_width_snapshot = baseline_snapshot
                        pre_width_loss     = baseline_loss
                        width_failures = 0
                    else:
                        logger.info("[Width] REJECT proposal; rolling back")
                        self.model = arch_snapshot_before_w.to(device)
                        self.model.restore_state(baseline_snapshot)
                        width_failures += 1
                        logger.info(f"[Width] width_failures={width_failures}/{self.trials_width}")
                        if width_failures >= self.trials_width:
                            logger.info("[Width] patience exhausted; rollback to pre-width-series baseline and stop width series.")
                            self.model.restore_state(pre_width_snapshot)
                            baseline_snapshot = pre_width_snapshot
                            baseline_loss     = pre_width_loss
                            break

                # After width series, ensure we are at best baseline
                self.model.restore_state(baseline_snapshot)

            else:
                logger.info("[Depth] REJECT proposal; rolling back")
                self.model = arch_snapshot_before.to(device)
                self.model.restore_state(baseline_snapshot)
                depth_failures += 1
                logger.info(f"[Depth] depth_failures={depth_failures}/{self.trials_depth}")
                if depth_failures >= self.trials_depth:
                    logger.info("[Depth] patience exhausted; stopping outer loop.")
                    break

        # Final: restore best baseline
        logger.info("[Outer] final restore to best baseline snapshot")
        self.model.restore_state(baseline_snapshot)
        logger.info(
            f"[Outer] done | final_val_loss={baseline_loss:.6f} | neurons={_total_neurons(self.model)} | "
            f"depth={_depth(self.model)} | widths={_widths_list(self.model)}"
        )

    @torch.no_grad()
    def evaluate(self, loader) -> Tuple[float, float]:
        return evaluate(self.model, loader, self.device)
