"""
ADP-CNN Depth-Only (mask-free) for CIFAR-100
────────────────────────────────────────────
This mirrors the line-by-line control flow of your ADP-DEN depth-only variant,
but refactored for a CNN classifier with NO constraints/bounded-activation logic.

Core behavior:
• Inner loop  : early-stopping on validation loss
• Outer loop  : append one conv block (Conv-BN-ReLU) of same width as last
                accept expansion iff best_val_loss < pre_exp_val_loss - delta
                allow up to `trials_depth` consecutive failed expansions before rollback
                final model = best-by-validation-loss snapshot (weights + architecture)

Extras:
• Saves/updates a plot every ~60s: X = total "neurons" (sum of conv out_channels + head.in_features), Y = best val loss
• Verbose logging at: epochs, expansions, accept/reject, patience counters, plot updates, rollbacks, final restore
"""

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
# Ensure at least one stream handler with a simple format (avoid duplicate handlers)
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
# Utility
# ════════════════════════════════════════════════════════════════════════

def _total_neurons(m: "AdaptiveCNN") -> int:
    # for parity with MLP "sum of widths": sum of conv out_channels + head.in_features
    s = sum(b.conv.out_channels for b in m.convs)
    s += m.head.in_features
    return int(s)

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
        self.global_epoch_count = 0

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
        # head fan-in unchanged (still w)

    def widen_all(self, ex_k: int):
        # Not used in this depth-only class, kept for parity/testing
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
        # Match architecture depth
        if len(tgt) != len(self.convs):
            new_convs = nn.ModuleList()
            prev_in = self.convs[0].conv.in_channels if self.convs else 3
            for w in tgt:
                blk = ConvBNReLU(prev_in, w, 3, 1, 1).to(dev)
                new_convs.append(blk)
                prev_in = w
            self.convs = new_convs
        else:
            # Resize in-place to match each width
            prev_out = None
            for i, (blk, w) in enumerate(zip(self.convs, tgt)):
                in_ch = blk.conv.in_channels if prev_out is None else prev_out
                if blk.conv.in_channels != in_ch or blk.conv.out_channels != w:
                    blk.conv = _resize_conv2d(blk.conv, w, in_ch)
                    blk.bn   = _resize_bn2d(blk.bn, w)
                prev_out = w
        # Head fan-in must match last width
        last = tgt[-1]
        if self.head.in_features != last:
            self.head = _resize_head(self.head, last).to(dev)
        # Load weights
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
    logger.info(f"[Inner] start early-stop train | budget_epochs={max_epochs} | patience={patience} | lr={lr} | wd={weight_decay}")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce  = nn.CrossEntropyLoss()
    best_state = copy.deepcopy(model.state_dict())
    best_val   = math.inf
    best_acc   = 0.0
    bad = 0
    ran = 0

    last_plot_t = time.time()

    for epoch in range(max_epochs):
        # ---- Train one epoch ----
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

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

        # ---- Validation ----
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

        # 🔹 Epoch log now includes depth, last_width, and full widths list
        logger.info(
            f"[Inner][epoch {ran}/{max_epochs} | global_epoch={global_epoch_ref[0] if global_epoch_ref else 'NA'}] "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | "
            f"best_val={best_val:.4f} | best_acc={best_acc:.4f} | bad={bad}/{patience} | "
            f"neurons={_total_neurons(model)} | depth={_depth(model)} | last_width={_last_width(model)} | widths={_widths_list(model)}"
        )

        # periodic plot update
        if plot_path is not None and plot_history is not None:
            now = time.time()
            if now - last_plot_t >= plot_interval_s:
                plot_history.append((_total_neurons(model), best_val))
                _save_plot(plot_path, plot_history)
                logger.info(f"[Plot] updated '{plot_path}' (points={len(plot_history)})")
                last_plot_t = now

        if bad >= patience:
            logger.info(f"[Inner] early-stop triggered at epoch={ran} (bad={bad} >= patience={patience})")
            break

    model.load_state_dict(best_state)
    logger.info(f"[Inner] done | epochs_run={ran} | best_val={best_val:.4f} | best_acc={best_acc:.4f}")
    return best_state, best_val, best_acc, ran

def _depth(m: "AdaptiveCNN") -> int:
    return len(m.convs)

def _last_width(m: "AdaptiveCNN") -> int:
    # uses conv out_channels (same as bn.num_features)
    return m.convs[-1].conv.out_channels

def _widths_list(m: "AdaptiveCNN") -> List[int]:
    return [b.conv.out_channels for b in m.convs]

def _save_plot(path: str, hist: List[Tuple[int, float]]):
    if not hist:
        return
    xs, ys = zip(*hist)

    # Guard for log scale (no zeros/negatives)
    xs = [max(int(x), 1) for x in xs]
    ys = [max(float(y), 1e-12) for y in ys]

    plt.figure(figsize=(6, 4))
    plt.semilogy(xs, ys, marker="o")  # <- Y is log scale
    plt.xlabel("Total neurons (channels sum + head fan-in)")
    plt.ylabel("Best validation loss (log)")
    plt.title("Architecture vs Validation Loss (updated)")
    plt.grid(True, ls="--", alpha=0.5)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# ════════════════════════════════════════════════════════════════════════
# Algorithm: ADP_CNN_DepthOnly (mask-free)
# ════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    delta: float = 0.0
    trials_depth: int = 100
    patience: int = 100
    max_epochs: int = 1000000
    init_widths: List[int] = None
    num_classes: int = 100
    pooling_indices: Optional[List[int]] = None
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_neurons: int = 1_000_000  # large guard
    # Optional: width patience for parity in logs (not used by depth-only search)
    trials_width: Optional[int] = None

class ADP_CNN_DepthOnly:
    def __init__(self, config: Optional[Config] = None, device: Optional[torch.device] = None, plot_path: Optional[str] = None):
        cfg = config or Config(init_widths=[64, 64, 128, 128])
        self.config = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = AdaptiveCNN(3, cfg.init_widths or [64,64,128,128], cfg.num_classes, cfg.pooling_indices).to(self.device)
        self.delta = float(cfg.delta)
        self.trials_depth = int(cfg.trials_depth)
        self.patience = int(cfg.patience)
        self.max_epochs = int(cfg.max_epochs)
        self.lr = float(cfg.lr)
        self.weight_decay = float(cfg.weight_decay)
        self.max_neurons = int(cfg.max_neurons)
        self.global_epoch = [0]
        # plot state
        self.plot_path = plot_path or os.path.join("results_adp_cnn", f"{self.__class__.__name__}_neuron_loss_plot.png")
        self.plot_history: List[Tuple[int, float]] = []

        # Initial architecture log
        init_neurons = _total_neurons(self.model)
        logger.info(
            f"[Init] widths={ [b.bn.num_features for b in self.model.convs] } | "
            f"pooling_indices={sorted(self.model.pooling_indices)} | "
            f"num_classes={cfg.num_classes} | neurons={init_neurons}"
        )
        logger.info(
            f"[Config] delta={self.delta} | patience(inner)={self.patience} | "
            f"trials_depth(outer)={self.trials_depth} | trials_width(outer)={cfg.trials_width if cfg.trials_width is not None else 'N/A'} | "
            f"max_epochs={self.max_epochs} | lr={self.lr} | wd={self.weight_decay} | max_neurons={self.max_neurons}"
        )

    def fit(self, train_loader, val_loader):
        device = self.device

        # ── baseline training ──
        logger.info("[Outer] baseline training start")
        best_state, base_loss, base_acc, _ = train_with_early_stop(
            self.model, train_loader, val_loader, device,
            max_epochs=self.max_epochs, patience=self.patience,
            lr=self.lr, weight_decay=self.weight_decay,
            global_epoch_ref=self.global_epoch,
            plot_path=self.plot_path, plot_history=self.plot_history, plot_interval_s=60.0,
        )
        baseline_snapshot = self.model.snapshot_state()
        baseline_loss     = base_loss
        logger.info(f"[Outer] baseline done | baseline_val_loss={baseline_loss:.4f} | baseline_val_acc={base_acc:.4f} | neurons={_total_neurons(self.model)}")

        depth_failures    = 0
        just_expanded     = False
        pre_exp_snapshot  = None
        pre_exp_loss      = None

        # Outer loop: depth expansions
        while self.global_epoch[0] < self.max_epochs:
            # ---- Compare post-exp baseline vs pre-exp baseline (acceptance) ----
            if just_expanded:
                # acceptance check vs. pre-exp
                accepted_vs_pre = (baseline_loss + 1e-12) < (pre_exp_loss - self.delta)
                logger.info(
                    f"[Outer] post-exp evaluation | baseline_val_loss={baseline_loss:.6f} | "
                    f"pre_exp_val_loss={pre_exp_loss:.6f} | delta={self.delta:.6f} | "
                    f"accepted_vs_pre={accepted_vs_pre} | depth_failures={depth_failures}/{self.trials_depth}"
                )
                if accepted_vs_pre:
                    # accepted
                    depth_failures = 0
                    just_expanded = False
                    logger.info("[Outer] expansion retained (accepted vs pre-exp)")
                else:
                    depth_failures += 1
                    logger.info(f"[Outer] expansion did not beat pre-exp by delta; failures={depth_failures}/{self.trials_depth}")
                    if depth_failures >= self.trials_depth:
                        logger.info("[Outer] depth patience exhausted; rollback to pre-exp snapshot and stop expanding")
                        self.model.restore_state(pre_exp_snapshot)
                        baseline_snapshot = pre_exp_snapshot
                        baseline_loss     = pre_exp_loss
                        break
                    # try another expansion
                    just_expanded = False
            else:
                # Establish acceptance baseline for upcoming expansion
                pre_exp_snapshot = baseline_snapshot
                pre_exp_loss     = baseline_loss
                logger.info(f"[Outer] set pre-exp baseline | pre_exp_val_loss={pre_exp_loss:.6f}")

            # Capacity guard
            total_neurons = _total_neurons(self.model)
            if total_neurons >= self.max_neurons:
                logger.info(f"[Outer] Max neuron budget reached ({total_neurons} >= {self.max_neurons}); stopping expansions.")
                break

            # ---- Propose: append one conv block (same width as last) ----
            arch_snapshot_before = copy.deepcopy(self.model)  # full arch+weights snapshot
            before_depth = len(self.model.convs)
            before_width = self.model.last_width
            logger.info(
                f"[Outer] PROPOSE depth+1 | depth_before={before_depth} -> {before_depth+1} | "
                f"last_width={before_width} | neurons_before={total_neurons} | "
                f"patience(inner)={self.patience} | patience_depth(outer)={self.trials_depth} | "
                f"patience_width(outer)={self.config.trials_width if self.config.trials_width is not None else 'N/A'}"
            )
            self.model.append_depth(device=device)

            # ---- Train proposal (remaining epoch budget) ----
            remaining_epochs = max(self.max_epochs - self.global_epoch[0], 0)
            logger.info(f"[Outer] train proposal | remaining_epochs_budget={remaining_epochs}")
            _, prop_loss, prop_acc, _ = train_with_early_stop(
                self.model, train_loader, val_loader, device,
                max_epochs=remaining_epochs,
                patience=self.patience, lr=self.lr, weight_decay=self.weight_decay,
                global_epoch_ref=self.global_epoch,
                plot_path=self.plot_path, plot_history=self.plot_history, plot_interval_s=60.0,
            )
            after_neurons = _total_neurons(self.model)
            logger.info(
                f"[Outer] proposal result | prop_val_loss={prop_loss:.6f} | prop_val_acc={prop_acc:.4f} | "
                f"baseline_val_loss={baseline_loss:.6f} | delta={self.delta:.6f} | neurons_after={after_neurons}"
            )

            # ---- Decide accept/reject vs current baseline ----
            if prop_loss + 1e-12 < baseline_loss - self.delta:
                baseline_snapshot = self.model.snapshot_state()
                baseline_loss     = prop_loss
                depth_failures    = 0
                just_expanded     = True  # next loop compares against pre-exp
                logger.info("[Outer] ACCEPT proposal (improved baseline by delta)")
            else:
                # Reject: rollback architecture & weights to snapshot_before, then ensure baseline weights
                logger.info("[Outer] REJECT proposal (no sufficient improvement); rolling back")
                self.model = arch_snapshot_before.to(device)
                self.model.restore_state(baseline_snapshot)
                depth_failures += 1
                logger.info(f"[Outer] depth_failures incremented -> {depth_failures}/{self.trials_depth}")
                if depth_failures >= self.trials_depth:
                    logger.info("[Outer] depth patience exhausted; stopping expansions")
                    break
                just_expanded = False

            if self.global_epoch[0] >= self.max_epochs:
                logger.info("[Outer] global epoch budget exhausted; stopping expansions")
                break

        # Final: restore best baseline weights
        logger.info("[Outer] final restore to best baseline snapshot")
        self.model.restore_state(baseline_snapshot)
        logger.info(f"[Outer] done | final_val_loss={baseline_loss:.6f} | neurons={_total_neurons(self.model)}")

    @torch.no_grad()
    def evaluate(self, loader) -> Tuple[float, float]:
        return evaluate(self.model, loader, self.device)
