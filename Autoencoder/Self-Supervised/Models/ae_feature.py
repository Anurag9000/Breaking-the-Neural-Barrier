import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Utility block ( CNN style)
# -----------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# -----------------------------
# Feature Reconstruction Convolutional Autoencoder (single-model)
# - Reconstructs *engineered* features of the input (Sobel edges or HOG-like orientation bins)
# - 0-based pooling indices during forward (mirrors  models)
# -----------------------------
class FeatureConvAE(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        widths: List[int] = [32, 64, 128],
        pooling_indices: List[int] = [0, 2],
        out_ch: int = 1,  # target feature channels (1 for Sobel mag; B for HOG-like)
    ):
        super().__init__()
        assert len(widths) >= 1
        self.in_ch = in_ch
        self.widths = list(widths)
        self.pooling_indices = set(pooling_indices)
        self.out_ch = int(out_ch)

        # Encoder
        enc = []
        ch = in_ch
        for w in widths:
            enc.append(ConvBNReLU(ch, w))
            ch = w
        self.encoder = nn.ModuleList(enc)

        # Decoder (mirror widths)
        rev = list(reversed(widths))
        dec = []
        ch = rev[0]
        for w in rev[1:]:
            dec.append(ConvBNReLU(ch, w))
            ch = w
        self.decoder = nn.ModuleList(dec)
        self.head = nn.Conv2d(ch, out_ch, kernel_size=1, stride=1, padding=0)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    # capacity stats
    def total_neurons(self) -> int:
        enc_neurons = sum(m.conv.out_channels for m in self.encoder)
        dec_neurons = sum(m.conv.out_channels for m in self.decoder)
        return enc_neurons + dec_neurons

    def depth(self) -> int:
        return len(self.widths)

    def widths_list(self) -> List[int]:
        return list(self.widths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        down_ct = 0
        for i, blk in enumerate(self.encoder):
            h = blk(h)
            if i in self.pooling_indices:
                h = self.pool(h)
                down_ct += 1
        z = h
        for j, blk in enumerate(self.decoder):
            if down_ct > 0:
                z = self.upsample(z)
                down_ct -= 1
            z = blk(z)
        while down_ct > 0:
            z = self.upsample(z)
            down_ct -= 1
        out = self.head(z)
        return out

# -----------------------------
# Target feature computation utilities (no-grad)
# -----------------------------

def _rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
    # x: (B,3,H,W) normalized image (mean/std doesn't matter for edges); return (B,1,H,W)
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

@torch.no_grad()
def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    # x: (B,3,H,W) or (B,1,H,W) -> (B,1,H,W) magnitude in [0,1] approx
    if x.size(1) == 3:
        xg = _rgb_to_gray(x)
    else:
        xg = x
    device = x.device
    gx_k = torch.tensor([[1, 0, -1],[2, 0, -2],[1, 0, -1]], dtype=x.dtype, device=device).view(1,1,3,3)
    gy_k = torch.tensor([[1, 2, 1],[0, 0, 0],[-1, -2, -1]], dtype=x.dtype, device=device).view(1,1,3,3)
    Gx = F.conv2d(xg, gx_k, padding=1)
    Gy = F.conv2d(xg, gy_k, padding=1)
    mag = torch.sqrt(Gx*Gx + Gy*Gy)
    # normalize per-image robustly
    mag = mag / (mag.amax(dim=(2,3), keepdim=True) + 1e-8)
    return mag.clamp(0, 1)

@torch.no_grad()
def hog_like(x: torch.Tensor, bins: int = 8) -> torch.Tensor:
    # x: (B,3,H,W) or (B,1,H,W) -> (B,bins,H,W) soft orientation histogram per pixel (no cell/block normalization)
    if x.size(1) == 3:
        xg = _rgb_to_gray(x)
    else:
        xg = x
    device = x.device
    gx_k = torch.tensor([[1, 0, -1],[2, 0, -2],[1, 0, -1]], dtype=x.dtype, device=device).view(1,1,3,3)
    gy_k = torch.tensor([[1, 2, 1],[0, 0, 0],[-1, -2, -1]], dtype=x.dtype, device=device).view(1,1,3,3)
    Gx = F.conv2d(xg, gx_k, padding=1)
    Gy = F.conv2d(xg, gy_k, padding=1)
    mag = torch.sqrt(Gx*Gx + Gy*Gy) + 1e-12
    ang = torch.atan2(Gy, Gx)  # [-pi, pi]
    ang = (ang + math.pi) / (2*math.pi)  # -> [0,1)
    # soft binning into bins
    B, _, H, W = xg.shape
    ang = ang.squeeze(1)
    mag = mag.squeeze(1)
    # bin centers in [0,1)
    centers = torch.linspace(0, 1, bins+1, device=device)[:-1] + 0.5/bins  # (bins,)
    # compute distance to centers (circular)
    ang_exp = ang.unsqueeze(1)  # (B,1,H,W)
    centers_exp = centers.view(1, bins, 1, 1)
    dist = torch.abs(ang_exp - centers_exp)
    dist = torch.minimum(dist, 1 - dist)  # wrap-around
    # triangular kernel weights
    weights = torch.clamp(1 - dist * bins, min=0.0)
    # apply magnitude
    out = weights * mag.unsqueeze(1)  # (B,bins,H,W)
    # normalize per-pixel max to 1 for stability
    maxv = out.amax(dim=1, keepdim=True) + 1e-8
    out = out / maxv
    return out

# -----------------------------
# Trainer (feature reconstruction), early stopping, plotting
# -----------------------------
import time
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class FeatureAETrainer:
    def __init__(
        self,
        model: FeatureConvAE,
        device: torch.device,
        feature_type: str = "edge",   # "edge" (Sobel mag) or "hog"
        hog_bins: int = 8,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        es_patience: int = 30,
        max_epochs: int = 300,
        results_dir: str = "results_ae_feature",
        loss_type: str = "mse",  # or "l1"
    ):
        self.model = model.to(device)
        self.device = device
        self.feature_type = feature_type
        self.hog_bins = int(hog_bins)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.grad_clip = grad_clip
        self.es_patience = es_patience
        self.max_epochs = max_epochs
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        if loss_type == "mse":
            self.crit = nn.MSELoss(reduction="mean")
        elif loss_type == "l1":
            self.crit = nn.L1Loss(reduction="mean")
        else:
            raise ValueError("loss_type must be 'mse' or 'l1'")
        self.loss_type = loss_type

        self.best_val = float("inf")
        self.best_state = None
        self.hist = {"train_loss": [], "val_loss": [], "neurons": [], "epoch": []}
        self._last_plot_ts = 0.0

    @torch.no_grad()
    def _make_target(self, x: torch.Tensor) -> torch.Tensor:
        if self.feature_type == "edge":
            return sobel_edges(x)
        elif self.feature_type == "hog":
            return hog_like(x, bins=self.hog_bins)
        else:
            raise ValueError("feature_type must be 'edge' or 'hog'")

    @torch.no_grad()
    def _eval_epoch(self, loader) -> float:
        self.model.eval()
        loss_sum, n = 0.0, 0
        for x, _ in loader:
            x = x.to(self.device)
            tgt = self._make_target(x)
            y = self.model(x)
            loss = self.crit(y, tgt)
            loss_sum += loss.item() * x.size(0)
            n += x.size(0)
        return loss_sum / max(n, 1)

    def _maybe_plot(self):
        now = time.time()
        if now - self._last_plot_ts < 60:
            return
        self._last_plot_ts = now
        fig = plt.figure(figsize=(5,4))
        xs = list(range(len(self.hist["val_loss"])))
        ys = self.hist["val_loss"]
        plt.semilogy(xs, ys, marker="o", linewidth=1)
        plt.xlabel("log step")
        plt.ylabel("Val Feature Recon (log)")
        title = f"FeatureAE Val | type={self.feature_type} bins={self.hog_bins if self.feature_type=='hog' else 1}"
        plt.title(title)
        plt.grid(True, which="both", ls=":")
        out = os.path.join(self.results_dir, "FeatureAE_neuron_loss_plot.png")
        plt.tight_layout()
        fig.savefig(out)
        plt.close(fig)

    def fit(self, train_loader, val_loader):
        patience = self.es_patience
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            train_loss_sum, n = 0.0, 0
            for x, _ in train_loader:
                x = x.to(self.device)
                tgt = self._make_target(x)
                y = self.model(x)
                loss = self.crit(y, tgt)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.opt.step()
                train_loss_sum += loss.item() * x.size(0)
                n += x.size(0)
            train_loss = train_loss_sum / max(n, 1)

            val_loss = self._eval_epoch(val_loader)

            self.hist["train_loss"].append(train_loss)
            self.hist["val_loss"].append(val_loss)
            self.hist["neurons"].append(self.model.total_neurons())
            self.hist["epoch"].append(epoch)

            improved = val_loss < self.best_val - 1e-6
            if improved:
                self.best_val = val_loss
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience = self.es_patience
            else:
                patience -= 1

            self._maybe_plot()
            print(f"Epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f} | best {self.best_val:.6f}")
            if patience <= 0:
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        self._maybe_plot()
        return self.best_val

    @torch.no_grad()
    def evaluate(self, test_loader) -> float:
        self.model.eval()
        loss_sum, n = 0.0, 0
        for x, _ in test_loader:
            x = x.to(self.device)
            tgt = self._make_target(x)
            y = self.model(x)
            loss = self.crit(y, tgt)
            loss_sum += loss.item() * x.size(0)
            n += x.size(0)
        return loss_sum / max(n, 1)

    def save(self, path: str):
        torch.save({
            "model": self.model.state_dict(),
            "best_val": self.best_val,
            "hist": self.hist,
            "widths": self.model.widths_list(),
            "pooling_indices": list(self.model.pooling_indices),
            "out_ch": int(self.model.out_ch),
            "feature_type": self.feature_type,
            "hog_bins": int(self.hog_bins),
            "loss_type": self.loss_type,
        }, path)

    def load(self, path: str, map_location=None):
        blob = torch.load(path, map_location=map_location)
        self.model.load_state_dict(blob["model"])
        self.best_val = blob.get("best_val", float("inf"))
        self.hist = blob.get("hist", self.hist)
