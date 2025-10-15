import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Utility blocks ( CNN style)
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
# Blind-Spot (Noise2Void-style) Convolutional Autoencoder (single-model)
# - Predicts masked pixels from their context only
# - Loss is computed only on randomly masked pixels (per-pixel Bernoulli mask)
# - 0-based pooling indices during forward (mirrors  models)
# -----------------------------

class BlindSpotConvAE(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        widths: List[int] = [32, 64, 128],
        pooling_indices: List[int] = [1],
        center_kernel: int = 3,  # center area to blank out (odd number)
    ):
        super().__init__()
        assert len(widths) >= 1
        assert center_kernel % 2 == 1, "center_kernel must be odd"
        self.in_ch = in_ch
        self.widths = list(widths)
        self.pooling_indices = set(pooling_indices)
        self.center_kernel = center_kernel

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
        self.head = nn.Conv2d(ch, in_ch, kernel_size=1, stride=1, padding=0)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # A small conv to erase center pixel information before encoding (optional blind-spot prefilter)
        # We implement a depthwise conv initialized to be an identity minus a delta at center, then keep it fixed.
        k = center_kernel
        pad = k // 2
        self.prefilter = nn.Conv2d(in_ch, in_ch, kernel_size=k, padding=pad, groups=in_ch, bias=False)
        with torch.no_grad():
            w = torch.zeros(in_ch, 1, k, k)
            w[:, 0, :, :] = 0.0
            w[:, 0, pad, pad] = -1.0  # negative delta to cancel center
            self.prefilter.weight.copy_(w)
        for p in self.prefilter.parameters():
            p.requires_grad = False

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
        # Optional blind-spot prefilter to reduce leakage of center pixel into features
        x_pref = self.prefilter(x)
        h = x_pref
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
# Trainer: per-pixel blind-spot masking, loss on masked pixels only
# -----------------------------

import time
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class BlindSpotAETrainer:
    def __init__(
        self,
        model: BlindSpotConvAE,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        es_patience: int = 30,
        max_epochs: int = 300,
        results_dir: str = "results_ae_blindspot",
        mask_prob: float = 0.05,
        gaussian_std: float = 0.0,
    ):
        self.model = model.to(device)
        self.device = device
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.grad_clip = grad_clip
        self.es_patience = es_patience
        self.max_epochs = max_epochs
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        self.mask_prob = float(mask_prob)
        self.gaussian_std = float(gaussian_std)
        self.crit = nn.MSELoss(reduction="none")  # we'll reduce over masked pixels

        self.best_val = float("inf")
        self.best_state = None
        self.hist = {"train_loss": [], "val_loss": [], "neurons": [], "epoch": []}
        self._last_plot_ts = 0.0

    def _make_pixel_mask(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        with torch.no_grad():
            m = (torch.rand(B, 1, H, W, device=x.device) < self.mask_prob).float()
        return m

    def _corrupt(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # erase masked pixels and optionally add Gaussian noise elsewhere
        x_masked = x * (1.0 - mask)
        if self.gaussian_std > 0:
            noise = torch.randn_like(x) * self.gaussian_std
            x_masked = x_masked + noise * (1.0 - mask)  # don't add noise at masked pixels
        return x_masked

    @torch.no_grad()
    def _eval_epoch(self, loader) -> float:
        self.model.eval()
        loss_sum, n = 0.0, 0
        for x, _ in loader:
            x = x.to(self.device)
            mask = self._make_pixel_mask(x)
            x_in = self._corrupt(x, mask)
            y = self.model(x_in)
            diff = (y - x) ** 2  # per-pixel MSE
            masked_mse = (diff * mask).sum() / (mask.sum() + 1e-8)
            bs = x.size(0)
            loss_sum += masked_mse.item() * bs
            n += bs
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
        plt.ylabel("Val masked MSE (log)")
        plt.title(f"BlindSpotAE Val | neurons={self.model.total_neurons()} depth={self.model.depth()} widths={self.model.widths_list()} mp={self.mask_prob}")
        plt.grid(True, which="both", ls=":")
        out = os.path.join(self.results_dir, "BlindSpotAE_neuron_loss_plot.png")
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
                mask = self._make_pixel_mask(x)
                x_in = self._corrupt(x, mask)
                y = self.model(x_in)
                diff = (y - x) ** 2
                masked_mse = (diff * mask).sum() / (mask.sum() + 1e-8)

                self.opt.zero_grad(set_to_none=True)
                masked_mse.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.opt.step()

                bs = x.size(0)
                train_loss_sum += masked_mse.item() * bs
                n += bs

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
            mask = self._make_pixel_mask(x)
            x_in = self._corrupt(x, mask)
            y = self.model(x_in)
            diff = (y - x) ** 2
            masked_mse = (diff * mask).sum() / (mask.sum() + 1e-8)
            bs = x.size(0)
            loss_sum += masked_mse.item() * bs
            n += bs
        return loss_sum / max(n, 1)

    def save(self, path: str):
        torch.save({
            "model": self.model.state_dict(),
            "best_val": self.best_val,
            "hist": self.hist,
            "widths": self.model.widths_list(),
            "pooling_indices": list(self.model.pooling_indices),
            "mask_prob": float(self.mask_prob),
            "gaussian_std": float(self.gaussian_std),
        }, path)

    def load(self, path: str, map_location=None):
        blob = torch.load(path, map_location=map_location)
        self.model.load_state_dict(blob["model"])
        self.best_val = blob.get("best_val", float("inf"))
        self.hist = blob.get("hist", self.hist)
        self.mask_prob = float(blob.get("mask_prob", self.mask_prob))
        self.gaussian_std = float(blob.get("gaussian_std", self.gaussian_std))
