import math
from typing import List, Tuple

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


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )
    def forward(self, x):
        return self.block(x)


# -----------------------------
# UNet-style Convolutional Autoencoder (single-model)
# - Symmetric encoder/decoder with skip connections
# - 0-based pooling indices applied during forward (mirrors  models)
# -----------------------------

class UNetConvAE(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        widths: List[int] = [32, 64, 128, 128],
        pooling_indices: List[int] = [0, 1],  # downsample after these encoder stages
        use_bilinear: bool = True,            # upsampling mode
    ):
        super().__init__()
        assert len(widths) >= 2, "Need at least two stages for UNet"
        self.in_ch = in_ch
        self.widths = list(widths)
        self.pooling_indices = set(pooling_indices)
        self.use_bilinear = use_bilinear

        # Encoder stages (DoubleConv)
        enc = []
        ch = in_ch
        for w in widths:
            enc.append(DoubleConv(ch, w))
            ch = w
        self.encoder = nn.ModuleList(enc)

        # Decoder stages: mirror except first which is bottleneck output
        rev = list(reversed(widths))
        dec_blocks = []
        up_blocks = []
        ch = rev[0]
        for w in rev[1:]:
            up_blocks.append(nn.Upsample(scale_factor=2, mode="bilinear" if use_bilinear else "nearest", align_corners=False if use_bilinear else None))
            dec_blocks.append(DoubleConv(ch + w, w))  # concat skip features
            ch = w
        self.up = nn.ModuleList(up_blocks)
        self.decoder = nn.ModuleList(dec_blocks)

        self.head = nn.Conv2d(ch, in_ch, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(2)

    # capacity stats
    def total_neurons(self) -> int:
        # count channels across encoder + decoder convs (approximate capacity proxy)
        enc_neurons = sum(m.block[0].conv.out_channels + m.block[1].conv.out_channels for m in self.encoder)
        dec_neurons = sum(m.block[0].conv.out_channels + m.block[1].conv.out_channels for m in self.decoder)
        return enc_neurons + dec_neurons

    def depth(self) -> int:
        return len(self.widths)

    def widths_list(self) -> List[int]:
        return list(self.widths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder with skip collection
        skips = []
        h = x
        for i, blk in enumerate(self.encoder):
            h = blk(h)
            skips.append(h)
            if i in self.pooling_indices:
                h = self.pool(h)
        # Decoder with symmetric upsampling and skip concatenation
        z = h
        # consume skips in reverse, skipping the deepest one already used as z
        skip_iter = list(reversed(skips[:-1]))
        for up, dec, skip in zip(self.up, self.decoder, skip_iter):
            z = up(z)
            # ensure spatial match (due to odd sizes, pad/crop as needed)
            if z.shape[-2:] != skip.shape[-2:]:
                # simple center-crop or pad to match skip
                diff_y = skip.size(-2) - z.size(-2)
                diff_x = skip.size(-1) - z.size(-1)
                z = F.pad(z, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
            z = torch.cat([z, skip], dim=1)
            z = dec(z)
        out = self.head(z)
        return out


# -----------------------------
# Trainer (reconstruction with MSE/L1), early stopping, plotting
# -----------------------------

import time
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class UNetAETrainer:
    def __init__(
        self,
        model: UNetConvAE,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        es_patience: int = 30,
        max_epochs: int = 300,
        results_dir: str = "results_ae_unet",
        loss_type: str = "mse",  # "mse" or "l1"
    ):
        self.model = model.to(device)
        self.device = device
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.grad_clip = grad_clip
        self.es_patience = es_patience
        self.max_epochs = max_epochs
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        self.loss_type = loss_type
        if loss_type == "mse":
            self.crit = nn.MSELoss(reduction="mean")
        elif loss_type == "l1":
            self.crit = nn.L1Loss(reduction="mean")
        else:
            raise ValueError("loss_type must be 'mse' or 'l1'")

        self.best_val = float("inf")
        self.best_state = None
        self.hist = {"train_loss": [], "val_loss": [], "neurons": [], "epoch": []}
        self._last_plot_ts = 0.0

    @torch.no_grad()
    def _eval_epoch(self, loader) -> float:
        self.model.eval()
        loss_sum, n = 0.0, 0
        for x, _ in loader:
            x = x.to(self.device)
            y = self.model(x)
            loss = self.crit(y, x)
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
        plt.ylabel("Val Recon (log)")
        plt.title(f"UNetAE Val | neurons={self.model.total_neurons()} depth={self.model.depth()} widths={self.model.widths_list()} loss={self.loss_type}")
        plt.grid(True, which="both", ls=":")
        out = os.path.join(self.results_dir, "UNetAE_neuron_loss_plot.png")
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
                y = self.model(x)
                loss = self.crit(y, x)

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
            y = self.model(x)
            loss = self.crit(y, x)
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
            "loss_type": self.loss_type,
        }, path)

    def load(self, path: str, map_location=None):
        blob = torch.load(path, map_location=map_location)
        self.model.load_state_dict(blob["model"])
        self.best_val = blob.get("best_val", float("inf"))
        self.hist = blob.get("hist", self.hist)
        self.loss_type = blob.get("loss_type", self.loss_type)
