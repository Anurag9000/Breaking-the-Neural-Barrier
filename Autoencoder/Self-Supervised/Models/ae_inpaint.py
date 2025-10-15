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


# -----------------------------
# Context / Inpainting Convolutional Autoencoder (single-model)
# - Reconstructs contiguous missing regions (holes) from visible context
# - Loss computed only on hole pixels
# - 0-based pooling indices during forward (mirrors  models)
# -----------------------------

class InpaintingConvAE(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        widths: List[int] = [32, 64, 128],
        pooling_indices: List[int] = [0, 2],
    ):
        super().__init__()
        assert len(widths) >= 1
        self.in_ch = in_ch
        self.widths = list(widths)
        self.pooling_indices = set(pooling_indices)

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
# Trainer: hole (block) masking, loss on holes only
# -----------------------------

import time
import json
import os
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class InpaintingAETrainer:
    def __init__(
        self,
        model: InpaintingConvAE,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        es_patience: int = 30,
        max_epochs: int = 300,
        results_dir: str = "results_ae_inpaint",
        holes_per_image: int = 1,
        min_hole_frac: float = 0.15,
        max_hole_frac: float = 0.35,
        random_seed: int = 123,
    ):
        self.model = model.to(device)
        self.device = device
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.grad_clip = grad_clip
        self.es_patience = es_patience
        self.max_epochs = max_epochs
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        self.holes_per_image = int(holes_per_image)
        self.min_hole_frac = float(min_hole_frac)
        self.max_hole_frac = float(max_hole_frac)
        self.rng = random.Random(random_seed)
        self.crit = nn.MSELoss(reduction="none")

        self.best_val = float("inf")
        self.best_state = None
        self.hist = {"train_loss": [], "val_loss": [], "neurons": [], "epoch": []}
        self._last_plot_ts = 0.0

    def _rect_mask(self, B: int, H: int, W: int, device) -> torch.Tensor:
        mask = torch.zeros(B, 1, H, W, device=device)
        for b in range(B):
            for _ in range(self.holes_per_image):
                # sample rectangle area as fraction of image area
                area = H * W
                target_area = self.rng.uniform(self.min_hole_frac, self.max_hole_frac) * area
                aspect = self.rng.uniform(0.5, 2.0)
                h = int(round((target_area * aspect) ** 0.5))
                w = int(round((target_area / aspect) ** 0.5))
                h = max(1, min(H, h))
                w = max(1, min(W, w))
                # sample top-left
                if H - h > 0: y0 = self.rng.randint(0, H - h)
                else: y0 = 0
                if W - w > 0: x0 = self.rng.randint(0, W - w)
                else: x0 = 0
                mask[b, 0, y0:y0+h, x0:x0+w] = 1.0
        return mask

    @torch.no_grad()
    def _eval_epoch(self, loader) -> float:
        self.model.eval()
        loss_sum, n = 0.0, 0
        for x, _ in loader:
            x = x.to(self.device)
            B, C, H, W = x.shape
            mask = self._rect_mask(B, H, W, x.device)
            x_in = x * (1.0 - mask)
            y = self.model(x_in)
            diff = (y - x) ** 2
            hole_mse = (diff * mask).sum() / (mask.sum() + 1e-8)
            loss_sum += hole_mse.item() * B
            n += B
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
        plt.ylabel("Val hole MSE (log)")
        plt.title(f"InpaintAE Val | neurons={self.model.total_neurons()} depth={self.model.depth()} widths={self.model.widths_list()}")
        plt.grid(True, which="both", ls=":")
        out = os.path.join(self.results_dir, "InpaintAE_neuron_loss_plot.png")
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
                B, C, H, W = x.shape
                mask = self._rect_mask(B, H, W, x.device)
                x_in = x * (1.0 - mask)
                y = self.model(x_in)
                diff = (y - x) ** 2
                hole_mse = (diff * mask).sum() / (mask.sum() + 1e-8)

                self.opt.zero_grad(set_to_none=True)
                hole_mse.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.opt.step()

                train_loss_sum += hole_mse.item() * B
                n += B

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
            B, C, H, W = x.shape
            mask = self._rect_mask(B, H, W, x.device)
            x_in = x * (1.0 - mask)
            y = self.model(x_in)
            diff = (y - x) ** 2
            hole_mse = (diff * mask).sum() / (mask.sum() + 1e-8)
            loss_sum += hole_mse.item() * B
            n += B
        return loss_sum / max(n, 1)

    def save(self, path: str):
        torch.save({
            "model": self.model.state_dict(),
            "best_val": self.best_val,
            "hist": self.hist,
            "widths": self.model.widths_list(),
            "pooling_indices": list(self.model.pooling_indices),
        }, path)

    def load(self, path: str, map_location=None):
        blob = torch.load(path, map_location=map_location)
        self.model.load_state_dict(blob["model"])
        self.best_val = blob.get("best_val", float("inf"))
        self.hist = blob.get("hist", self.hist)
