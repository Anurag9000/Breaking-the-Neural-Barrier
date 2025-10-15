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
# Energy-Based Autoencoder (single-model)
# - Architecture: plain conv encoder/decoder with bottleneck (no EMA/teacher/ensemble)
# - Energy function: E_θ(x) = SmoothL1(x - f_θ(x)) aggregated over pixels
# - Training objective (hinge):  L = E_pos + λ * max(0, m - E_neg)
#       where x_neg is a *single-model* corruption of x (no extra networks).
# - Early stopping measured on clean validation energy only (E_pos).
# -----------------------------
class EnergyConvAE(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        widths: List[int] = [32, 64, 128],
        pooling_indices: List[int] = [0, 2],
        huber_beta: float = 0.1,
    ):
        super().__init__()
        assert len(widths) >= 1
        self.in_ch = in_ch
        self.widths = list(widths)
        self.pooling_indices = set(pooling_indices)
        self.huber_beta = float(huber_beta)

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
        self._crit = nn.SmoothL1Loss(beta=self.huber_beta, reduction="none")

    # capacity stats
    def total_neurons(self) -> int:
        enc_neurons = sum(m.conv.out_channels for m in self.encoder)
        dec_neurons = sum(m.conv.out_channels for m in self.decoder)
        return enc_neurons + dec_neurons

    def depth(self) -> int:
        return len(self.widths)

    def widths_list(self) -> List[int]:
        return list(self.widths)

    # ----- forward / energy -----
    def _encode(self, x: torch.Tensor):
        h = x
        downs = 0
        for i, blk in enumerate(self.encoder):
            h = blk(h)
            if i in self.pooling_indices:
                h = self.pool(h)
                downs += 1
        return h, downs

    def _decode(self, h: torch.Tensor, downs: int):
        z = h
        ups = downs
        for blk in self.decoder:
            if ups > 0:
                z = self.upsample(z)
                ups -= 1
            z = blk(z)
        while ups > 0:
            z = self.upsample(z)
            ups -= 1
        out = self.head(z)
        return out

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        h, downs = self._encode(x)
        return self._decode(h, downs)

    def energy_map(self, x: torch.Tensor) -> torch.Tensor:
        y = self.reconstruct(x)
        # SmoothL1 per-pixel per-channel, then mean over (C,H,W)
        per = self._crit(y, x)
        return per

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        per = self.energy_map(x)
        return per.mean(dim=(1,2,3))  # (B,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return reconstruction for compatibility
        return self.reconstruct(x)

# -----------------------------
# Corruption utilities for negatives (single-model, no extra nets)
# -----------------------------

def make_negative(x: torch.Tensor, mode: str = "batch_permute", strength: float = 0.5, cutout_frac: float = 0.4) -> torch.Tensor:
    """
    x: (B,C,H,W)
    mode:
      - batch_permute: in-batch roll to mismatched targets
      - gaussian: add N(0, strength) noise (clipped)
      - cutout: zero a random square of area ~ cutout_frac
    """
    B, C, H, W = x.shape
    if mode == "batch_permute":
        shift = 1
        return x.roll(shifts=shift, dims=0)
    elif mode == "gaussian":
        noise = torch.randn_like(x) * strength
        return torch.clamp(x + noise, -3.0, 3.0)  # assume normalized inputs
    elif mode == "cutout":
        out = x.clone()
        side = max(1, int((H * W * cutout_frac) ** 0.5))
        cy = torch.randint(0, H, (B,), device=x.device)
        cx = torch.randint(0, W, (B,), device=x.device)
        y0 = torch.clamp(cy - side // 2, 0, H-1)
        x0 = torch.clamp(cx - side // 2, 0, W-1)
        for i in range(B):
            y1 = min(H, y0[i] + side)
            x1 = min(W, x0[i] + side)
            out[i, :, y0[i]:y1, x0[i]:x1] = 0.0
        return out
    else:
        raise ValueError("Unknown corruption mode: " + str(mode))

# -----------------------------
# Trainer (hinge energy), early stopping, plotting
# -----------------------------
import time
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class EnergyAETrainer:
    def __init__(
        self,
        model: EnergyConvAE,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        es_patience: int = 30,
        max_epochs: int = 300,
        results_dir: str = "results_ae_energy",
        margin: float = 0.5,
        lambda_neg: float = 1.0,
        neg_mode: str = "batch_permute",
        neg_strength: float = 0.5,
        cutout_frac: float = 0.4,
    ):
        self.model = model.to(device)
        self.device = device
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.grad_clip = grad_clip
        self.es_patience = es_patience
        self.max_epochs = max_epochs
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        self.margin = float(margin)
        self.lambda_neg = float(lambda_neg)
        self.neg_mode = neg_mode
        self.neg_strength = float(neg_strength)
        self.cutout_frac = float(cutout_frac)

        self.best_val = float("inf")
        self.best_state = None
        self.hist = {"train_pos": [], "train_neg": [], "val_pos": [], "epoch": []}
        self._last_plot_ts = 0.0

    @torch.no_grad()
    def _eval_epoch(self, loader) -> float:
        self.model.eval()
        # validation considers *positive* energy only
        e_sum, n = 0.0, 0
        for x, _ in loader:
            x = x.to(self.device)
            e_pos = self.model.energy(x)  # (B,)
            e_sum += e_pos.sum().item()
            n += x.size(0)
        return e_sum / max(n, 1)

    def _maybe_plot(self):
        now = time.time()
        if now - self._last_plot_ts < 60:
            return
        self._last_plot_ts = now
        fig = plt.figure(figsize=(5,4))
        xs = list(range(len(self.hist["val_pos"])))
        plt.semilogy(xs, self.hist["val_pos"], marker="o", linewidth=1)
        plt.xlabel("epoch")
        plt.ylabel("Val E_pos (log)")
        plt.title(f"EnergyAE | m={self.margin} λ={self.lambda_neg} mode={self.neg_mode}")
        plt.grid(True, which="both", ls=":")
        out = os.path.join(self.results_dir, "EnergyAE_val_energy.png")
        plt.tight_layout()
        fig.savefig(out)
        plt.close(fig)

    def fit(self, train_loader, val_loader):
        patience = self.es_patience
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            pos_sum, neg_sum, n = 0.0, 0.0, 0
            for x, _ in train_loader:
                x = x.to(self.device)
                x_neg = make_negative(x, mode=self.neg_mode, strength=self.neg_strength, cutout_frac=self.cutout_frac)

                e_pos = self.model.energy(x)      # (B,)
                e_neg = self.model.energy(x_neg)  # (B,)
                hinge = F.relu(self.margin - e_neg)
                loss = e_pos.mean() + self.lambda_neg * hinge.mean()

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.opt.step()

                pos_sum += e_pos.sum().item()
                neg_sum += e_neg.sum().item()
                n += x.size(0)

            train_pos = pos_sum / max(n, 1)
            train_neg = neg_sum / max(n, 1)
            val_pos = self._eval_epoch(val_loader)

            self.hist["train_pos"].append(train_pos)
            self.hist["train_neg"].append(train_neg)
            self.hist["val_pos"].append(val_pos)
            self.hist["epoch"].append(epoch)

            improved = val_pos < (self.best_val - 1e-6)
            if improved:
                self.best_val = val_pos
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience = self.es_patience
            else:
                patience -= 1

            self._maybe_plot()
            print(f"Epoch {epoch:03d} | E_pos {train_pos:.6f} | E_neg {train_neg:.6f} | Val E_pos {val_pos:.6f} | best {self.best_val:.6f}")
            if patience <= 0:
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        self._maybe_plot()
        return self.best_val

    @torch.no_grad()
    def evaluate(self, test_loader) -> float:
        # Positive energy on test set
        self.model.eval()
        e_sum, n = 0.0, 0
        for x, _ in test_loader:
            x = x.to(self.device)
            e = self.model.energy(x)
            e_sum += e.sum().item()
            n += x.size(0)
        return e_sum / max(n, 1)

    def save(self, path: str):
        torch.save({
            "model": self.model.state_dict(),
            "best_val": self.best_val,
            "hist": self.hist,
            "widths": self.model.widths_list(),
            "pooling_indices": list(self.model.pooling_indices),
            "huber_beta": float(self.model.huber_beta),
            "margin": self.margin,
            "lambda_neg": self.lambda_neg,
            "neg_mode": self.neg_mode,
            "neg_strength": self.neg_strength,
            "cutout_frac": self.cutout_frac,
        }, path)

    def load(self, path: str, map_location=None):
        blob = torch.load(path, map_location=map_location)
        self.model.load_state_dict(blob["model"])
        self.best_val = blob.get("best_val", float("inf"))
        self.hist = blob.get("hist", self.hist)
