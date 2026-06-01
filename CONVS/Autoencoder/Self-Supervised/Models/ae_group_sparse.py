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
# Group-Sparse Convolutional Autoencoder (single-model)
# - Encourages *block sparsity* in the latent by applying a Group-Lasso penalty
#   across contiguous channel groups at each spatial position.
# - For latent h in R^{B x C x H x W}, channels are partitioned into groups of size G.
#   Group norm at (b, g, y, x) = sqrt(sum_{c in group g} h[b,c,y,x]^2).
#   Loss adds lam * mean(group_norms) to the reconstruction objective.
# - 0-based pooling indices during forward (mirrors  models)
# -----------------------------

class GroupSparseConvAE(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        widths: List[int] = [32, 64, 128],
        pooling_indices: List[int] = [0, 2],
        group_size: int = 8,
    ):
        super().__init__()
        assert len(widths) >= 1
        self.in_ch = in_ch
        self.widths = list(widths)
        self.pooling_indices = set(pooling_indices)
        self.group_size = int(group_size)

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

    # ----- encode/decode helpers -----
    def encode(self, x: torch.Tensor):
        h = x
        down_ct = 0
        for i, blk in enumerate(self.encoder):
            h = blk(h)
            if i in self.pooling_indices:
                h = self.pool(h)
                down_ct += 1
        return h, down_ct

    def decode(self, h: torch.Tensor, down_ct: int) -> torch.Tensor:
        z = h
        ups = down_ct
        for j, blk in enumerate(self.decoder):
            if ups > 0:
                z = self.upsample(z)
                ups -= 1
            z = blk(z)
        while ups > 0:
            z = self.upsample(z)
            ups -= 1
        out = self.head(z)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, down_ct = self.encode(x)
        out = self.decode(h, down_ct)
        return out

    # Group-lasso penalty on latent tensor h (computed outside forward in trainer)
    def group_lasso(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B,C,H,W). Partition channels into contiguous groups of size G.
        Returns mean group norm across (B, groups, H, W).
        """
        B, C, H, W = h.shape
        G = max(1, self.group_size)
        if C % G != 0:
            # pad channels with zeros to be divisible by group size (no trainable params added)
            pad_c = G - (C % G)
            h = F.pad(h, (0,0,0,0,0,pad_c))
            C = C + pad_c
        h = h.view(B, C // G, G, H, W)
        # L2 over group axis, then mean over everything
        grp_norm = torch.sqrt(torch.clamp((h ** 2).sum(dim=2), min=1e-12))  # (B, C//G, H, W)
        return grp_norm.mean()


# -----------------------------
# Trainer with Group-Lasso penalty
# -----------------------------

import time
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class GroupSparseAETrainer:
    def __init__(
        self,
        model: GroupSparseConvAE,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        es_patience: int = 30,
        max_epochs: int = 300,
        results_dir: str = "results_ae_group_sparse",
        loss_type: str = "mse",
        lam_group: float = 1e-4,
    ):
        self.model = model.to(device)
        self.device = device
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        if loss_type == "mse":
            self.crit = nn.MSELoss(reduction="mean")
        elif loss_type == "l1":
            self.crit = nn.L1Loss(reduction="mean")
        else:
            raise ValueError("loss_type must be 'mse' or 'l1'")
        self.loss_type = loss_type
        self.grad_clip = grad_clip
        self.es_patience = es_patience
        self.max_epochs = max_epochs
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        self.lam_group = float(lam_group)

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
            recon = self.crit(y, x)
            # exclude group penalty on validation for early stopping fairness
            total = recon
            loss_sum += total.item() * x.size(0)
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
        plt.title(f"GroupSparseAE Val | neurons={self.model.total_neurons()} depth={self.model.depth()} widths={self.model.widths_list()} G={self.model.group_size} lam={self.lam_group}")
        plt.grid(True, which="both", ls=":")
        out = os.path.join(self.results_dir, "GroupSparseAE_neuron_loss_plot.png")
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
                # forward
                y = self.model(x)
                recon = self.crit(y, x)
                # group lasso on latent (stop-grad ok since it serves as regularizer)
                with torch.no_grad():
                    h, _ = self.model.encode(x)
                grp = self.model.group_lasso(h)
                loss = recon + self.lam_group * grp

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
            recon = self.crit(y, x)
            loss_sum += recon.item() * x.size(0)
            n += x.size(0)
        return loss_sum / max(n, 1)

    def save(self, path: str):
        torch.save({
            "model": self.model.state_dict(),
            "best_val": self.best_val,
            "hist": self.hist,
            "widths": self.model.widths_list(),
            "pooling_indices": list(self.model.pooling_indices),
            "group_size": int(self.model.group_size),
            "lam_group": float(self.lam_group),
            "loss_type": self.loss_type,
        }, path)

    def load(self, path: str, map_location=None):
        blob = torch.load(path, map_location=map_location)
        self.model.load_state_dict(blob["model"])
        self.best_val = blob.get("best_val", float("inf"))
        self.hist = blob.get("hist", self.hist)
        self.model.group_size = int(blob.get("group_size", self.model.group_size))
        self.lam_group = float(blob.get("lam_group", self.lam_group))
        self.loss_type = blob.get("loss_type", self.loss_type)
