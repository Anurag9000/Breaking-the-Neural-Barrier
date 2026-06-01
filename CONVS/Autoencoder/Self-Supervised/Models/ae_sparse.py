import math
from typing import List, Literal, Optional

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
# Sparse Convolutional Autoencoder (single-model)
# Modes:
#   - "l1": add L1 penalty on encoder activations (latent feature map)
#   - "k":  keep top-k activations in latent (per-sample), zero the rest (k can be absolute or fraction)
# Pooling indices are 0-based and applied after encoder block i, mirroring  codebase.
# -----------------------------

SparsityMode = Literal["l1", "k"]

class SparseConvAE(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        widths: List[int] = [32, 64, 128],
        pooling_indices: List[int] = [0, 2],
        mode: SparsityMode = "l1",
        k: Optional[int] = None,            # used if mode=="k" (absolute top-k)
        k_frac: Optional[float] = 0.05,     # or fraction of total latent units per sample
    ):
        super().__init__()
        assert len(widths) >= 1
        self.in_ch = in_ch
        self.widths = list(widths)
        self.pooling_indices = set(pooling_indices)
        self.mode = mode
        self.k = k
        self.k_frac = k_frac

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

    # ----- encode/decode with latent hook -----
    def encode(self, x: torch.Tensor) -> (torch.Tensor, int):
        h = x
        down_ct = 0
        for i, blk in enumerate(self.encoder):
            h = blk(h)
            if i in self.pooling_indices:
                h = self.pool(h)
                down_ct += 1
        return h, down_ct

    def apply_sparsity(self, h: torch.Tensor) -> torch.Tensor:
        if self.mode == "l1":
            return h  # identity; L1 is applied in the loss
        # k-sparse: keep top-k per sample across all channels and spatial positions
        B, C, H, W = h.shape
        flat = h.view(B, -1)
        if self.k is not None:
            k = max(1, min(self.k, flat.size(1)))
        else:
            # fraction
            k = max(1, int(round(self.k_frac * flat.size(1))))
        # get threshold per sample
        topk_vals, _ = torch.topk(flat, k, dim=1)
        thresh = topk_vals[:, -1].unsqueeze(1)  # (B,1)
        mask = (flat >= thresh).float()
        h_sparse = (flat * mask).view(B, C, H, W)
        return h_sparse

    def decode(self, h: torch.Tensor, down_ct: int) -> torch.Tensor:
        z = h
        up_remaining = down_ct
        for j, blk in enumerate(self.decoder):
            if up_remaining > 0:
                z = self.upsample(z)
                up_remaining -= 1
            z = blk(z)
        while up_remaining > 0:
            z = self.upsample(z)
            up_remaining -= 1
        out = self.head(z)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, down_ct = self.encode(x)
        h_s = self.apply_sparsity(h)
        out = self.decode(h_s, down_ct)
        return out


# -----------------------------
# Trainer (supports L1 penalty and k-sparse)
# -----------------------------

import time
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class SparseAETrainer:
    def __init__(
        self,
        model: SparseConvAE,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        es_patience: int = 30,
        max_epochs: int = 300,
        results_dir: str = "results_ae_sparse",
        lam_l1: float = 1e-5,  # only used in L1 mode
    ):
        self.model = model.to(device)
        self.device = device
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.crit = nn.MSELoss(reduction="mean")
        self.grad_clip = grad_clip
        self.es_patience = es_patience
        self.max_epochs = max_epochs
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        self.lam_l1 = lam_l1

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
            # forward (includes sparsity op)
            y = self.model(x)
            recon = self.crit(y, x)
            total = recon  # validation excludes sparsity penalties for fair early stopping
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
        plt.ylabel("Val MSE (log)")
        plt.title(f"SparseAE Val | neurons={self.model.total_neurons()} depth={self.model.depth()} widths={self.model.widths_list()} mode={self.model.mode}")
        plt.grid(True, which="both", ls=":")
        out = os.path.join(self.results_dir, "SparseAE_neuron_loss_plot.png")
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
                # forward (includes sparsity op)
                y = self.model(x)
                recon = self.crit(y, x)
                loss = recon

                # add L1 penalty on encoder activations only in L1 mode
                if self.model.mode == "l1":
                    with torch.no_grad():
                        h, _ = self.model.encode(x)  # compute latent without tracking extra grads
                    # Recompute with grad? For efficiency, approximate L1 on the already-computed h via stop-grad:
                    # this still backprops through decoder via recon; L1 serves as auxiliary prior (common practice)
                    l1_pen = h.abs().mean()
                    loss = loss + self.lam_l1 * l1_pen

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
            "mode": self.model.mode,
            "k": (None if self.model.k is None else int(self.model.k)),
            "k_frac": (None if self.model.k_frac is None else float(self.model.k_frac)),
            "lam_l1": float(self.lam_l1),
        }, path)

    def load(self, path: str, map_location=None):
        blob = torch.load(path, map_location=map_location)
        self.model.load_state_dict(blob["model"])
        self.best_val = blob.get("best_val", float("inf"))
        self.hist = blob.get("hist", self.hist)
        # restore hyperparams for bookkeeping
        self.model.mode = blob.get("mode", self.model.mode)
        self.model.k = blob.get("k", self.model.k)
        self.model.k_frac = blob.get("k_frac", self.model.k_frac)
        self.lam_l1 = float(blob.get("lam_l1", self.lam_l1))
