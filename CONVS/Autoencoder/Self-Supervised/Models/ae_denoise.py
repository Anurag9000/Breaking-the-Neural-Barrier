import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Utility blocks (mirrors  CNN style)
# -----------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ConvTBNReLU(nn.Module):
    """Transpose-conv block for decoder (optionally we can use Upsample+Conv for artifact-free upscaling)."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, output_padding: int = 0, bias: bool = True):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, output_padding=output_padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.deconv(x)))


# -----------------------------
# Denoising Convolutional Autoencoder (single-model, self-supervised)
# - 0-based pooling indices during forward, mirroring  models
# - Symmetric decoder; tracks where downsamples happened and upsamples accordingly
# -----------------------------

class DenoisingConvAE(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        widths: List[int] = [16, 32, 64],
        pooling_indices: List[int] = [0, 2],
        use_transpose_conv: bool = False,
    ):
        super().__init__()
        assert len(widths) >= 1, "Need at least one encoder block"
        self.in_ch = in_ch
        self.widths = list(widths)
        self.pooling_indices = set(pooling_indices)  # 0-based indices at which we downsample after the block
        self.use_transpose_conv = use_transpose_conv

        # Encoder
        enc = []
        ch = in_ch
        for i, w in enumerate(widths):
            enc.append(ConvBNReLU(ch, w))
            ch = w
        self.encoder = nn.ModuleList(enc)

        # Decoder (mirror the widths in reverse order)
        dec = []
        rev_widths = list(reversed(widths))
        ch = rev_widths[0]
        for j, w in enumerate(rev_widths[1:]):  # for hidden decoder blocks (not the final projection)
            dec.append(ConvBNReLU(ch, w))
            ch = w
        self.decoder = nn.ModuleList(dec)

        # Final reconstruction head to input channels
        self.head = nn.Conv2d(ch, in_ch, kernel_size=1, stride=1, padding=0)

        # Down/Up samplers
        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    # -------- capacity metric (for parity with  plots) --------
    def total_neurons(self) -> int:
        # Sum of encoder out_channels + sum of decoder out_channels (excluding the final 1x1 head)
        enc_neurons = sum([m.conv.out_channels for m in self.encoder])
        dec_neurons = sum([m.conv.out_channels for m in self.decoder])
        return enc_neurons + dec_neurons

    def depth(self) -> int:
        return len(self.widths)

    def widths_list(self) -> List[int]:
        return list(self.widths)

    # -------- forward with 0-based pooling locations --------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        downsamples = []  # record at which blocks we downsample to invert later
        h = x
        for i, block in enumerate(self.encoder):
            h = block(h)
            if i in self.pooling_indices:
                h = self.pool(h)
                downsamples.append(i)

        # Decoder path (mirror spatial sizes by upsampling in reverse order of occurrence)
        # We will upsample the same number of times as we downsampled, but distributed across decoder blocks.
        # A simple and robust approach: upsample *before* each decoder block if there are pending upsamples.
        pending_ups = len(downsamples)
        for j, block in enumerate(self.decoder):
            if pending_ups > 0:
                h = self.upsample(h)
                pending_ups -= 1
            h = block(h)

        # If any remaining ups (shouldn't happen if decoder mirrors encoder), apply them
        while pending_ups > 0:
            h = self.upsample(h)
            pending_ups -= 1

        # Final reconstruction
        out = self.head(h)
        return out


# -----------------------------
# Trainer with early stopping, logging, and semilog plots (val loss vs neurons)
# -----------------------------

import time
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class DAETrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        es_patience: int = 20,
        max_epochs: int = 200,
        results_dir: str = "results_ae",
        corruption_std: float = 0.1,
        corruption_mask_prob: float = 0.0,
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

        self.corruption_std = corruption_std
        self.corruption_mask_prob = corruption_mask_prob

        self.best_val = float("inf")
        self.best_state = None
        self.hist = {"train_loss": [], "val_loss": [], "neurons": [], "epoch": []}
        self._last_plot_ts = 0.0

    def _corrupt(self, x: torch.Tensor) -> torch.Tensor:
        # Gaussian noise + optional random masking (drop pixels)
        if self.corruption_std > 0:
            noise = torch.randn_like(x) * self.corruption_std
            x = x + noise
        if self.corruption_mask_prob > 0:
            mask = (torch.rand_like(x[:, :1, :, :]) < self.corruption_mask_prob).float()
            x = x * (1.0 - mask)
        return x.clamp(-1.0, 1.0)  # assuming normalized inputs in [-1, 1]

    @torch.no_grad()
    def _eval_epoch(self, loader) -> float:
        self.model.eval()
        loss_sum, n = 0.0, 0
        for x, _ in loader:
            x = x.to(self.device)
            noisy = self._corrupt(x.clone())
            y = self.model(noisy)
            loss = self.crit(y, x)
            bs = x.size(0)
            loss_sum += loss.item() * bs
            n += bs
        return loss_sum / max(n, 1)

    def _maybe_plot(self):
        now = time.time()
        if now - self._last_plot_ts < 60:
            return
        self._last_plot_ts = now
        # semilog plot: best val loss vs neurons (single point per log entry)
        fig = plt.figure(figsize=(5,4))
        xs = list(range(len(self.hist["val_loss"])))
        ys = self.hist["val_loss"]
        plt.semilogy(xs, ys, marker="o", linewidth=1)
        plt.xlabel("log step")
        plt.ylabel("Val MSE (log)")
        plt.title(f"DAE Val Loss vs steps | neurons={self.model.total_neurons()} depth={self.model.depth()} widths={self.model.widths_list()}")
        plt.grid(True, which="both", ls=":")
        out = os.path.join(self.results_dir, "DAE_neuron_loss_plot.png")
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
                noisy = self._corrupt(x.clone())
                y = self.model(noisy)
                loss = self.crit(y, x)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.opt.step()
                bs = x.size(0)
                train_loss_sum += loss.item() * bs
                n += bs
            train_loss = train_loss_sum / max(n, 1)

            val_loss = self._eval_epoch(val_loader)

            # history
            self.hist["train_loss"].append(train_loss)
            self.hist["val_loss"].append(val_loss)
            self.hist["neurons"].append(self.model.total_neurons())
            self.hist["epoch"].append(epoch)

            # early stopping
            improved = val_loss < self.best_val - 1e-6
            if improved:
                self.best_val = val_loss
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience = self.es_patience
            else:
                patience -= 1

            # periodic plot
            self._maybe_plot()

            # print minimal log (runner handles file logging)
            print(f"Epoch {epoch:03d} | train {train_loss:.5f} | val {val_loss:.5f} | best {self.best_val:.5f}")

            if patience <= 0:
                break

        # restore best
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        # final plot
        self._maybe_plot()
        return self.best_val

    @torch.no_grad()
    def evaluate(self, test_loader) -> float:
        self.model.eval()
        loss_sum, n = 0.0, 0
        for x, _ in test_loader:
            x = x.to(self.device)
            noisy = self._corrupt(x.clone())
            y = self.model(noisy)
            loss = self.crit(y, x)
            bs = x.size(0)
            loss_sum += loss.item() * bs
            n += bs
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
