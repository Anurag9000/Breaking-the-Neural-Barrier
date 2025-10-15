import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Utility block (encoder side)
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
# Tied-Weights Convolutional Autoencoder (single-model)
# - Decoder uses *tied* transposed-convolutional weights from the encoder convs
# - We implement tying functionally via F.conv_transpose2d, so the decoder holds no conv weights
# - 0-based pooling indices during forward (mirrors  models)
# -----------------------------

class TiedConvAE(nn.Module):
    def __init__(
        self,
        in_ch: int = 3,
        widths: List[int] = [32, 64, 128],
        pooling_indices: List[int] = [0, 2],
        use_decoder_bn: bool = True,
    ):
        super().__init__()
        assert len(widths) >= 1
        self.in_ch = in_ch
        self.widths = list(widths)
        self.pooling_indices = set(pooling_indices)
        self.use_decoder_bn = use_decoder_bn

        # Encoder blocks
        enc = []
        ch = in_ch
        for w in widths:
            enc.append(ConvBNReLU(ch, w))
            ch = w
        self.encoder = nn.ModuleList(enc)

        # Decoder BN/Act after each tied transposed-conv (weights come from encoder)
        # For layer i in decoder (mirrors encoder layer j in reverse), the out_ch == encoder[j-1].conv.in_channels
        dec_bn = []
        dec_act = []
        rev = list(reversed(widths))
        ch = rev[0]
        for w in rev[1:]:
            # tied deconv would map (in_ch=ch) -> (out_ch=w) because enc had Conv(in=w, out=ch) at that sym position
            if use_decoder_bn:
                dec_bn.append(nn.BatchNorm2d(w))
            else:
                dec_bn.append(nn.Identity())
            dec_act.append(nn.ReLU(inplace=True))
            ch = w
        self.dec_bn = nn.ModuleList(dec_bn)
        self.dec_act = nn.ModuleList(dec_act)

        # Final 1x1 conv head (learned) to reconstruct input channels
        self.head = nn.Conv2d(ch, in_ch, kernel_size=1, stride=1, padding=0)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    # capacity stats (count encoder channels only + BN on decoder)
    def total_neurons(self) -> int:
        enc_neurons = sum(m.conv.out_channels for m in self.encoder)
        # decoder carries only BN params; for a rough proxy, we include encoder only
        return enc_neurons

    def depth(self) -> int:
        return len(self.widths)

    def widths_list(self) -> List[int]:
        return list(self.widths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ---------- Encode ----------
        feats = []  # store for shape info; also needed to fetch tied weights
        h = x
        down_ct = 0
        for i, blk in enumerate(self.encoder):
            h = blk(h)
            feats.append(h)
            if i in self.pooling_indices:
                h = self.pool(h)
                down_ct += 1

        # ---------- Decode (tied weights) ----------
        # Walk encoder layers in reverse; for encoder layer L with weight W[L] (shape [C_out, C_in, k, k]),
        # the corresponding decoder op is conv_transpose2d with weight of shape [C_in, C_out, k, k].
        z = h
        enc_layers = list(self.encoder)
        dec_bn = list(self.dec_bn)
        dec_act = list(self.dec_act)
        bn_idx = 0
        for enc_blk in reversed(enc_layers[1:]):  # mirror all but the very first enc layer; last deconv handled by head input ch
            if down_ct > 0:
                z = self.upsample(z)
                down_ct -= 1
            W = enc_blk.conv.weight  # [C_out, C_in, k, k]
            z = F.conv_transpose2d(z, W, bias=None, stride=1, padding=enc_blk.conv.padding, output_padding=0)
            z = dec_bn[bn_idx](z)
            z = dec_act[bn_idx](z)
            bn_idx += 1
        # one more upsample if needed
        while down_ct > 0:
            z = self.upsample(z)
            down_ct -= 1
        # final tied deconv from the very first encoder conv to reach input channel count, then head 1x1 for refinement
        W0 = self.encoder[0].conv.weight
        z = F.conv_transpose2d(z, W0, bias=None, stride=1, padding=self.encoder[0].conv.padding, output_padding=0)
        out = self.head(z)
        return out


# -----------------------------
# Trainer: reconstruction (MSE/L1), early stopping, plotting
# -----------------------------

import time
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class TiedAETrainer:
    def __init__(
        self,
        model: TiedConvAE,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        es_patience: int = 30,
        max_epochs: int = 300,
        results_dir: str = "results_ae_tied",
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
        plt.title(f"TiedAE Val | neurons={self.model.total_neurons()} depth={self.model.depth()} widths={self.model.widths_list()} loss={self.loss_type}")
        plt.grid(True, which="both", ls=":")
        out = os.path.join(self.results_dir, "TiedAE_neuron_loss_plot.png")
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
            loss_sum += self.crit(y, x).item() * x.size(0)
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
