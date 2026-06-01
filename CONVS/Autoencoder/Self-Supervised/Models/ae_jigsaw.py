import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Jigsaw / Permutation Prediction (single-model, no EMA, no teacher)
# - Split image into K=GxG patches.
# - Permute patches using an index from a fixed permutation set.
# - Model predicts the permutation ID (classification) from the shuffled patches.
# - This is a *context-predictive AE-style* pretext: no second network, no adversary.
# -----------------------------

class PatchEncoder(nn.Module):
    def __init__(self, in_ch: int = 3, width: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.Conv2d(width, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, 2*width, 3, padding=1), nn.BatchNorm2d(2*width), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.out_dim = 2*width
    def forward(self, x):
        h = self.conv(x)
        return h.flatten(1)  # (B, D)

class JigsawModel(nn.Module):
    def __init__(self, in_ch: int = 3, grid_size: int = 3, width: int = 64, num_permutations: int = 30):
        super().__init__()
        assert grid_size * grid_size >= 2
        self.in_ch = in_ch
        self.G = int(grid_size)
        self.K = self.G * self.G
        self.num_permutations = int(num_permutations)

        self.encoder = PatchEncoder(in_ch=in_ch, width=width)
        D = self.encoder.out_dim
        self.head = nn.Sequential(
            nn.Linear(D * self.K, 4*D), nn.ReLU(inplace=True),
            nn.Linear(4*D, 2*D), nn.ReLU(inplace=True),
            nn.Linear(2*D, num_permutations)
        )

    # capacity stats
    def total_neurons(self) -> int:
        # proxy: channel dims of encoder + MLP hidden dims
        D = self.encoder.out_dim
        return D * self.K + 4*D + 2*D + self.num_permutations

    def depth(self) -> int:
        return 3  # roughly conv stack + 2 FC blocks

    def widths_list(self) -> List[int]:
        return [self.encoder.out_dim, 4*self.encoder.out_dim, 2*self.encoder.out_dim]

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """
        patches: (B, K, C, h, w) shuffled order per sample
        returns logits: (B, num_permutations)
        """
        B, K, C, h, w = patches.shape
        assert K == self.K
        x = patches.view(B*K, C, h, w)
        feat = self.encoder(x)               # (B*K, D)
        feat = feat.view(B, K, -1).contiguous()
        feat = feat.flatten(1)               # (B, K*D)
        logits = self.head(feat)
        return logits


# -----------------------------
# Trainer for Jigsaw task
# -----------------------------
import time
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class JigsawTrainer:
    def __init__(
        self,
        model: JigsawModel,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        es_patience: int = 30,
        max_epochs: int = 200,
        results_dir: str = "results_ae_jigsaw",
    ):
        self.model = model.to(device)
        self.device = device
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.grad_clip = grad_clip
        self.es_patience = es_patience
        self.max_epochs = max_epochs
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

        self.crit = nn.CrossEntropyLoss()

        self.best_val = float("inf")
        self.best_state = None
        self.hist = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "epoch": []}
        self._last_plot_ts = 0.0

    @torch.no_grad()
    def _eval_epoch(self, loader) -> Tuple[float, float]:
        self.model.eval()
        loss_sum, correct, n = 0.0, 0, 0
        for patches, perm_id in loader:
            patches = patches.to(self.device)
            perm_id = perm_id.to(self.device)
            logits = self.model(patches)
            loss = self.crit(logits, perm_id)
            loss_sum += loss.item() * patches.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == perm_id).sum().item()
            n += patches.size(0)
        return loss_sum / max(n, 1), correct / max(n, 1)

    def _maybe_plot(self):
        now = time.time()
        if now - self._last_plot_ts < 60:
            return
        self._last_plot_ts = now
        fig = plt.figure(figsize=(6,4))
        xs = list(range(len(self.hist["val_loss"])))
        plt.subplot(1,2,1)
        plt.semilogy(xs, self.hist["val_loss"], marker="o", linewidth=1)
        plt.title("Val CE (log)")
        plt.grid(True, which="both", ls=":")
        plt.subplot(1,2,2)
        plt.plot(xs, self.hist["val_acc"], marker="o", linewidth=1)
        plt.title("Val Acc")
        plt.grid(True, ls=":")
        out = os.path.join(self.results_dir, "Jigsaw_val_curves.png")
        plt.tight_layout()
        fig.savefig(out)
        plt.close(fig)

    def fit(self, train_loader, val_loader):
        patience = self.es_patience
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            train_loss_sum, correct, n = 0.0, 0, 0
            for patches, perm_id in train_loader:
                patches = patches.to(self.device)
                perm_id = perm_id.to(self.device)
                logits = self.model(patches)
                loss = self.crit(logits, perm_id)

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.opt.step()

                train_loss_sum += loss.item() * patches.size(0)
                pred = logits.argmax(dim=1)
                correct += (pred == perm_id).sum().item()
                n += patches.size(0)

            train_loss = train_loss_sum / max(n, 1)
            train_acc = correct / max(n, 1)

            val_loss, val_acc = self._eval_epoch(val_loader)

            self.hist["train_loss"].append(train_loss)
            self.hist["val_loss"].append(val_loss)
            self.hist["train_acc"].append(train_acc)
            self.hist["val_acc"].append(val_acc)
            self.hist["epoch"].append(epoch)

            improved = val_loss < (self.best_val - 1e-6)
            if improved:
                self.best_val = val_loss
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience = self.es_patience
            else:
                patience -= 1

            self._maybe_plot()
            print(f"Epoch {epoch:03d} | train {train_loss:.4f} acc {train_acc:.3f} | val {val_loss:.4f} acc {val_acc:.3f} | best {self.best_val:.4f}")
            if patience <= 0:
                break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        self._maybe_plot()
        return self.best_val

    @torch.no_grad()
    def evaluate(self, test_loader) -> Tuple[float, float]:
        return self._eval_epoch(test_loader)

    def save(self, path: str):
        torch.save({
            "model": self.model.state_dict(),
            "best_val": self.best_val,
            "hist": self.hist,
            "grid_size": self.model.G,
            "num_permutations": self.model.num_permutations,
        }, path)

    def load(self, path: str, map_location=None):
        blob = torch.load(path, map_location=map_location)
        self.model.load_state_dict(blob["model"])
        self.best_val = blob.get("best_val", float("inf"))
        self.hist = blob.get("hist", self.hist)
