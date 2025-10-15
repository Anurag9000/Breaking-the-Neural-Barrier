import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Predictive Sequence Autoencoder (single-model, GRU)
# - Input: (B, T, F). Predicts next step x_{t+1} from x_{<=t} (teacher forcing during training).
# - Loss computed between predictions and the next-step targets (shifted by 1).
# - Can be used for images treated as sequences (e.g., CIFAR rows).
# -----------------------------

class PredictiveSeqAE(nn.Module):
    def __init__(
        self,
        feature_dim: int,   # F
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.out_proj = nn.Linear(hidden_size, feature_dim)

    # ---- capacity stats ----
    def total_neurons(self) -> int:
        return self.hidden_size * self.num_layers

    def depth(self) -> int:
        return self.num_layers

    def widths_list(self):
        return [self.hidden_size] * self.num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F)
        Returns y_pred with shape (B, T, F), where y_pred[:, t] intends to match x[:, t+1].
        The last timestep has no target and may be ignored by the trainer.
        """
        h, _ = self.rnn(x)            # (B,T,H)
        y = self.out_proj(h)          # (B,T,F)
        return y


# -----------------------------
# Trainer (next-step loss), early stopping, plotting
# -----------------------------

import time
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

class PredictiveSeqAETrainer:
    def __init__(
        self,
        model: PredictiveSeqAE,
        device: torch.device,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip: float = 1.0,
        es_patience: int = 30,
        max_epochs: int = 200,
        results_dir: str = "results_ae_predictive",
        loss_type: str = "mse",
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

        self.best_val = float("inf")
        self.best_state = None
        self.hist = {"train_loss": [], "val_loss": [], "neurons": [], "epoch": []}
        self._last_plot_ts = 0.0

    def _next_step_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: (B,T,F) predicting next step; compare pred[:, :-1] with target[:, 1:]
        return self.crit(pred[:, :-1, :], target[:, 1:, :])

    @torch.no_grad()
    def _eval_epoch(self, loader) -> float:
        self.model.eval()
        loss_sum, n = 0.0, 0
        for x, _ in loader:
            x = x.to(self.device)
            y = self.model(x)
            loss = self._next_step_loss(y, x)
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
        plt.ylabel("Val Next-Step (log)")
        plt.title(f"PredictiveSeqAE Val | neurons={self.model.total_neurons()} depth={self.model.depth()} widths={self.model.widths_list()} loss={self.loss_type}")
        plt.grid(True, which="both", ls=":")
        out = os.path.join(self.results_dir, "PredictiveSeqAE_neuron_loss_plot.png")
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
                loss = self._next_step_loss(y, x)
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
            loss = self._next_step_loss(y, x)
            loss_sum += loss.item() * x.size(0)
            n += x.size(0)
        return loss_sum / max(n, 1)

    def save(self, path: str):
        torch.save({
            "model": self.model.state_dict(),
            "best_val": self.best_val,
            "hist": self.hist,
            "feature_dim": self.model.feature_dim,
            "hidden_size": self.model.hidden_size,
            "num_layers": self.model.num_layers,
            "loss_type": self.loss_type,
        }, path)

    def load(self, path: str, map_location=None):
        blob = torch.load(path, map_location=map_location)
        self.model.load_state_dict(blob["model"])
        self.best_val = blob.get("best_val", float("inf"))
        self.hist = blob.get("hist", self.hist)
