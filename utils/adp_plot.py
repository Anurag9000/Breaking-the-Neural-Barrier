import os
from pathlib import Path
from typing import Sequence, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_loss_vs_epoch(losses: Sequence[float], out_path: Path, title: Optional[str] = None):
    """Plot loss vs. epoch (log scale) and save to out_path."""
    if not losses:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    xs = list(range(1, len(losses) + 1))
    plt.figure(figsize=(5, 4))
    plt.semilogy(xs, losses, marker="o", linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("Val loss (log)")
    if title:
        plt.title(title)
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_loss_vs_neurons(neurons: Sequence[int], losses: Sequence[float], out_path: Path, title: Optional[str] = None):
    """Plot best val loss vs. total neurons (log scale) and save to out_path."""
    if not neurons or not losses or len(neurons) != len(losses):
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    plt.semilogy(neurons, losses, marker="o", linewidth=1)
    plt.xlabel("Total neurons")
    plt.ylabel("Best val loss (log)")
    if title:
        plt.title(title)
    plt.grid(True, which="both", ls=":")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
