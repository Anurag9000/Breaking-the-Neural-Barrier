import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Optional


def plot_loss_vs_epoch(
    val_history: List[float],
    save_path: Path,
    title: str = "Loss vs Epoch",
    log_scale: bool = True
):
    """
    Plot validation loss vs epoch.
    
    Args:
        val_history: List of validation losses per epoch
        save_path: Path to save the plot
        title: Plot title
        log_scale: Whether to use log scale for y-axis
    """
    plt.figure(figsize=(10, 6))
    epochs = list(range(1, len(val_history) + 1))
    
    plt.plot(epochs, val_history, 'b-', linewidth=2, marker='o', markersize=4, alpha=0.7)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    if log_scale and len(val_history) > 0 and min(val_history) > 0:
        plt.yscale('log')
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved loss vs epoch plot to {save_path}")


def plot_loss_vs_neurons(
    neurons: List[int],
    losses: List[float],
    save_path: Path,
    title: str = "Loss vs Neurons",
    log_scale_x: bool = False,
    log_scale_y: bool = True
):
    """
    Plot validation loss vs number of neurons.
    
    Args:
        neurons: List of neuron counts
        losses: List of corresponding validation losses
        save_path: Path to save the plot
        title: Plot title
        log_scale_x: Whether to use log scale for x-axis
        log_scale_y: Whether to use log scale for y-axis
    """
    plt.figure(figsize=(10, 6))
    
    plt.plot(neurons, losses, 'r-', linewidth=2, marker='s', markersize=6, alpha=0.7)
    plt.xlabel('Number of Neurons', fontsize=12)
    plt.ylabel('Validation Loss', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    if log_scale_x and len(neurons) > 0 and min(neurons) > 0:
        plt.xscale('log')
    if log_scale_y and len(losses) > 0 and min(losses) > 0:
        plt.yscale('log')
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved loss vs neurons plot to {save_path}")


def plot_comprehensive_stats(
    val_history: List[float],
    improvements: List[tuple],
    save_dir: Path,
    title_prefix: str = "ADP"
):
    """
    Create comprehensive plots for ADP training.
    
    Args:
        val_history: List of all validation losses
        improvements: List of (neurons, loss) tuples for each improvement
        save_dir: Directory to save plots
        title_prefix: Prefix for plot titles
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Loss vs Epoch
    if val_history:
        plot_loss_vs_epoch(
            val_history,
            save_dir / "loss_vs_epoch.png",
            title=f"{title_prefix} - Loss vs Epoch"
        )
    
    # Plot 2: Loss vs Neurons
    if improvements:
        neurons = [n for n, _ in improvements]
        losses = [l for _, l in improvements]
        plot_loss_vs_neurons(
            neurons,
            losses,
            save_dir / "loss_vs_neurons.png",
            title=f"{title_prefix} - Loss vs Neurons"
        )
    
    # Plot 3: Combined view
    if val_history and improvements:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: Loss vs Epoch
        epochs = list(range(1, len(val_history) + 1))
        ax1.plot(epochs, val_history, 'b-', linewidth=2, marker='o', markersize=3, alpha=0.7)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Validation Loss', fontsize=12)
        ax1.set_title('Loss vs Epoch', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        if min(val_history) > 0:
            ax1.set_yscale('log')
        
        # Right: Loss vs Neurons
        neurons = [n for n, _ in improvements]
        losses = [l for _, l in improvements]
        ax2.plot(neurons, losses, 'r-', linewidth=2, marker='s', markersize=6, alpha=0.7)
        ax2.set_xlabel('Number of Neurons', fontsize=12)
        ax2.set_ylabel('Validation Loss', fontsize=12)
        ax2.set_title('Loss vs Neurons', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        # Keep neurons on a linear scale
        if min(losses) > 0:
            ax2.set_yscale('log')
        
        plt.suptitle(f"{title_prefix} - Comprehensive Training Stats", fontsize=15, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_dir / "comprehensive_stats.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved comprehensive stats to {save_dir / 'comprehensive_stats.png'}")


def plot_best_loss_per_neurons_from_csv(
    csv_path: Path,
    save_path: Optional[Path] = None,
    title: str = "Loss vs Neurons (best per width)",
    log_scale_x: bool = False,
    log_scale_y: bool = True,
) -> None:
    """
    Utility to recompute loss-vs-neurons from an ADP training_stats.csv.

    For each distinct neuron count, it takes the *best* (minimum) validation
    loss across all epochs and plots that curve.
    """
    csv_path = Path(csv_path)
    if save_path is None:
        save_path = csv_path.with_name("loss_vs_neurons_best_per_width.png")

    best_by_neurons = {}
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                neurons = int(row["neurons"])
                loss = float(row["val_loss"])
            except (KeyError, ValueError):
                continue
            if loss <= 0:
                continue
            prev = best_by_neurons.get(neurons, float("inf"))
            if loss < prev:
                best_by_neurons[neurons] = loss

    if not best_by_neurons:
        print(f"No valid rows found in {csv_path} to plot.")
        return

    neurons = sorted(best_by_neurons.keys())
    losses = [best_by_neurons[n] for n in neurons]

    plot_loss_vs_neurons(
        neurons,
        losses,
        save_path,
        title=title,
        log_scale_x=log_scale_x,
        log_scale_y=log_scale_y,
    )
