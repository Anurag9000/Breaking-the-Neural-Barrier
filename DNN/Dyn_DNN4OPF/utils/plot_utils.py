import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
import torch
from typing import List
from torch.utils.data import DataLoader
from Dyn_DNN4OPF.utils.config import get_io_dims_from_loader

def load_logs(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def get_param_names(loader: DataLoader) -> List[str]:
    """
    Generate output parameter names dynamically from the data loader.

    Names are returned in this order:
      • pg0 … pg_{n_gen-1}
      • qg0 … qg_{n_gen-1}
      • Va0 … Va_{n_bus-1}
      • Vm0 … Vm_{n_bus-1}

    Uses get_io_dims_from_loader to infer:
      input_dim  = 2 * n_bus
      output_dim = 2 * (n_gen + n_bus)
    """
    # infer dimensions exactly as in your run-scripts
    input_dim, output_dim = get_io_dims_from_loader(loader)
    n_bus = input_dim // 2
    n_gen = (output_dim // 2) - n_bus

    pg = [f"pg{i}" for i in range(n_gen)]
    qg = [f"qg{i}" for i in range(n_gen)]
    va = [f"Va{i}" for i in range(n_bus)]
    vm = [f"Vm{i}" for i in range(n_bus)]

    return pg + qg + va + vm

def plot_aggregate(df: pd.DataFrame, model_name: str, out_dir: Path) -> None:
    """
    Plot aggregate train vs. val loss as side-by-side linear & log-scale subplots.
    """
    # ensure output directory exists
    out_dir.mkdir(exist_ok=True, parents=True)
    # build filename
    fname = out_dir / f"{model_name}_aggregate_loss.png"

    # create 1×2 subplot: linear & log
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # — Linear scale —
    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"],   label="val", linestyle="--")
    axes[0].set_title(f"{model_name} — aggregate (linear)")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(True)

    # — Log scale —
    axes[1].plot(df["epoch"], df["train_loss"], label="train")
    axes[1].plot(df["epoch"], df["val_loss"],   label="val", linestyle="--")
    axes[1].set_title(f"{model_name} — aggregate (log)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(True, which="both")

    plt.tight_layout()
    plt.savefig(fname)
    plt.close()


def plot_per_output(
    df: pd.DataFrame,
    model_name: str,
    out_dir: Path,
    param_names: List[str]
) -> None:
    """
    Plot per-output MSE across epochs, emitting two figures per parameter:
      - {model_name}_{param}_train_mse.png
      - {model_name}_{param}_val_mse.png
    Each figure is a 1×2 subplot: linear & log scale.
    """
    # ensure output directory exists
    out_dir.mkdir(exist_ok=True, parents=True)

    # find and sort the per-output columns
    train_cols = sorted(
        [c for c in df.columns if c.startswith("train_output_")],
        key=lambda c: int(c.split("_")[2])
    )
    val_cols = sorted(
        [c for c in df.columns if c.startswith("val_output_")],
        key=lambda c: int(c.split("_")[2])
    )

    for idx, (tc, vc) in enumerate(zip(train_cols, val_cols)):
        name = param_names[idx]

        # ---- TRAIN-ONLY plot ----
        fname_tr = out_dir / f"{model_name}_{name}_train_mse.png"
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # linear
        axes[0].plot(df["epoch"], df[tc], label="train")
        axes[0].set_title(f"{model_name} — {name} train (linear)")
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("MSE")
        axes[0].grid(True)

        # log
        axes[1].plot(df["epoch"], df[tc], label="train")
        axes[1].set_title(f"{model_name} — {name} train (log)")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("MSE")
        axes[1].set_yscale("log")
        axes[1].grid(True, which="both")

        plt.tight_layout()
        plt.savefig(fname_tr)
        plt.close(fig)

        # ---- VAL-ONLY plot ----
        fname_va = out_dir / f"{model_name}_{name}_val_mse.png"
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # linear
        axes[0].plot(df["epoch"], df[vc], label="val", color="C1")
        axes[0].set_title(f"{model_name} — {name} val (linear)")
        axes[0].set_xlabel("epoch"); axes[0].set_ylabel("MSE")
        axes[0].grid(True)

        # log
        axes[1].plot(df["epoch"], df[vc], label="val", color="C1")
        axes[1].set_title(f"{model_name} — {name} val (log)")
        axes[1].set_xlabel("epoch"); axes[1].set_ylabel("MSE")
        axes[1].set_yscale("log")
        axes[1].grid(True, which="both")

        plt.tight_layout()
        plt.savefig(fname_va)
        plt.close(fig)

def save_metadata_to_json(metadata, filename="metadata.json"):
    """
    Save metadata to a JSON file.

    Args:
        metadata (dict): Metadata to save.
        filename (str): Path to the JSON file.

    Returns:
        None
    """
    if isinstance(metadata, torch.Tensor):
        metadata = metadata.tolist()

    dir_path = os.path.dirname(filename)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)

    # Save metadata to JSON
    with open(filename, "w") as json_file:
        json.dump(metadata, json_file, indent=4)
    print(f"Metadata successfully saved to {filename}")

