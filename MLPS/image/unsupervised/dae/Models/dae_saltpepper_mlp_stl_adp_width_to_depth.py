import copy
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger  # type: ignore
from utils.adp_plot import plot_loss_vs_epoch, plot_loss_vs_neurons  # type: ignore
from utils.adp_contract import run_module_adp
from utils.image_dae_mlp_adp import (
    expand_unsup_depth,
    expand_unsup_width,
    infer_hidden_widths,
    restore_unsup_model,
    snapshot_unsup_model,
    unsup_total_neurons,
)

from .dae_saltpepper_mlp_stl import DAESaltPepperMLP, dae_total_neurons


@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"
    delta: float = 1e-3
    patience: int = 5
    trials_width: int = 10
    trials_depth: int = 5
    ex_k: int = 128
    max_width: int = 2048
    max_depth: int = 5
    max_neurons: int = 10_000_000
    width_stage_margin_patience: int = 5
    width_stage_min_improve_pct: float = 1.0
    depth_stage_margin_patience: int = 5
    depth_stage_min_improve_pct: float = 1.0
    min_new_layer_width: int = 10
    depth_first_seed_width: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    max_epochs: int = 100_000_000
    sp_prob: float = 0.1


def expand_width(model: DAESaltPepperMLP, ex_k: int, max_width: int, device: torch.device) -> Optional[DAESaltPepperMLP]:
    return expand_unsup_width(DAESaltPepperMLP, model, ex_k, max_width, device)


def expand_depth(model: DAESaltPepperMLP, max_depth: int, device: torch.device, min_new_layer_width: int = 10) -> Optional[DAESaltPepperMLP]:
    return expand_unsup_depth(DAESaltPepperMLP, model, max_depth, device, min_new_layer_width=min_new_layer_width)


def total_neurons(model: DAESaltPepperMLP) -> int:
    return unsup_total_neurons(model)


def snapshot_arch_and_state(model: DAESaltPepperMLP, state_dict=None) -> Dict[str, Any]:
    return snapshot_unsup_model(model, state_dict)


def restore_arch_and_state(model_or_snap, snap=None, device=None) -> DAESaltPepperMLP:
    if snap is None:
        snap = model_or_snap
    return restore_unsup_model(DAESaltPepperMLP, snap, device)


def make_loaders(
    dataset: str,
    data_root: str,
    batch_size: int,
    val_split: float,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    ds_name = dataset.lower()
    if ds_name == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        ds_cls = datasets.CIFAR100
    else:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        ds_cls = datasets.CIFAR10

    tf_train = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    tf_eval = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    full_train = ds_cls(root=data_root, train=True, transform=tf_train, download=True)
    full_eval = ds_cls(root=data_root, train=True, transform=tf_eval, download=True)

    n_total = len(full_train)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    train_ds, _ = random_split(full_train, [n_train, n_val], generator=torch.Generator().manual_seed(1337))
    _, val_ds = random_split(full_eval, [n_train, n_val], generator=torch.Generator().manual_seed(1337))

    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    return dl_train, dl_val


def adp_search(
    model: DAESaltPepperMLP,
    dl_train: DataLoader,
    dl_val: DataLoader,
    acfg: ADPConfig,
    device: torch.device,
    logger: Optional[ContinuousLogger],
    results_dir: Path,
    log_loss: bool,
    log_neurons: bool,
) -> Tuple[float, DAESaltPepperMLP, int, int]:
    results_dir.mkdir(parents=True, exist_ok=True)
    if logger is not None:
        logger.log_console(f"[ADP] Mode={acfg.adp_mode}")
    best_val, best_model = run_module_adp(
        globals(),
        model,
        dl_train,
        dl_val,
        acfg,
        device,
        log_loss=log_loss,
        log_neurons=log_neurons,
        results_dir=results_dir,
        logger=logger,
    )
    hidden_widths = infer_hidden_widths(best_model)
    return best_val, best_model, max(hidden_widths), len(hidden_widths)


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="ADP Salt-and-pepper MLP DAE on CIFAR")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--val-split", type=float, default=0.1)

    p.add_argument("--width", type=int, default=512)
    p.add_argument("--depth", type=int, default=3)

    p.add_argument(
        "--adp-mode",
        type=str,
        default="width_to_depth",
        choices=["alt_width", "alt_depth", "width_to_depth", "depth_to_width"],
    )
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--trials-width", type=int, default=10)
    p.add_argument("--trials-depth", type=int, default=2)
    p.add_argument("--ex-k", type=int, default=128)
    p.add_argument("--max-width", type=int, default=2048)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--max-neurons", type=int, default=10_000_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--sp-prob", type=float, default=0.1)

    p.add_argument("--results-dir", type=str, default="results_adp_dae_saltpepper_mlp")
    p.add_argument("--plot-loss", action="store_true")
    p.add_argument("--plot-neurons", action="store_true")

    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dl_train, dl_val = make_loaders(
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
    )

    model = DAESaltPepperMLP(in_channels=3, img_size=32, width=args.width, depth=args.depth).to(device)

    acfg = ADPConfig(
        adp_mode=args.adp_mode,
        delta=args.delta,
        patience=args.patience,
        trials_width=args.trials_width,
        trials_depth=args.trials_depth,
        ex_k=args.ex_k,
        max_width=args.max_width,
        max_depth=args.max_depth,
        max_neurons=args.max_neurons,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        max_epochs=args.max_epochs,
        sp_prob=args.sp_prob,
    )

    results_dir = Path(args.results_dir)
    logger = ContinuousLogger(results_dir, "dae_saltpepper_mlp", args.adp_mode)

    best_val, best_model, best_w, best_d = adp_search(
        model,
        dl_train,
        dl_val,
        acfg,
        device,
        logger=logger,
        results_dir=results_dir,
        log_loss=args.plot_loss,
        log_neurons=args.plot_neurons,
    )

    logger.log_console(f"[DONE] Best val={best_val:.6f}, width={best_w}, depth={best_d}")
    logger.close()
