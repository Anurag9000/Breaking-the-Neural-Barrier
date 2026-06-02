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

from .dae_contractive_mlp_stl import DAEContractiveMLP, dae_total_neurons


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
    noise_std: float = 0.1
    contractive_lambda: float = 1e-4


def expand_width(model: DAEContractiveMLP, ex_k: int, max_width: int, device: torch.device) -> Optional[DAEContractiveMLP]:
    return expand_unsup_width(DAEContractiveMLP, model, ex_k, max_width, device)


def expand_depth(model: DAEContractiveMLP, max_depth: int, device: torch.device, min_new_layer_width: int = 10) -> Optional[DAEContractiveMLP]:
    return expand_unsup_depth(DAEContractiveMLP, model, max_depth, device, min_new_layer_width=min_new_layer_width)


def total_neurons(model: DAEContractiveMLP) -> int:
    return unsup_total_neurons(model)


def snapshot_arch_and_state(model: DAEContractiveMLP, state_dict=None) -> Dict[str, Any]:
    return snapshot_unsup_model(model, state_dict)


def restore_arch_and_state(model_or_snap, snap=None, device=None) -> DAEContractiveMLP:
    if snap is None:
        snap = model_or_snap
    return restore_unsup_model(DAEContractiveMLP, snap, device)


def make_loaders(
    dataset: str,
    data_root: str,
    batch_size: int,
    val_split: float,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    if dataset.lower() == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        ds_class = datasets.CIFAR10
    elif dataset.lower() == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        ds_class = datasets.CIFAR100
    else:
        raise ValueError("dataset must be cifar10 or cifar100")

    tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    full = ds_class(root=data_root, train=True, transform=tf, download=True)
    n_val = int(len(full) * val_split)
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(full, [n_train, n_val])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader


def train_with_early_stopping(
    model: DAEContractiveMLP,
    dl_train: DataLoader,
    dl_val: DataLoader,
    acfg: ADPConfig,
    device: torch.device,
    history: List[float],
    logger: Optional[ContinuousLogger] = None,
    verbose: bool = True,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    mse = nn.MSELoss()
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    es_counter = 0

    for epoch in range(1, acfg.max_epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, _ in dl_train:
            xb = xb.to(device, non_blocking=True)
            xb_noisy = add_gaussian_noise(xb, acfg.noise_std)

            opt.zero_grad(set_to_none=True)
            xb_rec, _ = model(xb_noisy)
            recon_loss = mse(xb_rec, xb)
            # FAITHFULNESS FIX: Use correct input-dependent contractive penalty
            c_pen = model.contractive_loss(xb)
            loss = recon_loss + acfg.contractive_lambda * c_pen
            loss.backward()
            if acfg.grad_clip is not None and acfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), acfg.grad_clip)
            opt.step()

            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)

        train_loss = total / max(n, 1)

        model.eval()
        val_total, val_n = 0.0, 0
        with torch.no_grad():
            for xb, _ in dl_val:
                xb = xb.to(device, non_blocking=True)
                xb_noisy = add_gaussian_noise(xb, acfg.noise_std)
                xb_rec, _ = model(xb_noisy)
                recon = nn.functional.mse_loss(xb_rec, xb, reduction="sum")
                # FAITHFULNESS FIX: input-dependent penalty
                c_pen = model.contractive_loss(xb) * xb.size(0) # scale to sum if mean returned
                val_total += float(recon.item()) + float(acfg.contractive_lambda * c_pen.item())
                val_n += xb.size(0)

        val_loss = val_total / max(val_n, 1)
        history.append(val_loss)

        improved = val_loss < best_val - acfg.delta
        if improved:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            es_counter = 0
        else:
            es_counter += 1

        if logger is not None and verbose:
            logger.log_epoch(epoch, train_loss, val_loss, best_val, es_counter)

        if es_counter >= acfg.patience:
            break

    model.load_state_dict(best_state)
    return best_val, best_state


def adp_search(
    model: DAEContractiveMLP,
    dl_train: DataLoader,
    dl_val: DataLoader,
    acfg: ADPConfig,
    device: torch.device,
    logger: Optional[ContinuousLogger],
    results_dir: Path,
    log_loss: bool,
    log_neurons: bool,
) -> Tuple[float, DAEContractiveMLP, int, int]:
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

    p = argparse.ArgumentParser(description="ADP Contractive MLP DAE on CIFAR")
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
    p.add_argument("--noise-std", type=float, default=0.1)
    p.add_argument("--contractive-lambda", type=float, default=1e-4)

    p.add_argument("--results-dir", type=str, default="results_adp_dae_contractive_mlp")
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

    model = DAEContractiveMLP(in_channels=3, img_size=32, width=args.width, depth=args.depth).to(device)

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
        noise_std=args.noise_std,
        contractive_lambda=args.contractive_lambda,
    )

    results_dir = Path(args.results_dir)
    logger = ContinuousLogger(results_dir, "dae_contractive_mlp", args.adp_mode)

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
