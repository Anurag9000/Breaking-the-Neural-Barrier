import copy
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger  # type: ignore
from utils.adp_plot import plot_loss_vs_epoch, plot_loss_vs_neurons  # type: ignore
from utils.adp_contract import run_module_adp
from utils.image_dae_mlp_adp import (
    expand_sup_depth,
    expand_sup_width,
    infer_hidden_widths,
    restore_sup_model,
    snapshot_sup_model,
    sup_total_neurons,
)

from .dae_sparse_mlp_sup_stl import SupDAESparseMLP, sup_dae_total_neurons
from ..Runs.run_dae_sparse_mlp_sup_stl import build_dataloaders, add_gaussian_noise  # type: ignore


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
    max_epochs: int = 200
    noise_std: float = 0.1
    lambda_recon: float = 1.0
    lambda_sparse: float = 1e-3


def expand_width(
    model: SupDAESparseMLP,
    num_classes: int,
    ex_k: int,
    max_width: int,
    device: torch.device,
) -> Optional[SupDAESparseMLP]:
    return expand_sup_width(SupDAESparseMLP, model, ex_k, max_width, device)


def expand_depth(
    model: SupDAESparseMLP,
    num_classes: int,
    max_depth: int,
    device: torch.device,
    min_new_layer_width: int = 10,
) -> Optional[SupDAESparseMLP]:
    return expand_sup_depth(SupDAESparseMLP, model, max_depth, device, min_new_layer_width=min_new_layer_width)


def snapshot_arch_and_state(model: SupDAESparseMLP, state: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
    return snapshot_sup_model(model, state)


def restore_arch_and_state(model_or_snap, snap=None, device=None) -> SupDAESparseMLP:
    if snap is None:
        snap = model_or_snap
    return restore_sup_model(SupDAESparseMLP, snap, device)


def total_neurons(model: SupDAESparseMLP) -> int:
    return sup_total_neurons(model)


def train_with_early_stopping(
    model: SupDAESparseMLP,
    dl_train: DataLoader,
    dl_val: DataLoader,
    acfg: ADPConfig,
    device: torch.device,
    history: List[float],
    logger: Optional[ContinuousLogger] = None,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    es_counter = 0

    for epoch in range(1, acfg.max_epochs + 1):
        # Train
        model.train()
        total_train, total_cls, n_train = 0.0, 0.0, 0
        for xb, yb in dl_train:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            xb_noisy = add_gaussian_noise(xb, acfg.noise_std)

            opt.zero_grad(set_to_none=True)
            xb_rec, logits, z = model(xb_noisy)
            loss_recon = mse(xb_rec, xb)
            loss_cls = ce(logits, yb)
            loss_sparse = z.abs().mean()
            loss = acfg.lambda_recon * loss_recon + loss_cls + acfg.lambda_sparse * loss_sparse
            loss.backward()
            if acfg.grad_clip and acfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), acfg.grad_clip)
            opt.step()

            bs = xb.size(0)
            total_train += float(loss.item()) * bs
            total_cls += float(loss_cls.item()) * bs
            n_train += bs
        train_loss = total_train / max(n_train, 1)

        # Validation
        model.eval()
        total_val, n_val = 0.0, 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                xb_noisy = add_gaussian_noise(xb, acfg.noise_std)
                xb_rec, logits, z = model(xb_noisy)
                loss_recon = mse(xb_rec, xb) / xb.size(0)
                loss_cls = ce(logits, yb) / xb.size(0)
                loss_sparse = z.abs().mean()
                loss = acfg.lambda_recon * loss_recon + loss_cls + acfg.lambda_sparse * loss_sparse
                bs = xb.size(0)
                total_val += float(loss.item()) * bs
                n_val += bs
        val_loss = total_val / max(n_val, 1)
        history.append(val_loss)

        improved = val_loss < best_val - acfg.delta
        if improved:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            es_counter = 0
        else:
            es_counter += 1

        msg = (
            f"  Epoch {epoch:03d}/{acfg.max_epochs} | "
            f"Train={train_loss:.6f} | Val={val_loss:.6f} | "
            f"Best={best_val:.6f} | ES={es_counter}/{acfg.patience}"
        )
        if logger:
            logger.log_console(msg)
        else:
            print(msg)

        if es_counter >= acfg.patience:
            if logger:
                logger.log_console(f"  Early stopping at epoch {epoch}")
            else:
                print(f"  Early stopping at epoch {epoch}")
            break

    return best_val, best_state


def make_loaders_for_adp(
    dataset: str,
    data_root: str,
    batch_size: int,
    val_split: float,
    num_workers: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, int]:
    train_loader, val_loader, _, num_classes = build_dataloaders(
        dataset=dataset,
        data_dir=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        val_frac=val_split,
        seed=seed,
    )
    return train_loader, val_loader, num_classes


def adp_search(
    model: SupDAESparseMLP,
    dl_train: DataLoader,
    dl_val: DataLoader,
    acfg: ADPConfig,
    device: torch.device,
    logger: ContinuousLogger,
    results_dir: Path,
    num_classes: int,
    log_loss: bool = False,
    log_neurons: bool = False,
) -> Tuple[float, SupDAESparseMLP, int, int]:
    results_dir.mkdir(parents=True, exist_ok=True)
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

    p = argparse.ArgumentParser(description="ADP supervised sparse MLP DAE encoder + classifier")
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
    p.add_argument("--max-epochs", type=int, default=200)
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
    p.add_argument("--lambda-recon", type=float, default=1.0)
    p.add_argument("--lambda-sparse", type=float, default=1e-3)

    p.add_argument("--results-dir", type=str, default="results_adp_dae_sparse_mlp_sup")
    p.add_argument("--plot-loss", action="store_true")
    p.add_argument("--plot-neurons", action="store_true")

    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dl_train, dl_val, num_classes = make_loaders_for_adp(
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=1337,
    )

    model = SupDAESparseMLP(
        num_classes=num_classes,
        in_channels=3,
        img_size=32,
        width=args.width,
        depth=args.depth,
    ).to(device)

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
        lambda_recon=args.lambda_recon,
        lambda_sparse=args.lambda_sparse,
    )

    results_dir = Path(args.results_dir)
    logger = ContinuousLogger(results_dir, "sup_dae_sparse_mlp", args.adp_mode)

    best_val, best_model, best_w, best_d = adp_search(
        model,
        dl_train,
        dl_val,
        acfg,
        device,
        logger=logger,
        results_dir=results_dir,
        num_classes=num_classes,
        log_loss=args.plot_loss,
        log_neurons=args.plot_neurons,
    )

    logger.log_console(f"[DONE] Best val={best_val:.6f}, width={best_w}, depth={best_d}")
    logger.close()
