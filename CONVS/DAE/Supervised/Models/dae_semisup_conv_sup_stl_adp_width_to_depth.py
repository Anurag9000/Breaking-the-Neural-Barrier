import copy
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from itertools import zip_longest

sys.path.append(str(Path(__file__).resolve().parents[4]))
from utils.adp_logging import ContinuousLogger  # type: ignore
from utils.adp_plot import plot_loss_vs_epoch, plot_loss_vs_neurons  # type: ignore

from .dae_semisup_conv_sup_stl import SupDAESemiConv, sup_dae_semisup_total_neurons
from ..Runs.run_dae_semisup_conv_sup_stl import (  # type: ignore
    build_dataloaders_semisup,
    add_gaussian_noise,
)


@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"
    delta: float = 1e-3
    patience: int = 20
    trials_width: int = 2
    trials_depth: int = 2
    ex_k: int = 16
    max_width: int = 512
    max_depth: int = 16
    max_neurons: int = 5_000_000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    max_epochs: int = 300
    noise_std: float = 0.1
    lambda_recon: float = 1.0
    label_frac: float = 0.1


def _resize_tensor(to_shape: torch.Size, src: torch.Tensor) -> torch.Tensor:
    tgt = torch.zeros(to_shape, device=src.device, dtype=src.dtype)
    common = tuple(min(a, b) for a, b in zip(to_shape, src.shape))
    slices = tuple(slice(0, c) for c in common)
    tgt[slices] = src[slices]
    return tgt


def _merge_state(new_state: Dict[str, torch.Tensor], old_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    merged: Dict[str, torch.Tensor] = {}
    for k, v in new_state.items():
        if k in old_state:
            ov = old_state[k]
            merged[k] = ov if ov.shape == v.shape else _resize_tensor(v.shape, ov)
        else:
            merged[k] = v
    return merged


def rebuild_model(
    model: SupDAESemiConv,
    num_classes: int,
    width: int,
    depth: int,
    device: torch.device,
) -> SupDAESemiConv:
    new_model = SupDAESemiConv(
        num_classes=num_classes,
        in_channels=model.in_channels,
        width=width,
        depth=depth,
        pool_after=[2],
    ).to(device)
    merged = _merge_state(new_model.state_dict(), model.state_dict())
    new_model.load_state_dict(merged, strict=False)
    return new_model


def expand_width(
    model: SupDAESemiConv,
    num_classes: int,
    ex_k: int,
    max_width: int,
    device: torch.device,
) -> Optional[SupDAESemiConv]:
    new_w = min(max_width, model.width + ex_k)
    if new_w == model.width:
        return None
    return rebuild_model(model, num_classes, new_w, model.depth, device)


def expand_depth(
    model: SupDAESemiConv,
    num_classes: int,
    max_depth: int,
    device: torch.device,
) -> Optional[SupDAESemiConv]:
    if model.depth >= max_depth:
        return None
    return rebuild_model(model, num_classes, model.width, model.depth + 1, device)


def snapshot_arch_and_state(model: SupDAESemiConv, state: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
    st = state if state is not None else model.state_dict()
    return {
        "in_channels": model.in_channels,
        "width": model.width,
        "depth": model.depth,
        "num_classes": model.num_classes,
        "state": copy.deepcopy(st),
    }


def restore_arch_and_state(snap: Dict[str, Any], device: torch.device) -> SupDAESemiConv:
    mdl = SupDAESemiConv(
        num_classes=snap["num_classes"],
        in_channels=snap.get("in_channels", 3),
        width=snap["width"],
        depth=snap["depth"],
        pool_after=[2],
    ).to(device)
    mdl.load_state_dict(snap["state"], strict=False)
    return mdl


def train_with_early_stopping(
    model: SupDAESemiConv,
    dl_labeled: DataLoader,
    dl_unlabeled: DataLoader,
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
        model.train()
        total_train, total_cls, n_train = 0.0, 0.0, 0
        for lab_batch, unlab_batch in zip_longest(dl_labeled, dl_unlabeled):
            loss = 0.0
            loss_cls = 0.0
            bs_total = 0

            if lab_batch is not None:
                xb_lab, yb_lab = lab_batch
                xb_lab = xb_lab.to(device, non_blocking=True)
                yb_lab = yb_lab.to(device, non_blocking=True)
                xb_lab_noisy = add_gaussian_noise(xb_lab, acfg.noise_std)
                x_rec_lab, logits_lab = model(xb_lab_noisy)
                loss_recon_lab = mse(x_rec_lab, xb_lab)
                loss_cls_lab = ce(logits_lab, yb_lab)
                loss = loss + acfg.lambda_recon * loss_recon_lab + loss_cls_lab
                loss_cls = loss_cls + loss_cls_lab
                bs_total += xb_lab.size(0)

            if unlab_batch is not None:
                xb_unlab, _ = unlab_batch
                xb_unlab = xb_unlab.to(device, non_blocking=True)
                xb_unlab_noisy = add_gaussian_noise(xb_unlab, acfg.noise_std)
                x_rec_unlab, _ = model(xb_unlab_noisy)
                loss_recon_unlab = mse(x_rec_unlab, xb_unlab)
                loss = loss + acfg.lambda_recon * loss_recon_unlab
                bs_total += xb_unlab.size(0)

            if bs_total == 0:
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if acfg.grad_clip is not None and acfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), acfg.grad_clip)

            opt.step()

            total_train += float(loss.item()) * bs_total
            total_cls += float(loss_cls.item()) * (lab_batch[0].size(0) if lab_batch is not None else 0)  # type: ignore[index]
            n_train += bs_total

        train_loss = total_train / max(n_train, 1)

        model.eval()
        total_val, total_val_cls, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                xb_noisy = add_gaussian_noise(xb, acfg.noise_std)
                x_rec, logits = model(xb_noisy)

                loss_recon = mse(x_rec, xb)
                loss_cls = ce(logits, yb)
                loss = acfg.lambda_recon * loss_recon + loss_cls

                bs = xb.size(0)
                total_val += float(loss.item()) * bs
                total_val_cls += float(loss_cls.item()) * bs
                n_val += bs

        val_loss = total_val / max(n_val, 1)
        val_cls = total_val_cls / max(n_val, 1)
        history.append(val_loss)

        if logger is not None:
            logger.log_epoch(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                extra={"train_cls": train_loss, "val_cls": val_cls, "width": model.width, "depth": model.depth},
            )

        if val_loss < best_val - acfg.delta:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            es_counter = 0
        else:
            es_counter += 1

        if es_counter >= acfg.patience:
            break

    return best_val, best_state


def make_loaders_for_adp(
    dataset: str,
    data_root: str,
    batch_size: int,
    val_split: float,
    label_frac: float,
    num_workers: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    lab, unlab, val, _, num_classes = build_dataloaders_semisup(
        dataset=dataset,
        data_dir=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        val_frac=val_split,
        label_frac=label_frac,
        seed=seed,
    )
    return lab, unlab, val, num_classes


def adp_search(
    model: SupDAESemiConv,
    dl_labeled: DataLoader,
    dl_unlabeled: DataLoader,
    dl_val: DataLoader,
    acfg: ADPConfig,
    device: torch.device,
    logger: ContinuousLogger,
    results_dir: Path,
    num_classes: int,
    log_loss: bool = False,
    log_neurons: bool = False,
) -> Tuple[float, SupDAESemiConv, int, int]:
    results_dir.mkdir(parents=True, exist_ok=True)
    logger.log_console(f"[ADP] Mode={acfg.adp_mode}")

    val_history: List[float] = []
    improvements: List[Tuple[int, float]] = []

    global_best_val, global_best_state = train_with_early_stopping(
        model, dl_labeled, dl_unlabeled, dl_val, acfg, device, val_history, logger=logger
    )
    global_best_snap = snapshot_arch_and_state(model, global_best_state)

    def can_widen(m: SupDAESemiConv) -> bool:
        return (
            sup_dae_semisup_total_neurons(m.width + acfg.ex_k, m.depth, num_classes) <= acfg.max_neurons
            and m.width < acfg.max_width
        )

    def can_deepen(m: SupDAESemiConv) -> bool:
        return (
            sup_dae_semisup_total_neurons(m.width, m.depth + 1, num_classes) <= acfg.max_neurons
            and m.depth < acfg.max_depth
        )

    def optimize_width_at_fixed_depth(
        snap: Dict[str, Any],
        best_val: float,
    ) -> Tuple[Dict[str, Any], float]:
        local_best_val = best_val
        local_best_snap = snap
        fail = 0
        while fail < acfg.trials_width:
            curr = restore_arch_and_state(local_best_snap, device)
            if not can_widen(curr):
                break
            wider = expand_width(curr, num_classes, acfg.ex_k, acfg.max_width, device)
            if wider is None:
                break

            val_loss, state = train_with_early_stopping(
                wider,
                dl_labeled,
                dl_unlabeled,
                dl_val,
                acfg,
                device,
                val_history,
            )
            if val_loss < local_best_val - acfg.delta:
                local_best_val = val_loss
                local_best_snap = snapshot_arch_and_state(wider, state)
                fail = 0
                neurons = sup_dae_semisup_total_neurons(wider.width, wider.depth, num_classes)
                improvements.append((neurons, val_loss))
            else:
                fail += 1
        return local_best_snap, local_best_val

    def optimize_depth_at_fixed_width(
        snap: Dict[str, Any],
        best_val: float,
    ) -> Tuple[Dict[str, Any], float]:
        local_best_val = best_val
        local_best_snap = snap
        fail = 0
        while fail < acfg.trials_depth:
            curr = restore_arch_and_state(local_best_snap, device)
            if not can_deepen(curr):
                break
            deeper = expand_depth(curr, num_classes, acfg.max_depth, device)
            if deeper is None:
                break

            val_loss, state = train_with_early_stopping(
                deeper,
                dl_labeled,
                dl_unlabeled,
                dl_val,
                acfg,
                device,
                val_history,
            )
            if val_loss < local_best_val - acfg.delta:
                local_best_val = val_loss
                local_best_snap = snapshot_arch_and_state(deeper, state)
                fail = 0
                neurons = sup_dae_semisup_total_neurons(deeper.width, deeper.depth, num_classes)
                improvements.append((neurons, val_loss))
            else:
                fail += 1
        return local_best_snap, local_best_val

    mode = acfg.adp_mode
    if mode in ("width_only", "width"):
        global_best_snap, global_best_val = optimize_width_at_fixed_depth(global_best_snap, global_best_val)
    elif mode in ("depth_only", "depth"):
        global_best_snap, global_best_val = optimize_depth_at_fixed_width(global_best_snap, global_best_val)
    elif mode == "width_to_depth":
        snap, val = optimize_width_at_fixed_depth(global_best_snap, global_best_val)
        if val < global_best_val - acfg.delta:
            global_best_snap, global_best_val = snap, val
        snap, val = optimize_depth_at_fixed_width(global_best_snap, global_best_val)
        if val < global_best_val - acfg.delta:
            global_best_snap, global_best_val = snap, val
    elif mode == "depth_to_width":
        snap, val = optimize_depth_at_fixed_width(global_best_snap, global_best_val)
        if val < global_best_val - acfg.delta:
            global_best_snap, global_best_val = snap, val
        snap, val = optimize_width_at_fixed_depth(global_best_snap, global_best_val)
        if val < global_best_val - acfg.delta:
            global_best_snap, global_best_val = snap, val
    elif mode in ("alt_width", "alt_depth"):
        phase = "width" if mode == "alt_width" else "depth"
        sat_w, sat_d = False, False
        while not (sat_w and sat_d):
            improved = False
            if phase == "width":
                snap, val = optimize_width_at_fixed_depth(global_best_snap, global_best_val)
                if val < global_best_val - acfg.delta:
                    global_best_val = val
                    global_best_snap = snap
                    improved = True
                sat_w = not improved
                phase = "depth"
            else:
                snap, val = optimize_depth_at_fixed_width(global_best_snap, global_best_val)
                if val < global_best_val - acfg.delta:
                    global_best_val = val
                    global_best_snap = snap
                    improved = True
                sat_d = not improved
                phase = "width"
    else:
        logger.log_console(f"[WARN] Unknown adp_mode={mode}, skipping search.")

    if log_loss:
        plot_loss_vs_epoch(val_history, results_dir / "loss_vs_epoch.png", title="SupDAESemiConv")
    if log_neurons and improvements:
        ns = [n for n, _ in improvements]
        vs = [v for _, v in improvements]
        plot_loss_vs_neurons(ns, vs, results_dir / "loss_vs_neurons.png", title="SupDAESemiConv")

    final_model = restore_arch_and_state(global_best_snap, device)
    return global_best_val, final_model, final_model.width, final_model.depth


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="ADP semi-supervised Conv DAE encoder + classifier")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--label-frac", type=float, default=0.1)

    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)

    p.add_argument(
        "--adp-mode",
        type=str,
        default="width_to_depth",
        choices=["alt_width", "alt_depth", "width_to_depth", "depth_to_width"],
    )
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--trials-width", type=int, default=2)
    p.add_argument("--trials-depth", type=int, default=2)
    p.add_argument("--ex-k", type=int, default=16)
    p.add_argument("--max-width", type=int, default=512)
    p.add_argument("--max-depth", type=int, default=16)
    p.add_argument("--max-neurons", type=int, default=5_000_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--noise-std", type=float, default=0.1)
    p.add_argument("--lambda-recon", type=float, default=1.0)

    p.add_argument("--results-dir", type=str, default="results_adp_dae_semisup_conv_sup")
    p.add_argument("--plot-loss", action="store_true")
    p.add_argument("--plot-neurons", action="store_true")

    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dl_lab, dl_unlab, dl_val, num_classes = make_loaders_for_adp(
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_split=args.val_split,
        label_frac=args.label_frac,
        num_workers=args.num_workers,
        seed=1337,
    )

    model = SupDAESemiConv(
        num_classes=num_classes,
        in_channels=3,
        width=args.width,
        depth=args.depth,
        pool_after=[2],
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
        label_frac=args.label_frac,
    )

    results_dir = Path(args.results_dir)
    logger = ContinuousLogger(results_dir, "sup_dae_semisup_conv", args.adp_mode)

    best_val, best_model, best_w, best_d = adp_search(
        model,
        dl_lab,
        dl_unlab,
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
