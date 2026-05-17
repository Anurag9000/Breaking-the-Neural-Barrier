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

from .dae_blockmask_mlp_stl import DAEBlockMaskMLP, dae_total_neurons


@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"
    delta: float = 1e-3
    patience: int = 20
    trials_width: int = 2
    trials_depth: int = 2
    ex_k: int = 128
    max_width: int = 2048
    max_depth: int = 8
    max_neurons: int = 10_000_000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    max_epochs: int = 100_000_000
    block_frac: float = 0.15


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


def rebuild_model(model: DAEBlockMaskMLP, width: int, depth: int, device: torch.device) -> DAEBlockMaskMLP:
    new_model = DAEBlockMaskMLP(
        in_channels=model.in_channels,
        img_size=model.img_size,
        width=width,
        depth=depth,
    ).to(device)
    merged = _merge_state(new_model.state_dict(), model.state_dict())
    new_model.load_state_dict(merged, strict=False)
    return new_model


def expand_width(model: DAEBlockMaskMLP, ex_k: int, max_width: int, device: torch.device) -> Optional[DAEBlockMaskMLP]:
    new_w = min(max_width, model.width + ex_k)
    if new_w == model.width:
        return None
    return rebuild_model(model, new_w, model.depth, device)


def expand_depth(model: DAEBlockMaskMLP, max_depth: int, device: torch.device) -> Optional[DAEBlockMaskMLP]:
    if model.depth >= max_depth:
        return None
    return rebuild_model(model, model.width, model.depth + 1, device)


def snapshot_arch_and_state(model: DAEBlockMaskMLP, state: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
    st = state if state is not None else model.state_dict()
    return {
        "width": model.width,
        "depth": model.depth,
        "in_channels": model.in_channels,
        "img_size": model.img_size,
        "state": copy.deepcopy(st),
    }


def restore_arch_and_state(snap: Dict[str, Any], device: torch.device) -> DAEBlockMaskMLP:
    mdl = DAEBlockMaskMLP(
        in_channels=snap.get("in_channels", 3),
        img_size=snap.get("img_size", 32),
        width=snap["width"],
        depth=snap["depth"],
    ).to(device)
    mdl.load_state_dict(snap["state"], strict=False)
    return mdl


def add_block_mask_noise(x: torch.Tensor, block_frac: float) -> torch.Tensor:
    if block_frac <= 0.0:
        return x
    b, c, h, w = x.shape
    area = h * w
    block_area = max(1, int(area * block_frac))
    side = max(1, int(block_area ** 0.5))
    bh = min(h, side)
    bw = min(w, side)
    x_noisy = x.clone()
    for i in range(b):
        top = torch.randint(0, max(1, h - bh + 1), (1,), device=x.device).item()
        left = torch.randint(0, max(1, w - bw + 1), (1,), device=x.device).item()
        x_noisy[i, :, top : top + bh, left : left + bw] = 0.0
    return x_noisy


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
    model: DAEBlockMaskMLP,
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
            xb_noisy = add_block_mask_noise(xb, acfg.block_frac)

            opt.zero_grad(set_to_none=True)
            xb_rec, _ = model(xb_noisy)
            loss = mse(xb_rec, xb)
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
                xb_noisy = add_block_mask_noise(xb, acfg.block_frac)
                xb_rec, _ = model(xb_noisy)
                val_total += float(nn.functional.mse_loss(xb_rec, xb, reduction="sum").item())
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
    model: DAEBlockMaskMLP,
    dl_train: DataLoader,
    dl_val: DataLoader,
    acfg: ADPConfig,
    device: torch.device,
    logger: Optional[ContinuousLogger],
    results_dir: Path,
    log_loss: bool,
    log_neurons: bool,
) -> Tuple[float, DAEBlockMaskMLP, int, int]:
    mode = acfg.adp_mode
    history: List[float] = []
    improvements: List[Tuple[int, float]] = []

    initial_snap = snapshot_arch_and_state(model)
    best_val, best_state = train_with_early_stopping(
        model, dl_train, dl_val, acfg, device, history, logger=logger
    )
    global_best_val = best_val
    global_best_snap = snapshot_arch_and_state(model, best_state)
    improvements.append((dae_total_neurons(model.width, model.depth), best_val))

    def optimize_width_at_fixed_depth(
        snap: Dict[str, Any],
        ref_best: float,
    ) -> Tuple[Dict[str, Any], float]:
        curr_snap = copy.deepcopy(snap)
        curr_best = ref_best
        curr_model = restore_arch_and_state(curr_snap, device)

        fail = 0

        while fail < acfg.trials_width:

            wider = expand_width(curr_model, acfg.ex_k, acfg.max_width, device)
            if wider is None:
                break
            curr_model = wider
            v, s = train_with_early_stopping(wider, dl_train, dl_val, acfg, device, history, logger=logger)
            if v < curr_best - acfg.delta:
                curr_best = v
                curr_snap = snapshot_arch_and_state(wider, s)
                improvements.append((dae_total_neurons(wider.width, wider.depth), v))
                fail = 0
            else:
                fail += 1
        return curr_snap, curr_best

    def optimize_depth_at_fixed_width(
        snap: Dict[str, Any],
        ref_best: float,
    ) -> Tuple[Dict[str, Any], float]:
        curr_snap = copy.deepcopy(snap)
        curr_best = ref_best
        curr_model = restore_arch_and_state(curr_snap, device)

        fail = 0

        while fail < acfg.trials_depth:

            deeper = expand_depth(curr_model, acfg.max_depth, device)

            if deeper is None:

                break

            curr_model = deeper
            v, s = train_with_early_stopping(deeper, dl_train, dl_val, acfg, device, history, logger=logger)
            if v < curr_best - acfg.delta:
                curr_best = v
                curr_snap = snapshot_arch_and_state(deeper, s)
                improvements.append((dae_total_neurons(deeper.width, deeper.depth), v))
                fail = 0
            else:
                fail += 1
        return curr_snap, curr_best

    if mode in ["width_only", "width"]:
        global_best_snap, global_best_val = optimize_width_at_fixed_depth(global_best_snap, global_best_val)
    elif mode in ["depth_only", "depth"]:
        global_best_snap, global_best_val = optimize_depth_at_fixed_width(global_best_snap, global_best_val)
    elif mode == "width_to_depth":
        global_best_snap, global_best_val = optimize_width_at_fixed_depth(global_best_snap, global_best_val)
        fail = 0
        while fail < acfg.trials_depth:
            tmp = restore_arch_and_state(global_best_snap, device)
            deeper = expand_depth(tmp, acfg.max_depth, device)
            if deeper is None:
                break
            deeper_snap = snapshot_arch_and_state(deeper)
            deeper_snap, val = optimize_width_at_fixed_depth(deeper_snap, global_best_val)
            if val < global_best_val - acfg.delta:
                global_best_val = val
                global_best_snap = deeper_snap
                fail = 0
            else:
                fail += 1
    elif mode == "depth_to_width":
        global_best_snap, global_best_val = optimize_depth_at_fixed_width(global_best_snap, global_best_val)
        fail = 0
        while fail < acfg.trials_width:
            tmp = restore_arch_and_state(global_best_snap, device)
            wider = expand_width(tmp, acfg.ex_k, acfg.max_width, device)
            if wider is None:
                break
            wider_snap = snapshot_arch_and_state(wider)
            wider_snap, val = optimize_depth_at_fixed_width(wider_snap, global_best_val)
            if val < global_best_val - acfg.delta:
                global_best_val = val
                global_best_snap = wider_snap
                fail = 0
            else:
                fail += 1
    elif mode in ["alt_width", "alt_depth"]:
        phase = "width" if mode == "alt_width" else "depth"
        sat_w = sat_d = False
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
        if logger is not None:
            logger.log_console(f"[WARN] Unknown adp_mode={mode}, skipping search.")

    if log_loss:
        plot_loss_vs_epoch(history, results_dir / "loss_vs_epoch.png", title="DAEBlockMaskMLP")
    if log_neurons and improvements:
        ns = [n for n, _ in improvements]
        vs = [v for _, v in improvements]
        plot_loss_vs_neurons(ns, vs, results_dir / "loss_vs_neurons.png", title="DAEBlockMaskMLP")

    final_model = restore_arch_and_state(global_best_snap, device)
    return global_best_val, final_model, final_model.width, final_model.depth


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="ADP Block-mask MLP DAE on CIFAR")
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
        choices=["width_only", "depth_only", "width_to_depth", "depth_to_width", "alt_width", "alt_depth", "width", "depth"],
    )
    p.add_argument("--max-epochs", type=int, default=300)
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--trials-width", type=int, default=2)
    p.add_argument("--trials-depth", type=int, default=2)
    p.add_argument("--ex-k", type=int, default=128)
    p.add_argument("--max-width", type=int, default=2048)
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--max-neurons", type=int, default=10_000_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--block-frac", type=float, default=0.15)

    p.add_argument("--results-dir", type=str, default="results_adp_dae_blockmask_mlp")
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

    model = DAEBlockMaskMLP(in_channels=3, img_size=32, width=args.width, depth=args.depth).to(device)

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
        block_frac=args.block_frac,
    )

    results_dir = Path(args.results_dir)
    logger = ContinuousLogger(results_dir, "dae_blockmask_mlp", args.adp_mode)

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
