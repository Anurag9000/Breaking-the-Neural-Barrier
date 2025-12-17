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

from .dae_unet_conv_stl import DAEUNetConv, dae_total_neurons


@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"
    delta: float = 1e-3
    patience: int = 20
    trials_width: int = 2
    trials_depth: int = 2
    ex_k: int = 16
    max_width: int = 512
    max_depth: int = 8
    max_neurons: int = 5_000_000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    max_epochs: int = 100_000_000
    noise_std: float = 0.1


def _resize_tensor(to_shape: torch.Size, src: torch.Tensor) -> torch.Tensor:
    tgt = torch.zeros(to_shape, device=src.device, dtype=src.dtype)
    common = tuple(min(a, b) for a, b in zip(to_shape, src.shape))
    slices = tuple(slice(0, c) for c in common)
    tgt[slices] = src[slices]
    return tgt


def _merge_state(
    new_state: Dict[str, torch.Tensor],
    old_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    merged: Dict[str, torch.Tensor] = {}
    for k, v in new_state.items():
        if k in old_state:
            ov = old_state[k]
            merged[k] = ov if ov.shape == v.shape else _resize_tensor(v.shape, ov)
        else:
            merged[k] = v
    return merged


def rebuild_model(model: DAEUNetConv, width: int, depth: int, device: torch.device) -> DAEUNetConv:
    new_model = DAEUNetConv(
        in_channels=model.in_channels,
        width=width,
        depth=depth,
    ).to(device)
    merged = _merge_state(new_model.state_dict(), model.state_dict())
    new_model.load_state_dict(merged, strict=False)
    return new_model


def expand_width(
    model: DAEUNetConv,
    ex_k: int,
    max_width: int,
    device: torch.device,
) -> Optional[DAEUNetConv]:
    new_w = min(max_width, model.width + ex_k)
    if new_w == model.width:
        return None
    return rebuild_model(model, new_w, model.depth, device)


def expand_depth(
    model: DAEUNetConv,
    max_depth: int,
    device: torch.device,
) -> Optional[DAEUNetConv]:
    if model.depth >= max_depth:
        return None
    return rebuild_model(model, model.width, model.depth + 1, device)


def snapshot_arch_and_state(model: DAEUNetConv, state: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
    st = state if state is not None else model.state_dict()
    return {
        "width": model.width,
        "depth": model.depth,
        "in_channels": model.in_channels,
        "state": copy.deepcopy(st),
    }


def restore_arch_and_state(snap: Dict[str, Any], device: torch.device) -> DAEUNetConv:
    mdl = DAEUNetConv(
        in_channels=snap.get("in_channels", 3),
        width=snap["width"],
        depth=snap["depth"],
    ).to(device)
    mdl.load_state_dict(snap["state"], strict=False)
    return mdl


def add_gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return x
    return x + torch.randn_like(x) * sigma


def train_with_early_stopping(
    model: DAEUNetConv,
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
            loss = mse(xb_rec, xb)
            loss.backward()
            if acfg.grad_clip is not None and acfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), acfg.grad_clip)
            opt.step()

            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        train_loss = total / max(n, 1)

        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for xb, _ in dl_val:
                xb = xb.to(device, non_blocking=True)
                xb_noisy = add_gaussian_noise(xb, acfg.noise_std)
                xb_rec, _ = model(xb_noisy)
                total += float(mse(xb_rec, xb).item())
                n += xb.size(0)
        val_loss = total / max(n, 1)
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
        elif verbose:
            print(msg)

        if es_counter >= acfg.patience:
            if logger:
                logger.log_console(f"  Early stopping at epoch {epoch}")
            elif verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    return best_val, best_state


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
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
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

    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dl_train, dl_val


def adp_search(
    model: DAEUNetConv,
    dl_train: DataLoader,
    dl_val: DataLoader,
    acfg: ADPConfig,
    device: torch.device,
    logger: ContinuousLogger,
    results_dir: Path,
    log_loss: bool,
    log_neurons: bool,
) -> Tuple[float, DAEUNetConv, int, int]:
    results_dir.mkdir(parents=True, exist_ok=True)
    val_history: List[float] = []
    improvements: List[Tuple[int, float]] = []

    base_val, base_state = train_with_early_stopping(
        model, dl_train, dl_val, acfg, device, val_history, logger=logger, verbose=True
    )
    best_snap = snapshot_arch_and_state(model, base_state)
    global_best_val = base_val
    global_best_snap = copy.deepcopy(best_snap)
    best_w, best_d = model.width, model.depth
    improvements.append((dae_total_neurons(best_w, best_d), global_best_val))

    def optimize_width_at_fixed_depth(
        start_snap: Dict[str, Any],
        current_best_val: float,
    ) -> Tuple[Dict[str, Any], float]:
        nonlocal improvements
        snap = copy.deepcopy(start_snap)
        fail = 0
        while fail < acfg.trials_width:
            curr_model = restore_arch_and_state(snap, device)
            wider = expand_width(curr_model, acfg.ex_k, acfg.max_width, device)
            if wider is None or dae_total_neurons(wider.width, wider.depth) > acfg.max_neurons:
                break
            logger.log_console(
                f"[WIDTH OPT] Trying width={wider.width}, depth={wider.depth}, "
                f"neurons={dae_total_neurons(wider.width, wider.depth)}"
            )
            val, state = train_with_early_stopping(
                wider, dl_train, dl_val, acfg, device, val_history, logger=logger, verbose=False
            )
            if val + acfg.delta < current_best_val:
                current_best_val = val
                snap = snapshot_arch_and_state(wider, state)
                improvements.append((dae_total_neurons(wider.width, wider.depth), val))
                logger.log_console(
                    f"[WIDTH OPT] ✓ IMPROVEMENT: width={wider.width}, depth={wider.depth}, val={val:.6f}"
                )
                fail = 0
            else:
                fail += 1
                logger.log_console(
                    f"[WIDTH OPT] ✗ No improvement: width={wider.width}, depth={wider.depth}, val={val:.6f}"
                )
        return snap, current_best_val

    def optimize_depth_at_fixed_width(
        start_snap: Dict[str, Any],
        current_best_val: float,
    ) -> Tuple[Dict[str, Any], float]:
        nonlocal improvements
        snap = copy.deepcopy(start_snap)
        fail = 0
        while fail < acfg.trials_depth:
            curr_model = restore_arch_and_state(snap, device)
            deeper = expand_depth(curr_model, acfg.max_depth, device)
            if deeper is None or dae_total_neurons(deeper.width, deeper.depth) > acfg.max_neurons:
                break
            logger.log_console(
                f"[DEPTH OPT] Trying width={deeper.width}, depth={deeper.depth}, "
                f"neurons={dae_total_neurons(deeper.width, deeper.depth)}"
            )
            val, state = train_with_early_stopping(
                deeper, dl_train, dl_val, acfg, device, val_history, logger=logger, verbose=False
            )
            if val + acfg.delta < current_best_val:
                current_best_val = val
                snap = snapshot_arch_and_state(deeper, state)
                improvements.append((dae_total_neurons(deeper.width, deeper.depth), val))
                logger.log_console(
                    f"[DEPTH OPT] ✓ IMPROVEMENT: width={deeper.width}, depth={deeper.depth}, val={val:.6f}"
                )
                fail = 0
            else:
                fail += 1
                logger.log_console(
                    f"[DEPTH OPT] ✗ No improvement: width={deeper.width}, depth={deeper.depth}, val={val:.6f}"
                )
        return snap, current_best_val

    mode = acfg.adp_mode
    logger.log_console(f"[ADP] Starting search mode={mode}")

    if mode == "width_only":
        global_best_snap, global_best_val = optimize_width_at_fixed_depth(global_best_snap, global_best_val)
    elif mode == "depth_only":
        global_best_snap, global_best_val = optimize_depth_at_fixed_width(global_best_snap, global_best_val)
    elif mode == "width_to_depth":
        global_best_snap, global_best_val = optimize_width_at_fixed_depth(global_best_snap, global_best_val)
        global_best_snap, global_best_val = optimize_depth_at_fixed_width(global_best_snap, global_best_val)
    elif mode == "depth_to_width":
        global_best_snap, global_best_val = optimize_depth_at_fixed_width(global_best_snap, global_best_val)
        global_best_snap, global_best_val = optimize_width_at_fixed_depth(global_best_snap, global_best_val)
    elif mode == "alt_width":
        turn_width = True
        while True:
            prev_val = global_best_val
            if turn_width:
                global_best_snap, global_best_val = optimize_width_at_fixed_depth(global_best_snap, global_best_val)
            else:
                global_best_snap, global_best_val = optimize_depth_at_fixed_width(global_best_snap, global_best_val)
            if abs(global_best_val - prev_val) < acfg.delta:
                break
            turn_width = not turn_width
    elif mode == "alt_depth":
        turn_width = False
        while True:
            prev_val = global_best_val
            if turn_width:
                global_best_snap, global_best_val = optimize_width_at_fixed_depth(global_best_snap, global_best_val)
            else:
                global_best_snap, global_best_val = optimize_depth_at_fixed_width(global_best_snap, global_best_val)
            if abs(global_best_val - prev_val) < acfg.delta:
                break
            turn_width = not turn_width
    else:
        raise ValueError(f"Unknown adp_mode: {mode}")

    best_model = restore_arch_and_state(global_best_snap, device)
    best_w, best_d = best_model.width, best_model.depth

    if log_loss:
        plot_loss_vs_epoch(val_history, results_dir / "loss_vs_epoch.png")
    if log_neurons and improvements:
        ns = [n for n, _ in improvements]
        vs = [v for _, v in improvements]
        plot_loss_vs_neurons(ns, vs, results_dir / "loss_vs_neurons.png")

    return global_best_val, best_model, best_w, best_d


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--results-dir", type=str, default="results_adp_dae_unet_conv_cifar")
    p.add_argument("--adp-mode", type=str, default="width_to_depth")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--ex-k", type=int, default=16)
    p.add_argument("--trials-width", type=int, default=2)
    p.add_argument("--trials-depth", type=int, default=2)
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--max-width", type=int, default=512)
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--max-neurons", type=int, default=5_000_000)
    p.add_argument("--max-epochs", type=int, default=100_000_000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--noise-std", type=float, default=0.1)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--plot-loss", action="store_true")
    p.add_argument("--plot-neurons", action="store_true")

    args = p.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    logger = ContinuousLogger(
        results_dir / "training_log.txt",
        console_prefix="[ADP DAE U-Net]",
    )
    logger.log_console(f"Initialized ADP search (mode={args.adp_mode})")

    dl_train, dl_val = make_loaders(
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
    )

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
        max_epochs=args.max_epochs,
        noise_std=args.noise_std,
    )

    base_model = DAEUNetConv(in_channels=3, width=args.width, depth=args.depth).to(device)
    logger.log_console(
        f"Base model: width={base_model.width}, depth={base_model.depth}, "
        f"neurons={dae_total_neurons(base_model.width, base_model.depth)}"
    )

    best_val, best_model, best_w, best_d = adp_search(
        base_model,
        dl_train,
        dl_val,
        acfg,
        device,
        logger,
        results_dir,
        log_loss=args.plot_loss,
        log_neurons=args.plot_neurons,
    )

    logger.log_console(
        f"[ADP DONE] Best val={best_val:.6f} at width={best_w}, depth={best_d}, "
        f"neurons={dae_total_neurons(best_w, best_d)}"
    )

    torch.save(
        {
            "model": best_model.state_dict(),
            "width": best_w,
            "depth": best_d,
            "best_val": best_val,
            "config": vars(args),
        },
        results_dir / "best_adp_model.pt",
    )


if __name__ == "__main__":
    main()

