import copy
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

sys.path.append(str(Path(__file__).resolve().parents[4]))
from utils.adp_logging import ContinuousLogger  # type: ignore
from utils.adp_plot import plot_loss_vs_epoch, plot_loss_vs_neurons  # type: ignore

from .dae_metric_conv_sup_stl import SupDAEGaussianMetricConv, sup_metric_dae_total_neurons


@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"
    delta: float = 1e-3
    patience: int = 20
    trials_width: int = 2
    trials_depth: int = 2
    ex_k: int = 16
    max_width: int = 512
    max_depth: int = 5
    max_neurons: int = 5_000_000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    max_epochs: int = 300
    noise_std: float = 0.1
    lambda_recon: float = 1.0
    lambda_triplet: float = 1.0
    proj_dim: int = 128
    margin: float = 1.0


def add_gaussian_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return x
    return x + torch.randn_like(x) * sigma


def _resize_tensor(to_shape: torch.Size, src: torch.Tensor) -> torch.Tensor:
    tgt = torch.zeros(to_shape, device=src.device, dtype=src.dtype)
    common = tuple(min(a, b) for a, b in zip(to_shape, src.shape))
    slices = tuple(slice(0, c) for c in common)
    tgt[slices] = src[slices]
    return tgt


def _merge_state(
    new_state: Dict[str, torch.Tensor], old_state: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    merged: Dict[str, torch.Tensor] = {}
    for k, v in new_state.items():
        if k in old_state:
            ov = old_state[k]
            merged[k] = ov if ov.shape == v.shape else _resize_tensor(v.shape, ov)
        else:
            merged[k] = v
    return merged


def rebuild_model(
    model: SupDAEGaussianMetricConv,
    proj_dim: int,
    width: int,
    depth: int,
    device: torch.device,
) -> SupDAEGaussianMetricConv:
    new_model = SupDAEGaussianMetricConv(
        proj_dim=proj_dim,
        in_channels=model.in_channels,
        width=width,
        depth=depth,
        pool_after=list(model.dae.pool_after),
    ).to(device)
    merged = _merge_state(new_model.state_dict(), model.state_dict())
    new_model.load_state_dict(merged, strict=False)
    return new_model


def expand_width(
    model: SupDAEGaussianMetricConv,
    proj_dim: int,
    ex_k: int,
    max_width: int,
    device: torch.device,
) -> Optional[SupDAEGaussianMetricConv]:
    new_w = min(max_width, model.width + ex_k)
    if new_w == model.width:
        return None
    return rebuild_model(model, proj_dim, new_w, model.depth, device)


def expand_depth(
    model: SupDAEGaussianMetricConv,
    proj_dim: int,
    max_depth: int,
    device: torch.device,
) -> Optional[SupDAEGaussianMetricConv]:
    if model.depth >= max_depth:
        return None
    return rebuild_model(model, proj_dim, model.width, model.depth + 1, device)


def snapshot_arch_and_state(
    model: SupDAEGaussianMetricConv, state: Optional[Dict[str, torch.Tensor]] = None
) -> Dict[str, Any]:
    st = state if state is not None else model.state_dict()
    return {
        "in_channels": model.in_channels,
        "width": model.width,
        "depth": model.depth,
        "proj_dim": model.proj_dim,
        "pool_after": list(model.dae.pool_after),
        "state": copy.deepcopy(st),
    }


def restore_arch_and_state(snap: Dict[str, Any], device: torch.device) -> SupDAEGaussianMetricConv:
    mdl = SupDAEGaussianMetricConv(
        proj_dim=snap["proj_dim"],
        in_channels=snap.get("in_channels", 3),
        width=snap["width"],
        depth=snap["depth"],
        pool_after=list(snap.get("pool_after", [])),
    ).to(device)
    mdl.load_state_dict(snap["state"], strict=False)
    return mdl


def _make_triplets(z: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Simple in-batch triplet sampling: for each anchor, choose one positive and
    one negative when possible.
    """
    device = z.device
    anchors: List[int] = []
    positives: List[int] = []
    negatives: List[int] = []
    for i in range(z.size(0)):
        mask_pos = (y == y[i]).nonzero(as_tuple=False).view(-1)
        mask_neg = (y != y[i]).nonzero(as_tuple=False).view(-1)
        mask_pos = mask_pos[mask_pos != i]
        if mask_pos.numel() == 0 or mask_neg.numel() == 0:
            continue
        anchors.append(i)
        positives.append(mask_pos[0].item())
        negatives.append(mask_neg[0].item())
    if not anchors:
        return z[:0], z[:0], z[:0]
    a = z[torch.tensor(anchors, device=device)]
    p = z[torch.tensor(positives, device=device)]
    n = z[torch.tensor(negatives, device=device)]
    return a, p, n


def train_with_early_stopping(
    model: SupDAEGaussianMetricConv,
    dl_train: DataLoader,
    dl_val: DataLoader,
    acfg: ADPConfig,
    device: torch.device,
    history: List[float],
    logger: Optional[ContinuousLogger] = None,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    mse = nn.MSELoss()
    triplet = nn.TripletMarginLoss(margin=acfg.margin)
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    es_counter = 0

    for epoch in range(1, acfg.max_epochs + 1):
        model.train()
        total, n = 0.0, 0
        for xb, yb in dl_train:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            xb_noisy = add_gaussian_noise(xb, acfg.noise_std)

            opt.zero_grad(set_to_none=True)
            xb_rec, z = model(xb_noisy)
            loss_recon = mse(xb_rec, xb)
            a, p, n_vec = _make_triplets(z, yb)
            if a.numel() > 0:
                loss_trip = triplet(a, p, n_vec)
                loss = acfg.lambda_recon * loss_recon + acfg.lambda_triplet * loss_trip
            else:
                loss = acfg.lambda_recon * loss_recon
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
            for xb, yb in dl_val:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                xb_noisy = add_gaussian_noise(xb, acfg.noise_std)
                xb_rec, z = model(xb_noisy)
                loss_recon = mse(xb_rec, xb)
                a, p, n_vec = _make_triplets(z, yb)
                if a.numel() > 0:
                    loss_trip = triplet(a, p, n_vec)
                    loss = acfg.lambda_recon * loss_recon + acfg.lambda_triplet * loss_trip
                else:
                    loss = acfg.lambda_recon * loss_recon
                total += float(loss.item()) * xb.size(0)
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
        else:
            print(msg)

        if es_counter >= acfg.patience:
            if logger:
                logger.log_console(f"  Early stopping at epoch {epoch}")
            else:
                print(f"  Early stopping at epoch {epoch}")
            break

    return best_val, best_state


def make_loaders(
    dataset: str,
    data_root: str,
    batch_size: int,
    val_split: float,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, int]:
    ds_name = dataset.lower()
    if ds_name == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        ds_cls = datasets.CIFAR100
        num_classes = 100
    else:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        ds_cls = datasets.CIFAR10
        num_classes = 10

    tf_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    tf_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    full_train = ds_cls(root=data_root, train=True, transform=tf_train, download=True)
    n_val = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val
    g = torch.Generator().manual_seed(1337)
    ds_train, ds_val = random_split(full_train, [n_train, n_val], generator=g)

    dl_train = DataLoader(
        ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False
    )
    dl_val = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
    )
    return dl_train, dl_val, num_classes


def adp_search(
    dataset: str,
    data_root: str,
    log_dir: Path,
    acfg: ADPConfig,
    device: torch.device,
    logger: ContinuousLogger,
    val_split: float = 0.1,
) -> Tuple[float, SupDAEGaussianMetricConv]:
    dl_train, dl_val, num_classes = make_loaders(
        dataset=dataset,
        data_root=data_root,
        batch_size=logger.batch_size,
        val_split=val_split,
        num_workers=logger.num_workers,
    )

    history: List[float] = []
    start_model = SupDAEGaussianMetricConv(
        proj_dim=acfg.proj_dim,
        in_channels=3,
        width=logger.width,
        depth=logger.depth,
        pool_after=[2],
    ).to(device)

    neurons = sup_metric_dae_total_neurons(logger.width, logger.depth, acfg.proj_dim)
    logger.log_architecture(logger.width, logger.depth, neurons)

    best_val, best_state = train_with_early_stopping(
        start_model, dl_train, dl_val, acfg, device, history, logger
    )
    global_best_snap = snapshot_arch_and_state(start_model, best_state)
    global_best_val = best_val

    def try_width_only(
        snap: Dict[str, Any], best_val_so_far: float
    ) -> Tuple[Dict[str, Any], float]:
        fail = 0
        width_history: List[Tuple[int, float]] = []
        curr_snap = snap
        while fail < acfg.trials_width:
            model = restore_arch_and_state(curr_snap, device)
            widened = expand_width(model, acfg.proj_dim, acfg.ex_k, acfg.max_width, device)
            if widened is None:
                break
            w = widened.width
            if (
                sup_metric_dae_total_neurons(w, widened.depth, acfg.proj_dim)
                > acfg.max_neurons
            ):
                break
            v, s = train_with_early_stopping(
                widened, dl_train, dl_val, acfg, device, history, logger
            )
            width_history.append((w, v))
            if v < best_val_so_far - acfg.delta:
                best_val_so_far = v
                curr_snap = snapshot_arch_and_state(widened, s)
                fail = 0
            else:
                fail += 1
        if width_history:
            logger.log_width_search(width_history)
        return curr_snap, best_val_so_far

    def try_depth_only(
        snap: Dict[str, Any], best_val_so_far: float
    ) -> Tuple[Dict[str, Any], float]:
        fail = 0
        depth_history: List[Tuple[int, float]] = []
        curr_snap = snap
        while fail < acfg.trials_depth:
            model = restore_arch_and_state(curr_snap, device)
            deeper = expand_depth(model, acfg.proj_dim, acfg.max_depth, device)
            if deeper is None:
                break
            d = deeper.depth
            if (
                sup_metric_dae_total_neurons(deeper.width, d, acfg.proj_dim)
                > acfg.max_neurons
            ):
                break
            v, s = train_with_early_stopping(
                deeper, dl_train, dl_val, acfg, device, history, logger
            )
            depth_history.append((d, v))
            if v < best_val_so_far - acfg.delta:
                best_val_so_far = v
                curr_snap = snapshot_arch_and_state(deeper, s)
                fail = 0
            else:
                fail += 1
        if depth_history:
            logger.log_depth_search(depth_history)
        return curr_snap, best_val_so_far

    mode = acfg.adp_mode.lower()
    if mode == "width_only":
        global_best_snap, global_best_val = try_width_only(global_best_snap, global_best_val)
    elif mode == "depth_only":
        global_best_snap, global_best_val = try_depth_only(global_best_snap, global_best_val)
    elif mode == "width_to_depth":
        global_best_snap, global_best_val = try_width_only(global_best_snap, global_best_val)
        global_best_snap, global_best_val = try_depth_only(global_best_snap, global_best_val)
    elif mode == "depth_to_width":
        global_best_snap, global_best_val = try_depth_only(global_best_snap, global_best_val)
        global_best_snap, global_best_val = try_width_only(global_best_snap, global_best_val)
    elif mode == "alt_width":
        for _ in range(max(acfg.trials_width, acfg.trials_depth)):
            global_best_snap, global_best_val = try_width_only(global_best_snap, global_best_val)
            global_best_snap, global_best_val = try_depth_only(global_best_snap, global_best_val)
    elif mode == "alt_depth":
        for _ in range(max(acfg.trials_width, acfg.trials_depth)):
            global_best_snap, global_best_val = try_depth_only(global_best_snap, global_best_val)
            global_best_snap, global_best_val = try_width_only(global_best_snap, global_best_val)
    else:
        raise ValueError(f"Unknown adp_mode={acfg.adp_mode}")

    final_model = restore_arch_and_state(global_best_snap, device)
    return global_best_val, final_model


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--adp-mode", type=str, default="width_to_depth")
    parser.add_argument("--delta", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--trials-width", type=int, default=2)
    parser.add_argument("--trials-depth", type=int, default=2)
    parser.add_argument("--ex-k", type=int, default=16)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--max-width", type=int, default=512)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--max-neurons", type=int, default=5_000_000)
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--lambda-triplet", type=float, default=1.0)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--results-dir", type=str, default="results_dae_triplet_conv_sup_adp")
    parser.add_argument("--plot-loss", action="store_true")
    parser.add_argument("--plot-neurons", action="store_true")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    cfg = ADPConfig(
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
        lambda_triplet=args.lambda_triplet,
        proj_dim=args.proj_dim,
        margin=args.margin,
    )

    logger = ContinuousLogger(
        experiment_name="dae_triplet_conv_sup",
        mode=args.adp_mode,
        results_dir=results_dir,
    )
    logger.batch_size = args.batch_size
    logger.num_workers = args.num_workers
    logger.width = args.width
    logger.depth = args.depth

    best_val, _ = adp_search(
        dataset=args.dataset,
        data_root=args.data_root,
        log_dir=results_dir,
        acfg=cfg,
        device=device,
        logger=logger,
        val_split=args.val_split,
    )

    logger.log_final(best_val)

    if args.plot_loss or args.plot_neurons:
        stats_csv = results_dir / "training_stats.csv"
        if stats_csv.exists():
            plot_loss_vs_epoch(stats_csv, results_dir / "loss_vs_epoch.png")
            plot_loss_vs_neurons(stats_csv, results_dir / "loss_vs_neurons.png")
