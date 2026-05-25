"""
ADP runner for a ResNet-style CIFAR backbone (ADPResNet).

Supports ADP modes:
  - width_only
  - depth_only
  - width_to_depth
  - depth_to_width
  - alt_width
  - alt_depth

ADP behaviour (per architecture):
  - Train current (width, depth) once with early stopping.
  - When validation loss stops improving (ES hits patience), immediately try
    expanded architectures (wider/deeper) until no improvement for a fixed
    number of expansion trials.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split, Subset
import torchvision

from CNN.ADP_ResNet.adp_resnet_backbone import ADPResNet, ADPResNetConfig, make_adp_resnet, estimate_neurons
from utils.cnn_data import make_cifar_transforms
from utils.adp_logging import ContinuousLogger
from utils.adp_plot import plot_loss_vs_epoch, plot_loss_vs_neurons
from utils.adp_state import merge_state_preserving_init
from utils.cutmix import cutmix_batch


@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"
    delta: float = 1e-3
    patience: int = 20
    trials_width: int = 2
    trials_depth: int = 2
    ex_k: int = 4
    max_width: int = 128
    max_depth: int = 4
    max_neurons: int = 5_000_000
    lr: float = 1e-3
    min_lr: float = 1e-5
    weight_decay: float = 5e-4
    grad_clip: float = 1.0
    max_epochs: int = 300
    dropout: float = 0.0
    cutmix_p: float = 0.0
    cutmix_alpha: float = 1.0


def _resize_tensor(to_shape: torch.Size, src: torch.Tensor) -> torch.Tensor:
    tgt = torch.zeros(to_shape, device=src.device, dtype=src.dtype)
    common = tuple(min(a, b) for a, b in zip(to_shape, src.shape))
    slices = tuple(slice(0, c) for c in common)
    tgt[slices] = src[slices]
    return tgt


def _merge_state(new_state: Dict[str, torch.Tensor], old_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return merge_state_preserving_init(new_state, old_state)


def rebuild_model(model: ADPResNet, width: int, depth: int, device: torch.device) -> ADPResNet:
    cfg = ADPResNetConfig(
        input_channels=model.input_channels,
        num_classes=model.num_classes,
        width=width,
        depth=depth,
        dropout=model.dropout,
    )
    new_model = ADPResNet(cfg).to(device)
    merged = _merge_state(new_model.state_dict(), model.state_dict())
    new_model.load_state_dict(merged, strict=False)
    return new_model


def expand_width(model: ADPResNet, ex_k: int, max_width: int, device: torch.device) -> Optional[ADPResNet]:
    new_w = min(model.width + ex_k, max_width)
    if new_w == model.width and ex_k > 0:
        return None
    if new_w > max_width:
        return None
    return rebuild_model(model, new_w, model.depth, device)


def expand_depth(model: ADPResNet, max_depth: int, device: torch.device) -> Optional[ADPResNet]:
    new_d = min(model.depth + 1, max_depth)
    if new_d == model.depth:
        return None
    return rebuild_model(model, model.width, new_d, device)


def snapshot_arch_and_state(model: ADPResNet, state_dict: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, Any]:
    state = state_dict if state_dict is not None else model.state_dict()
    return {
        "width": model.width,
        "depth": model.depth,
        "input_channels": model.input_channels,
        "num_classes": model.num_classes,
        "dropout": model.dropout,
        "state": copy.deepcopy(state),
    }


def restore_arch_and_state(snap: Dict[str, Any], device: torch.device) -> ADPResNet:
    cfg = ADPResNetConfig(
        input_channels=snap.get("input_channels", 3),
        num_classes=snap.get("num_classes", 10),
        width=snap["width"],
        depth=snap["depth"],
        dropout=snap.get("dropout", 0.0),
    )
    model = ADPResNet(cfg).to(device)
    model.load_state_dict(snap["state"], strict=False)
    return model


def make_loaders(
    dataset: str,
    data_root: str,
    batch_size: int,
    val_split: float,
    num_workers: int,
    use_augment: bool,
    num_classes_limit: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, int]:
    train_tfms, eval_tfms = make_cifar_transforms(dataset, use_augment=use_augment)

    ds_name = dataset.lower()
    if ds_name == "cifar100":
        full_train = torchvision.datasets.CIFAR100(root=data_root, train=True, transform=train_tfms, download=True)
        full_train_eval = torchvision.datasets.CIFAR100(root=data_root, train=True, transform=eval_tfms, download=True)
        base_num_classes = 100
    else:
        full_train = torchvision.datasets.CIFAR10(root=data_root, train=True, transform=train_tfms, download=True)
        full_train_eval = torchvision.datasets.CIFAR10(root=data_root, train=True, transform=eval_tfms, download=True)
        base_num_classes = 10

    num_classes = base_num_classes

    # Optional restriction to first N classes (labels 0..N-1)
    if num_classes_limit is not None:
        n_lim = max(2, min(num_classes_limit, base_num_classes))
        allowed = set(range(n_lim))
        targets = getattr(full_train, "targets", None)
        if targets is None and hasattr(full_train, "labels"):
            targets = full_train.labels
        if targets is not None:
            indices = [i for i, y in enumerate(targets) if y in allowed]
            full_train = Subset(full_train, indices)
            full_train_eval = Subset(full_train_eval, indices)
            num_classes = n_lim

    n_total = len(full_train)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(n_total, generator=generator).tolist()
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    train_ds = Subset(full_train, train_idx)
    val_ds = Subset(full_train_eval, val_idx)

    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dl_train, dl_val, num_classes


def train_with_early_stopping(
    model: ADPResNet,
    dl_train: DataLoader,
    dl_val: DataLoader,
    acfg: ADPConfig,
    device: torch.device,
    history: List[float],
    logger: Optional[ContinuousLogger] = None,
    verbose: bool = True,
) -> Tuple[float, Dict[str, torch.Tensor]]:
    opt = AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    scheduler = CosineAnnealingLR(opt, T_max=acfg.max_epochs, eta_min=acfg.min_lr)
    criterion = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    es_counter = 0

    gen = torch.Generator(device=device)

    for epoch in range(1, acfg.max_epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for x, y in dl_train:
            x, y = x.to(device), y.to(device)
            x_mix, targets = cutmix_batch(
                x,
                y,
                alpha=acfg.cutmix_alpha,
                p=acfg.cutmix_p,
                generator=gen,
            )
            opt.zero_grad(set_to_none=True)
            logits = model(x_mix)
            if targets is None:
                loss = criterion(logits, y)
            else:
                y1, y2, lam = targets
                loss = lam * criterion(logits, y1) + (1.0 - lam) * criterion(logits, y2)
            loss.backward()
            if acfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), acfg.grad_clip)
            opt.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)
        train_loss = running / max(1, seen)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for x, y in dl_val:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                n_val += x.size(0)
        val_loss = val_loss / max(1, n_val)

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
            f"train={train_loss:.4f} | val={val_loss:.4f} | "
            f"best={best_val:.4f} | ES={es_counter}/{acfg.patience}"
        )
        if logger:
            logger.log_console(msg)
            logger.log_epoch_stats(
                {
                    "epoch": len(history),
                    "width": getattr(model, "width", 0),
                    "depth": getattr(model, "depth", 0),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val": best_val,
                    "es_counter": es_counter,
                }
            )
        elif verbose:
            print(msg)

        if es_counter >= acfg.patience:
            break

        scheduler.step()

    return best_val, best_state


def adp_search(
    model: ADPResNet,
    dl_train: DataLoader,
    dl_val: DataLoader,
    acfg: ADPConfig,
    device: torch.device,
    logger: ContinuousLogger,
    results_dir: Path,
    log_loss: bool,
    log_neurons: bool,
) -> Tuple[float, ADPResNet, int, int]:
    results_dir.mkdir(parents=True, exist_ok=True)
    val_history: List[float] = []
    improvements: List[Tuple[int, float]] = []

    # Initial training at starting width/depth
    logger.log_console("[INITIAL TRAINING]")
    best_val, best_state = train_with_early_stopping(model, dl_train, dl_val, acfg, device, val_history, logger)
    model.load_state_dict(best_state)
    best_snap = snapshot_arch_and_state(model, best_state)
    improvements.append((estimate_neurons(model.width, model.depth), best_val))

    def can_widen(m: ADPResNet) -> bool:
        return (m.width + acfg.ex_k <= acfg.max_width) and (
            estimate_neurons(m.width + acfg.ex_k, m.depth) <= acfg.max_neurons
        )

    def can_deepen(m: ADPResNet) -> bool:
        return (m.depth + 1 <= acfg.max_depth) and (
            estimate_neurons(m.width, m.depth + 1) <= acfg.max_neurons
        )

    def optimize_width_at_fixed_depth(
        curr_snap: Dict[str, Any],
        start_val: float,
    ) -> Tuple[Dict[str, Any], float]:
        """Repeatedly widen while validation loss keeps improving."""
        local_best_val = start_val
        local_best_snap = curr_snap
        curr_model = restore_arch_and_state(curr_snap, device)

        failure_count = 0
        while failure_count < acfg.trials_width:
            if not can_widen(curr_model):
                break
            widened = expand_width(curr_model, acfg.ex_k, acfg.max_width, device)
            if widened is None:
                break
            curr_model = widened
            v, s = train_with_early_stopping(curr_model, dl_train, dl_val, acfg, device, val_history, logger)
            if v < local_best_val - acfg.delta:
                local_best_val = v
                local_best_snap = snapshot_arch_and_state(curr_model, s)
                failure_count = 0
                improvements.append((estimate_neurons(curr_model.width, curr_model.depth), v))
                logger.log_console(f"[WIDTH OPT] ✓ IMPROVEMENT: New best: {v:.6f}")
                if log_loss:
                    plot_loss_vs_epoch(val_history, results_dir / "loss_vs_epoch.png", title=f"ADPResNet ({acfg.adp_mode})")
                if log_neurons:
                    n_list = [n for n, _ in improvements]
                    l_list = [vv for _, vv in improvements]
                    plot_loss_vs_neurons(n_list, l_list, results_dir / "loss_vs_neurons.png", title="ADPResNet")
            else:
                failure_count += 1
                logger.log_console("[WIDTH OPT] ✗ No improvement")

        return local_best_snap, local_best_val

    def optimize_depth_at_fixed_width(
        curr_snap: Dict[str, Any],
        start_val: float,
    ) -> Tuple[Dict[str, Any], float]:
        """Repeatedly deepen while validation loss keeps improving."""
        local_best_val = start_val
        local_best_snap = curr_snap
        curr_model = restore_arch_and_state(curr_snap, device)

        failure_count = 0
        while failure_count < acfg.trials_depth:
            if not can_deepen(curr_model):
                break
            deepened = expand_depth(curr_model, acfg.max_depth, device)
            if deepened is None:
                break
            curr_model = deepened
            v, s = train_with_early_stopping(curr_model, dl_train, dl_val, acfg, device, val_history, logger)
            if v < local_best_val - acfg.delta:
                local_best_val = v
                local_best_snap = snapshot_arch_and_state(curr_model, s)
                failure_count = 0
                improvements.append((estimate_neurons(curr_model.width, curr_model.depth), v))
                logger.log_console(f"[DEPTH OPT] ✓ IMPROVEMENT: New best: {v:.6f}")
                if log_loss:
                    plot_loss_vs_epoch(val_history, results_dir / "loss_vs_epoch.png", title=f"ADPResNet ({acfg.adp_mode})")
                if log_neurons:
                    n_list = [n for n, _ in improvements]
                    l_list = [vv for _, vv in improvements]
                    plot_loss_vs_neurons(n_list, l_list, results_dir / "loss_vs_neurons.png", title="ADPResNet")
            else:
                failure_count += 1
                logger.log_console("[DEPTH OPT] ✗ No improvement")

        return local_best_snap, local_best_val

    global_best_val = best_val
    global_best_snap = best_snap

    mode = acfg.adp_mode
    if mode in ["width_only", "width"]:
        global_best_snap, global_best_val = optimize_width_at_fixed_depth(global_best_snap, global_best_val)
    elif mode in ["depth_only", "depth"]:
        global_best_snap, global_best_val = optimize_depth_at_fixed_width(global_best_snap, global_best_val)
    elif mode == "width_to_depth":
        # Width then depth refinement after each width step
        global_best_snap, global_best_val = optimize_width_at_fixed_depth(global_best_snap, global_best_val)
        fail = 0
        while fail < acfg.trials_depth:
            tmp_model = restore_arch_and_state(global_best_snap, device)
            if not can_deepen(tmp_model):
                break
            deeper = expand_depth(tmp_model, acfg.max_depth, device)
            if deeper is None:
                break
            deeper_snap = snapshot_arch_and_state(deeper, deeper.state_dict())
            deeper_snap, val = optimize_width_at_fixed_depth(deeper_snap, global_best_val)
            if val < global_best_val - acfg.delta:
                global_best_val = val
                global_best_snap = deeper_snap
                fail = 0
            else:
                fail += 1
    elif mode == "depth_to_width":
        # Depth then width refinement after each depth step
        global_best_snap, global_best_val = optimize_depth_at_fixed_width(global_best_snap, global_best_val)
        fail = 0
        while fail < acfg.trials_width:
            tmp_model = restore_arch_and_state(global_best_snap, device)
            if not can_widen(tmp_model):
                break
            wider = expand_width(tmp_model, acfg.ex_k, acfg.max_width, device)
            if wider is None:
                break
            wider_snap = snapshot_arch_and_state(wider, wider.state_dict())
            wider_snap, val = optimize_depth_at_fixed_width(wider_snap, global_best_val)
            if val < global_best_val - acfg.delta:
                global_best_val = val
                global_best_snap = wider_snap
                fail = 0
            else:
                fail += 1
    elif mode in ["alt_width", "alt_depth"]:
        # Alternate between width and depth optimisation until both saturate.
        phase = "width" if mode == "alt_width" else "depth"
        sat_w, sat_d = False, False
        while not (sat_w and sat_d):
            improved = False
            if phase == "width":
                new_snap, val = optimize_width_at_fixed_depth(global_best_snap, global_best_val)
                if val < global_best_val - acfg.delta:
                    global_best_val = val
                    global_best_snap = new_snap
                    improved = True
                sat_w = not improved
                phase = "depth"
            else:
                new_snap, val = optimize_depth_at_fixed_width(global_best_snap, global_best_val)
                if val < global_best_val - acfg.delta:
                    global_best_val = val
                    global_best_snap = new_snap
                    improved = True
                sat_d = not improved
                phase = "width"
    else:
        logger.log_console(f"[WARN] Unknown adp_mode={mode}, skipping ADP search.")

    if log_loss:
        plot_loss_vs_epoch(val_history, results_dir / "loss_vs_epoch.png", title="ADPResNet - Loss vs Epoch")
    if log_neurons and improvements:
        n_list = [n for n, _ in improvements]
        l_list = [vv for _, vv in improvements]
        plot_loss_vs_neurons(n_list, l_list, results_dir / "loss_vs_neurons.png", title="ADPResNet - Loss vs Neurons")

    final_model = restore_arch_and_state(global_best_snap, device)
    return global_best_val, final_model, final_model.width, final_model.depth


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ADP on ResNet-style CIFAR backbone (ADPResNet).")

    # Data
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--no-augment", action="store_true", help="Disable CIFAR crop/flip (normalization stays)")
    p.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Use only the first N classes (labels 0..N-1). For CIFAR-10, 2–10 are valid.",
    )

    # Architecture
    p.add_argument("--width", type=int, default=16, help="Base channels in stage 1")
    p.add_argument("--depth", type=int, default=2, help="Blocks per stage (3 stages total)")
    p.add_argument("--dropout", type=float, default=0.0, help="Dropout probability inside residual blocks")

    # ADP config
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
    p.add_argument("--ex-k", type=int, default=4)
    p.add_argument("--max-width", type=int, default=128)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--max-neurons", type=int, default=5_000_000)

    # Optimisation
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # Regularisation
    p.add_argument("--cutmix-p", type=float, default=0.0, help="Probability of applying CutMix to a batch")
    p.add_argument("--cutmix-alpha", type=float, default=1.0, help="Alpha parameter for CutMix Beta distribution")

    # Logging / results
    p.add_argument("--results-dir", type=str, default="results_adp_resnet")
    p.add_argument("--plot-loss", action="store_true")
    p.add_argument("--plot-neurons", action="store_true")

    return p


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dl_train, dl_val, num_classes = make_loaders(
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        use_augment=not args.no_augment,
        num_classes_limit=args.num_classes,
    )

    model = make_adp_resnet(
        input_channels=3,
        num_classes=num_classes,
        width=args.width,
        depth=args.depth,
        dropout=args.dropout,
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
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        max_epochs=args.max_epochs,
        dropout=args.dropout,
        cutmix_p=args.cutmix_p,
        cutmix_alpha=args.cutmix_alpha,
    )

    results_dir = Path(args.results_dir)
    logger = ContinuousLogger(results_dir, "adp_resnet", args.adp_mode)

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


if __name__ == "__main__":
    main()
