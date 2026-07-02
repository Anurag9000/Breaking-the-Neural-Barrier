import copy
import datetime as _dt
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger
from utils.adp_plot import plot_best_loss_per_neurons_from_csv, plot_val_loss_from_csv  # type: ignore
from utils.adp_contract import run_module_adp

# Load baseline
BASELINE_PATH = Path(__file__).with_name("mlp_ae_stl.py").resolve()
_spec = importlib.util.spec_from_file_location("baseline_module", BASELINE_PATH)
baseline_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(baseline_module)
MLPAutoencoder = baseline_module.MLPAutoencoder  # type: ignore


@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"
    delta: float = 1e-3
    patience: int = 5  # early-stopping patience (per single-shot training)
    trials_width: int = 10  # <=0 => infinite
    trials_depth: int = 5  # <=0 => infinite
    ex_k: int = 1
    width_stage_margin_patience: int = 5
    width_stage_min_improve_pct: float = 1.0
    max_width: int = 4096
    max_depth: int = 5
    width_stage_margin_patience: int = 5
    width_stage_min_improve_pct: float = 1.0
    depth_stage_margin_patience: int = 5
    depth_stage_min_improve_pct: float = 1.0
    min_new_layer_width: int = 10
    depth_first_seed_width: int = 20
    max_neurons: int = 10_000_000
    min_new_layer_width: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    batch_size: int = 128
    val_split: float = 0.1
    max_epochs: int = 100_000_000
    dataset: str = "cifar10"
    data_dir: str = "./data"
    img_size: Tuple[int, int] = (32, 32)
    seed: int = 0
    num_workers: int = 0


def _resize_linear(old: nn.Linear, new_out: int, new_in: int) -> nn.Linear:
    new = nn.Linear(new_in, new_out, bias=old.bias is not None).to(old.weight.device)
    with torch.no_grad():
        r = min(old.out_features, new_out)
        c = min(old.in_features, new_in)
        new.weight[:r, :c] = old.weight[:r, :c]
        if old.bias is not None and new.bias is not None:
            new.bias[:r] = old.bias[:r]
    return new


def _rebuild_mlp_ae(model: MLPAutoencoder, hidden_widths: List[int]) -> None:
    device = next(model.parameters()).device
    in_dim = model.in_dim
    bottleneck = model.bottleneck
    use_bn = model.use_bn

    enc_layers = []
    prev = in_dim
    old_enc = list(model.enc)
    for w in hidden_widths:
        block = baseline_module.MLPBlock(prev, w, use_bn).to(device)  # type: ignore
        if old_enc:
            old_block = old_enc.pop(0)
            block.linear = _resize_linear(old_block.linear, w, prev)
        enc_layers.append(block)
        prev = w
    model.enc = nn.Sequential(*enc_layers)
    model.hidden_widths = list(hidden_widths)

    model.fc_mu = _resize_linear(model.fc_mu, bottleneck, prev)

    dec_layers = []
    prev_dec = bottleneck
    old_dec = list(model.dec)
    for w in reversed(hidden_widths):
        block = baseline_module.MLPBlock(prev_dec, w, use_bn).to(device)  # type: ignore
        if old_dec:
            old_block = old_dec.pop(0)
            block.linear = _resize_linear(old_block.linear, w, prev_dec)
        dec_layers.append(block)
        prev_dec = w
    model.dec = nn.Sequential(*dec_layers)
    model.out = _resize_linear(model.out, model.out.out_features, prev_dec)


def _next_staged_widths(hidden_widths: List[int], max_width: int, ex_k: int) -> List[int]:
    widths = [int(w) for w in hidden_widths]
    if not widths:
        return widths
    target = min(max(widths) + max(1, int(ex_k)), int(max_width)) if len(set(widths)) == 1 else max(widths)
    next_widths = list(widths)
    for idx, width in enumerate(next_widths):
        if width < target:
            next_widths[idx] = width + 1
            break
    return next_widths


def expand_width(model: MLPAutoencoder, ex_k: int, max_width: int) -> Optional[MLPAutoencoder]:
    new_h = _next_staged_widths(model.hidden_widths, max_width, ex_k)
    if new_h == model.hidden_widths:
        return None
    _rebuild_mlp_ae(model, new_h)
    return model


def expand_depth(model: MLPAutoencoder, max_depth: int) -> Optional[MLPAutoencoder]:
    if len(model.hidden_widths) >= max_depth:
        return None
    if not model.hidden_widths:
        return None
    if len(set(int(w) for w in model.hidden_widths)) != 1:
        return None
    if int(model.hidden_widths[-1]) <= 10:
        return None
    new_h = model.hidden_widths + [10]
    _rebuild_mlp_ae(model, new_h)
    return model


def total_neurons(model: MLPAutoencoder) -> int:
    enc = sum(model.hidden_widths)
    dec = sum(model.hidden_widths)
    return int(enc + dec + model.bottleneck)


def model_width(model: MLPAutoencoder) -> int:
    return int(max(model.hidden_widths)) if model.hidden_widths else 0


def model_depth(model: MLPAutoencoder) -> int:
    return int(len(model.hidden_widths))


def snapshot_arch_and_state(model: MLPAutoencoder, state_dict=None) -> Dict[str, Any]:
    state = state_dict if state_dict is not None else model.state_dict()
    return {
        "in_dim": model.in_dim,
        "hidden_widths": list(model.hidden_widths),
        "bottleneck": model.bottleneck,
        "use_bn": model.use_bn,
        "output_activation": getattr(model, "output_activation", "sigmoid"),
        "state": copy.deepcopy(state),
    }


def restore_arch_and_state(model: MLPAutoencoder, snap: Dict[str, Any], device) -> MLPAutoencoder:
    new_model = MLPAutoencoder(
        in_dim=snap["in_dim"],
        hidden_widths=snap["hidden_widths"],
        bottleneck=snap["bottleneck"],
        use_bn=snap["use_bn"],
        output_activation=snap.get("output_activation", "sigmoid"),
    ).to(device)
    new_model.load_state_dict(snap["state"])
    return new_model


def make_loaders(
    dataset: str,
    data_dir: str,
    img_size: Tuple[int, int],
    batch_size: int,
    val_split: float,
    seed: int,
    num_workers: int,
):
    tf = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    name = dataset.lower()
    if name == "cifar10":
        ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tf)
        in_ch = 3
    elif name == "cifar100":
        ds = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=tf)
        in_ch = 3
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Use cifar10 or cifar100.")

    n_val = int(len(ds) * val_split)
    n_train = len(ds) - n_val
    g = torch.Generator().manual_seed(int(seed))
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)

    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False)
    dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)

    in_dim = int(in_ch * img_size[0] * img_size[1])
    return dl_train, dl_val, in_dim


def train_epoch(model: MLPAutoencoder, dl, opt, device, *, grad_clip: float) -> float:
    model.train()
    total, n = 0.0, 0
    for x, _ in dl:
        x = x.to(device)
        target = x.view(x.size(0), -1)
        opt.zero_grad(set_to_none=True)
        xr = model(x)
        loss = F.mse_loss(xr, target)
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return float(total / max(n, 1))


@torch.no_grad()
def val_epoch(model: MLPAutoencoder, dl, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for x, _ in dl:
        x = x.to(device)
        target = x.view(x.size(0), -1)
        xr = model(x)
        loss = F.mse_loss(xr, target)
        total += loss.item() * x.size(0)
        n += x.size(0)
    return float(total / max(n, 1))


def train_with_early_stopping(
    model: MLPAutoencoder,
    dl_train,
    dl_val,
    acfg: ADPConfig,
    device,
    *,
    logger: Optional[ContinuousLogger] = None,
    verbose: bool = True,
) -> Tuple[float, Dict[str, Any]]:
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    es_counter = 0

    for epoch in range(1, int(acfg.max_epochs) + 1):
        tr = train_epoch(model, dl_train, opt, device, grad_clip=acfg.grad_clip)
        val = val_epoch(model, dl_val, device)

        if val < best_val:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())
            es_counter = 0
            improved = True
        else:
            es_counter += 1
            improved = False

        msg = f"Epoch {epoch} | train_mse={tr:.6f} val_mse={val:.6f} best={best_val:.6f} es={es_counter}/{acfg.patience}"
        if logger is not None:
            logger.log_console(msg)
            logger.log_epoch_stats(
                {
                    "epoch": epoch,
                    "width": model_width(model),
                    "depth": model_depth(model),
                    "neurons": total_neurons(model),
                    "train_loss": tr,
                    "val_loss": val,
                    "best_val": best_val,
                    "es_counter": es_counter,
                    "improved": improved,
                }
            )
        elif verbose:
            print(msg)

        if es_counter >= int(acfg.patience):
            break

    return best_val, best_state


def adp_search(
    model: MLPAutoencoder,
    dl_train,
    dl_val,
    acfg: ADPConfig,
    device,
    *,
    logger: Optional[ContinuousLogger] = None,
):
    return run_module_adp(
        globals(),
        model,
        dl_train,
        dl_val,
        acfg,
        device,
        logger=logger,
    )


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="ADP MLP Autoencoder (width/depth search)")
    p.add_argument("--hidden", type=int, nargs="+", default=[1024, 512], help="Hidden widths list; its length is fixed depth.")
    p.add_argument("--bottleneck", type=int, default=256)
    p.add_argument(
        "--adp-mode",
        type=str,
        default="width_to_depth",
        choices=["alt_width", "alt_depth", "width_to_depth", "depth_to_width"],
    )
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--img-size", type=int, nargs=2, default=[32, 32])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", type=str, default="results_adp/mlp_ae_stl")
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5, help="Early-stopping patience per (single-shot) training run")
    p.add_argument("--trials-width", type=int, default=10, help="Expansion patience for width; <=0 means infinite")
    p.add_argument("--trials-depth", type=int, default=2, help="Expansion patience for depth; <=0 means infinite")
    p.add_argument("--ex-k", type=int, default=1)
    p.add_argument("--width-stage-margin-patience", type=int, default=5)
    p.add_argument("--width-stage-min-improve-pct", type=float, default=1.0)
    p.add_argument("--max-width", type=int, default=4096)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--max-neurons", type=int, default=10_000_000)
    p.add_argument("--max-epochs", type=int, default=100000000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dl_train, dl_val, in_dim = make_loaders(
        args.dataset, args.data_dir, tuple(args.img_size), args.batch_size, 0.1, args.seed, args.num_workers
    )
    model = MLPAutoencoder(in_dim, hidden_widths=args.hidden, bottleneck=args.bottleneck)

    run_name = (
        f"{args.dataset}_{args.adp_mode}_d{len(args.hidden)}"
        f"_w{max(args.hidden) if args.hidden else 0}_exk{args.ex_k}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    results_dir = Path(args.results_dir) / run_name
    logger = ContinuousLogger(results_dir, "mlp_ae_stl", args.adp_mode)
    logger.log_console(f"Config: dataset={args.dataset} img_size={tuple(args.img_size)} hidden={args.hidden} bottleneck={args.bottleneck}")
    logger.log_console(
        f"ADP: mode={args.adp_mode} ex_k={args.ex_k} trials_width={args.trials_width} trials_depth={args.trials_depth} "
        f"width_stage_margin_patience={args.width_stage_margin_patience} width_stage_min_improve_pct={args.width_stage_min_improve_pct} "
        f"max_width={args.max_width} max_depth={args.max_depth} max_neurons={args.max_neurons}"
    )
    logger.log_console(
        f"Train: batch_size={args.batch_size} lr=1e-3 weight_decay=1e-4 es_patience={args.patience} max_epochs={args.max_epochs}"
    )
    logger.log_console(f"Device: {device}")

    acfg = ADPConfig(
        adp_mode=args.adp_mode,
        delta=args.delta,
        patience=args.patience,
        trials_width=args.trials_width,
        trials_depth=args.trials_depth,
        ex_k=args.ex_k,
        width_stage_margin_patience=args.width_stage_margin_patience,
        width_stage_min_improve_pct=args.width_stage_min_improve_pct,
        max_width=args.max_width,
        max_depth=args.max_depth,
        max_neurons=args.max_neurons,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        dataset=args.dataset,
        data_dir=args.data_dir,
        img_size=tuple(args.img_size),
        seed=args.seed,
        num_workers=args.num_workers,
    )

    try:
        best, model = adp_search(model.to(device), dl_train, dl_val, acfg, device, logger=logger)
        logger.log_console(f"[ADP MLP AE] best_val={best:.6f} hidden={model.hidden_widths} depth={len(model.hidden_widths)}")

        plot_val_loss_from_csv(logger.csv_file, results_dir / "val_loss_vs_step.png", title=f"{run_name} - val_mse")
        plot_best_loss_per_neurons_from_csv(
            logger.csv_file, results_dir / "loss_vs_neurons_best.png", title=f"{run_name} - best val_mse per neurons"
        )
    finally:
        logger.close()
