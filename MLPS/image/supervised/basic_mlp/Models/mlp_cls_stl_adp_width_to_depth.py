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
BASELINE_PATH = Path(__file__).with_name("mlp_cls_stl.py").resolve()
_spec = importlib.util.spec_from_file_location("baseline_module", BASELINE_PATH)
baseline_module = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(baseline_module)
MLPClassifier = baseline_module.MLPClassifier  # type: ignore


@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"
    delta: float = 1e-4
    patience: int = 5  # early-stopping patience (per single-shot training)
    trials_width: int = 10  # <=0 => infinite
    trials_depth: int = 5  # <=0 => infinite
    ex_k: int = 1
    width_stage_margin_patience: int = 5
    width_stage_min_improve_pct: float = 1.0
    max_width: int = 4096
    max_depth: int = 10
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
    img_size: Tuple[int, int] = (28, 28)
    seed: int = 0
    num_workers: int = 0
    num_classes: int = 10


def _resize_linear(old: nn.Linear, new_out: int, new_in: int) -> nn.Linear:
    new = nn.Linear(new_in, new_out, bias=old.bias is not None).to(old.weight.device)
    with torch.no_grad():
        r = min(old.out_features, new_out)
        c = min(old.in_features, new_in)
        new.weight[:r, :c] = old.weight[:r, :c]
        if old.bias is not None and new.bias is not None:
            new.bias[:r] = old.bias[:r]
    return new


def _rebuild_mlp_classifier(model: Any, hidden_widths: List[int]) -> None:
    device = next(model.parameters()).device
    in_dim = model.in_dim
    num_classes = model.num_classes
    use_bn = model.use_bn

    layers = []
    prev = in_dim
    old_blocks = list(model.backbone)
    for w in hidden_widths:
        block = baseline_module.MLPBlock(prev, w, use_bn).to(device)  # type: ignore
        if old_blocks:
            old_block = old_blocks.pop(0)
            if hasattr(old_block, "linear"):
                block.linear = _resize_linear(old_block.linear, w, prev)
        layers.append(block)
        prev = w
    model.backbone = nn.Sequential(*layers)
    model.hidden_widths = list(hidden_widths)

    model.head = _resize_linear(model.head, num_classes, prev)


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


def expand_width(model: Any, ex_k: int, max_width: int) -> Optional[Any]:
    new_h = _next_staged_widths(model.hidden_widths, max_width, ex_k)
    if new_h == model.hidden_widths:
        return None
    _rebuild_mlp_classifier(model, new_h)
    return model


def expand_depth(model: Any, max_depth: int) -> Optional[Any]:
    if len(model.hidden_widths) >= max_depth:
        return None
    if not model.hidden_widths:
        return None
    if len(set(int(w) for w in model.hidden_widths)) != 1:
        return None
    new_h = model.hidden_widths + [int(model.hidden_widths[-1])]
    _rebuild_mlp_classifier(model, new_h)
    return model


def total_neurons(model: Any) -> int:
    # Rough proxy consistent across widths: sum hidden + output logits.
    return int(sum(model.hidden_widths) + model.num_classes)


def model_width(model: Any) -> int:
    return int(max(model.hidden_widths)) if model.hidden_widths else 0


def model_depth(model: Any) -> int:
    return int(len(model.hidden_widths))


def snapshot_arch_and_state(model: Any, state_dict=None) -> Dict[str, Any]:
    state = state_dict if state_dict is not None else model.state_dict()
    return {
        "in_dim": model.in_dim,
        "hidden_widths": list(model.hidden_widths),
        "num_classes": model.num_classes,
        "use_bn": model.use_bn,
        "state": copy.deepcopy(state),
    }


def restore_arch_and_state(model: Any, snap: Dict[str, Any], device) -> Any:
    new_model = MLPClassifier(
        in_dim=snap["in_dim"],
        hidden_widths=snap["hidden_widths"],
        num_classes=snap["num_classes"],
        use_bn=snap["use_bn"],
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
    if name == "mnist":
        ds = datasets.MNIST(root=data_dir, train=True, download=True, transform=tf)
        in_ch = 1
        num_classes = 10
    elif name == "fashionmnist":
        ds = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=tf)
        in_ch = 1
        num_classes = 10
    elif name == "cifar10":
        ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tf)
        in_ch = 3
        num_classes = 10
    elif name == "cifar100":
        ds = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=tf)
        in_ch = 3
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset}. Use cifar10 or cifar100.")

    n_val = int(len(ds) * val_split)
    n_train = len(ds) - n_val
    g = torch.Generator().manual_seed(int(seed))
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=g)

    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    in_dim = int(in_ch * img_size[0] * img_size[1])
    return dl_train, dl_val, in_dim, num_classes


def train_epoch(model: Any, dl, opt, device, *, grad_clip: float) -> Tuple[float, float]:
    model.train()
    total_loss, total_correct, n = 0.0, 0, 0
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        total_loss += loss.item() * x.size(0)
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        n += x.size(0)
    return float(total_loss / max(n, 1)), float(total_correct / max(n, 1))


@torch.no_grad()
def val_epoch(model: Any, dl, device) -> Tuple[float, float]:
    model.eval()
    total_loss, total_correct, n = 0.0, 0, 0
    for x, y in dl:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
        total_correct += int((logits.argmax(dim=1) == y).sum().item())
        n += x.size(0)
    return float(total_loss / max(n, 1)), float(total_correct / max(n, 1))


def train_with_early_stopping(
    model: Any,
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
        tr_loss, tr_acc = train_epoch(model, dl_train, opt, device, grad_clip=acfg.grad_clip)
        val_loss, val_acc = val_epoch(model, dl_val, device)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            es_counter = 0
            improved = True
        else:
            es_counter += 1
            improved = False

        msg = (
            f"Epoch {epoch} | train_loss={tr_loss:.6f} train_acc={tr_acc:.4f} "
            f"val_loss={val_loss:.6f} val_acc={val_acc:.4f} best={best_val:.6f} es={es_counter}/{acfg.patience}"
        )
        if logger is not None:
            logger.log_console(msg)
            logger.log_epoch_stats(
                {
                    "epoch": epoch,
                    "width": model_width(model),
                    "depth": model_depth(model),
                    "neurons": total_neurons(model),
                    "train_loss": tr_loss,
                    "train_acc": tr_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
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
    model: Any,
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

    p = argparse.ArgumentParser(description="ADP MLP Classifier (width/depth search)")
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 128], help="Hidden widths list; its length is fixed depth.")
    p.add_argument(
        "--adp-mode",
        type=str,
        default="width_to_depth",
        choices=["alt_width", "alt_depth", "width_to_depth", "depth_to_width"],
    )
    p.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashionmnist", "cifar10", "cifar100"])
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--img-size", type=int, nargs=2, default=[28, 28])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--results-dir", type=str, default="results_adp/mlp_cls_stl")
    p.add_argument("--delta", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5, help="Early-stopping patience per (single-shot) training run")
    p.add_argument("--trials-width", type=int, default=10, help="Expansion patience for width; <=0 means infinite")
    p.add_argument("--trials-depth", type=int, default=2, help="Expansion patience for depth; <=0 means infinite")
    p.add_argument("--ex-k", type=int, default=1)
    p.add_argument("--width-stage-margin-patience", type=int, default=5)
    p.add_argument("--width-stage-min-improve-pct", type=float, default=1.0)
    p.add_argument("--max-width", type=int, default=4096)
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--max-neurons", type=int, default=10_000_000)
    p.add_argument("--max-epochs", type=int, default=100000000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args()

    if tuple(args.img_size) == (28, 28) and args.dataset.lower() in {"cifar10", "cifar100"}:
        args.img_size = [32, 32]

    torch.manual_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dl_train, dl_val, in_dim, num_classes = make_loaders(
        args.dataset, args.data_dir, tuple(args.img_size), args.batch_size, 0.1, args.seed, args.num_workers
    )
    model = MLPClassifier(in_dim, hidden_widths=args.hidden, num_classes=num_classes)

    run_name = (
        f"{args.dataset}_{args.adp_mode}_d{len(args.hidden)}"
        f"_w{max(args.hidden) if args.hidden else 0}_exk{args.ex_k}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    results_dir = Path(args.results_dir) / run_name
    logger = ContinuousLogger(results_dir, "mlp_cls_stl", args.adp_mode)
    logger.log_console(f"Config: dataset={args.dataset} img_size={tuple(args.img_size)} hidden={args.hidden} classes={num_classes}")
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
        num_classes=num_classes,
    )

    try:
        best, model = adp_search(model.to(device), dl_train, dl_val, acfg, device, logger=logger)
        logger.log_console(f"[ADP MLP CLS] best_val_loss={best:.6f} hidden={model.hidden_widths} depth={len(model.hidden_widths)}")

        plot_val_loss_from_csv(logger.csv_file, results_dir / "val_loss_vs_step.png", title=f"{run_name} - val_loss")
        plot_best_loss_per_neurons_from_csv(
            logger.csv_file, results_dir / "loss_vs_neurons_best.png", title=f"{run_name} - best val_loss per neurons"
        )
    finally:
        logger.close()
