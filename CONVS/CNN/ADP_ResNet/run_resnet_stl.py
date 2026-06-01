"""
STL runner for the ADPResNet backbone.

Trains a single ADPResNet configuration (fixed width/depth) on CIFAR-10/100
with cosine LR, optional CutMix and dropout, and saves simple loss/accuracy
plots plus a small JSON summary.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from CONVS.CNN.ADP_ResNet.adp_resnet_backbone import ADPResNet, ADPResNetConfig, make_adp_resnet, estimate_neurons
from CONVS.CNN.ADP_ResNet.run_adp_resnet import make_loaders
from utils.cutmix import cutmix_batch


@dataclass
class STLConfig:
    dataset: str = "cifar10"
    data_root: str = "./data"
    batch_size: int = 128
    num_workers: int = 2
    val_split: float = 0.1
    no_augment: bool = False
    num_classes: int = 10

    width: int = 16
    depth: int = 2
    dropout: float = 0.0

    epochs: int = 300
    lr: float = 1e-3
    min_lr: float = 1e-5
    weight_decay: float = 5e-4
    patience: int = 50
    log_every: int = 1
    grad_clip: float = 1.0

    cutmix_p: float = 0.0
    cutmix_alpha: float = 1.0

    results_dir: str = "results_resnet_stl"
    loss_plot: str = "epoch_vs_loss.png"
    acc_plot: str = "epoch_vs_acc.png"


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    crit = nn.CrossEntropyLoss()
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = crit(logits, y)
        total_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total += y.size(0)
    if total == 0:
        return 0.0, 0.0
    return total_loss / total, total_correct / total


def _save_loss_plot(path: Path, epochs: List[int], vals: List[float]) -> None:
    if not epochs:
        return
    _ensure_dir(path.parent)
    plt.figure(figsize=(6, 4))
    plt.semilogy(epochs, vals, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss (log scale)")
    plt.title("Epoch vs Validation Loss (ADPResNet)")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path.as_posix())
    plt.close()


def _save_acc_plot(path: Path, epochs: List[int], vals: List[float]) -> None:
    if not epochs:
        return
    _ensure_dir(path.parent)
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, vals, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.title("Epoch vs Validation Accuracy (ADPResNet)")
    plt.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path.as_posix())
    plt.close()


def train_stl(cfg: STLConfig) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl_train, dl_val, num_classes = make_loaders(
        dataset=cfg.dataset,
        data_root=cfg.data_root,
        batch_size=cfg.batch_size,
        val_split=cfg.val_split,
        num_workers=cfg.num_workers,
        use_augment=not cfg.no_augment,
        num_classes_limit=cfg.num_classes,
    )

    model: ADPResNet = make_adp_resnet(
        input_channels=3,
        num_classes=num_classes,
        width=cfg.width,
        depth=cfg.depth,
        dropout=cfg.dropout,
    ).to(device)

    crit = nn.CrossEntropyLoss()
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.min_lr)

    best_val = float("inf")
    best_acc = 0.0
    best_epoch = 0
    bad = 0

    history = []
    epochs_hist: List[int] = []
    val_losses_hist: List[float] = []
    val_accs_hist: List[float] = []

    gen = torch.Generator(device=device)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running, seen = 0.0, 0

        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            xb_mix, targets = cutmix_batch(
                xb,
                yb,
                alpha=cfg.cutmix_alpha,
                p=cfg.cutmix_p,
                generator=gen,
            )

            opt.zero_grad(set_to_none=True)
            logits = model(xb_mix)
            if targets is None:
                loss = crit(logits, yb)
            else:
                y1, y2, lam = targets
                loss = lam * crit(logits, y1) + (1.0 - lam) * crit(logits, y2)

            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()

            running += loss.item() * xb.size(0)
            seen += xb.size(0)

        train_loss = running / max(1, seen)
        val_loss, val_acc = evaluate(model, dl_val, device)
        sched.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_acc": float(val_acc),
                "lr": float(opt.param_groups[0]["lr"]),
            }
        )

        epochs_hist.append(epoch)
        val_losses_hist.append(max(float(val_loss), 1e-12))
        val_accs_hist.append(float(val_acc))

        if epoch % cfg.log_every == 0:
            print(
                f"[epoch {epoch}/{cfg.epochs}] "
                f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
                f"val_acc={val_acc:.4f} | lr={opt.param_groups[0]['lr']:.2e}"
            )

        if val_loss < best_val - 0:
            best_val = float(val_loss)
            best_acc = float(val_acc)
            best_epoch = epoch
            bad = 0
        else:
            bad += 1

        if bad >= cfg.patience:
            print(
                f"[early stopping] no improvement for {bad} epochs "
                f"(best={best_val:.4f} @ epoch {best_epoch})"
            )
            break

    results_dir = Path(cfg.results_dir)
    _ensure_dir(results_dir)
    loss_path = results_dir / cfg.loss_plot
    acc_path = results_dir / cfg.acc_plot
    _save_loss_plot(loss_path, epochs_hist, val_losses_hist)
    _save_acc_plot(acc_path, epochs_hist, val_accs_hist)

    summary = {
        "config": asdict(cfg),
        "num_classes": num_classes,
        "neurons": int(estimate_neurons(cfg.width, cfg.depth)),
        "best_val_loss": best_val,
        "best_val_acc": best_acc,
        "best_epoch": best_epoch,
        "epochs_ran": len(epochs_hist),
        "loss_plot": loss_path.as_posix(),
        "acc_plot": acc_path.as_posix(),
        "history": history,
    }

    with open(results_dir / "stl_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[DONE] width={cfg.width} depth={cfg.depth} neurons={summary['neurons']} "
        f"| best_val={best_val:.4f} | best_acc={best_acc:.4f}"
    )
    return summary


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="STL training for ADPResNet on CIFAR.")

    # Data
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--num-classes", type=int, default=10,
                   help="Use only the first N classes (labels 0..N-1). For CIFAR-10, 2–10 are valid.")

    # Architecture
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)

    # Optimisation
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # CutMix
    p.add_argument("--cutmix-p", type=float, default=0.0)
    p.add_argument("--cutmix-alpha", type=float, default=1.0)

    # Results
    p.add_argument("--results-dir", type=str, default="results_resnet_stl")
    p.add_argument("--loss-plot", type=str, default="epoch_vs_loss.png")
    p.add_argument("--acc-plot", type=str, default="epoch_vs_acc.png")

    return p


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    cfg = STLConfig(
        dataset=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        no_augment=args.no_augment,
        num_classes=args.num_classes,
        width=args.width,
        depth=args.depth,
        dropout=args.dropout,
        epochs=args.epochs,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        log_every=args.log_every,
        grad_clip=args.grad_clip,
        cutmix_p=args.cutmix_p,
        cutmix_alpha=args.cutmix_alpha,
        results_dir=args.results_dir,
        loss_plot=args.loss_plot,
        acc_plot=args.acc_plot,
    )

    # Better default flush behaviour for long runs
    os.environ["PYTHONUNBUFFERED"] = "1"
    torch.backends.cudnn.benchmark = True
    train_stl(cfg)


if __name__ == "__main__":
    main()
