import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from itertools import zip_longest

from ..Models.dae_tokenmask_text_semisup_stl import SupTokenMaskSemiDAE, sup_token_dae_total_neurons


class ToyTextDataset(Dataset):
    """
    Very simple synthetic token dataset for demonstration:
    sequences of integers in [1, vocab_size-1] with random labels.
    """

    def __init__(self, num_samples: int, seq_len: int, vocab_size: int, num_classes: int, seed: int = 1337):
        g = torch.Generator().manual_seed(seed)
        self.input_ids = torch.randint(1, vocab_size, (num_samples, seq_len), generator=g)
        self.labels = torch.randint(0, num_classes, (num_samples,), generator=g)

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, idx: int):
        return self.input_ids[idx], self.labels[idx]


def build_toy_text_loaders(
    num_samples: int,
    seq_len: int,
    vocab_size: int,
    num_classes: int,
    batch_size: int,
    val_frac: float,
    label_frac: float,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    full = ToyTextDataset(num_samples, seq_len, vocab_size, num_classes, seed=seed)
    n_val = int(len(full) * val_frac)
    n_train = len(full) - n_val
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    n_label = int(n_train * label_frac)
    labeled_ds, unlabeled_ds = random_split(
        train_ds, [n_label, n_train - n_label], generator=torch.Generator().manual_seed(seed + 1)
    )

    labeled_loader = DataLoader(labeled_ds, batch_size=batch_size, shuffle=True)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return labeled_loader, unlabeled_loader, val_loader, test_loader


def apply_token_mask(input_ids: torch.Tensor, mask_token_id: int, mask_prob: float) -> torch.Tensor:
    if mask_prob <= 0:
        return input_ids
    mask = torch.rand_like(input_ids.float()) < mask_prob
    return torch.where(mask, torch.full_like(input_ids, mask_token_id), input_ids)


def train_one_epoch(
    model: SupTokenMaskSemiDAE,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    opt: optim.Optimizer,
    device: torch.device,
    mask_token_id: int,
    mask_prob: float,
    lambda_recon: float,
) -> Tuple[float, float]:
    model.train()
    ce = nn.CrossEntropyLoss()
    total_loss, total_cls, n_labeled = 0.0, 0.0, 0

    for lab_batch, unlab_batch in zip_longest(labeled_loader, unlabeled_loader):
        loss = 0.0
        loss_cls = 0.0

        if lab_batch is not None:
            ids_lab, y_lab = lab_batch
            ids_lab = ids_lab.to(device)
            y_lab = y_lab.to(device)
            ids_lab_masked = apply_token_mask(ids_lab, mask_token_id, mask_prob)
            token_logits_lab, cls_logits_lab = model(ids_lab_masked)

            # Reconstruction loss only on non-masked tokens is complex; here we
            # simply use full-token CE as a proxy.
            B, L = ids_lab.shape
            recon_loss_lab = ce(
                token_logits_lab.view(B * L, -1),
                ids_lab.view(-1),
            )
            cls_loss_lab = ce(cls_logits_lab, y_lab)
            loss = loss + lambda_recon * recon_loss_lab + cls_loss_lab
            loss_cls = loss_cls + cls_loss_lab
            n_labeled += B

        if unlab_batch is not None:
            ids_un, _ = unlab_batch
            ids_un = ids_un.to(device)
            ids_un_masked = apply_token_mask(ids_un, mask_token_id, mask_prob)
            token_logits_un, _ = model(ids_un_masked)
            B, L = ids_un.shape
            recon_loss_un = ce(
                token_logits_un.view(B * L, -1),
                ids_un.view(-1),
            )
            loss = loss + lambda_recon * recon_loss_un

        if lab_batch is None and unlab_batch is None:
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        total_loss += float(loss.item())
        total_cls += float(loss_cls.item())

    return total_loss / max(len(labeled_loader), 1), total_cls / max(n_labeled, 1)


def eval_epoch(
    model: SupTokenMaskSemiDAE,
    loader: DataLoader,
    device: torch.device,
    mask_token_id: int,
    mask_prob: float,
    lambda_recon: float,
) -> Tuple[float, float]:
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_cls, n = 0.0, 0.0, 0
    with torch.no_grad():
        for ids, y in loader:
            ids = ids.to(device)
            y = y.to(device)
            ids_masked = apply_token_mask(ids, mask_token_id, mask_prob)
            token_logits, cls_logits = model(ids_masked)
            B, L = ids.shape
            recon_loss = ce(token_logits.view(B * L, -1), ids.view(-1)) / B
            cls_loss = ce(cls_logits, y) / B
            loss = lambda_recon * recon_loss + cls_loss

            total_loss += float(loss.item()) * B
            total_cls += float(cls_loss.item()) * B
            n += B
    return total_loss / max(n, 1), total_cls / max(n, 1)


def eval_accuracy(model: SupTokenMaskSemiDAE, loader: DataLoader, device: torch.device, mask_token_id: int, mask_prob: float) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for ids, y in loader:
            ids = ids.to(device)
            y = y.to(device)
            ids_masked = apply_token_mask(ids, mask_token_id, mask_prob)
            _, cls_logits = model(ids_masked)
            preds = cls_logits.argmax(dim=1)
            correct += int((preds == y).sum().item())
            total += ids.size(0)
    return correct / max(total, 1)


def main() -> None:
    p = argparse.ArgumentParser(description="Semi-supervised token-mask DAE (toy text)")
    p.add_argument("--num-samples", type=int, default=20000)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--vocab-size", type=int, default=1000)
    p.add_argument("--num-classes", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--label-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1337)

    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--dim-feedforward", type=int, default=1024)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--mask-prob", type=float, default=0.15)
    p.add_argument("--lambda-recon", type=float, default=1.0)

    p.add_argument("--out-dir", type=str, default="runs/dae_tokenmask_text_semisup_toy")

    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask_token_id = args.vocab_size - 1

    lab_loader, unlab_loader, val_loader, test_loader = build_toy_text_loaders(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        label_frac=args.label_frac,
        seed=args.seed,
    )

    model = SupTokenMaskSemiDAE(
        vocab_size=args.vocab_size,
        num_classes=args.num_classes,
        d_model=args.d_model,
        depth=args.depth,
        num_heads=args.num_heads,
        dim_feedforward=args.dim_feedforward,
        max_len=args.seq_len,
        pad_id=0,
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "training_log.txt"
    stats_path = out_dir / "training_stats.csv"
    ckpt_path = out_dir / "best.pt"

    log_f = log_path.open("w", encoding="utf-8")
    stats_f = stats_path.open("w", newline="", encoding="utf-8")
    stats_writer = csv.writer(stats_f)
    neurons = sup_token_dae_total_neurons(args.d_model, args.depth, args.num_classes, args.vocab_size)

    stats_writer.writerow(
        [
            "epoch",
            "d_model",
            "depth",
            "neurons",
            "train_loss",
            "train_cls",
            "val_loss",
            "val_cls",
            "val_acc",
            "best_val",
            "best_epoch",
        ]
    )

    best_val = float("inf")
    best_epoch = -1
    epochs_no_improve = 0

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss, train_cls = train_one_epoch(
                model,
                lab_loader,
                unlab_loader,
                opt,
                device,
                mask_token_id=mask_token_id,
                mask_prob=args.mask_prob,
                lambda_recon=args.lambda_recon,
            )
            val_loss, val_cls = eval_epoch(
                model,
                val_loader,
                device,
                mask_token_id=mask_token_id,
                mask_prob=args.mask_prob,
                lambda_recon=args.lambda_recon,
            )
            val_acc = eval_accuracy(
                model,
                val_loader,
                device,
                mask_token_id=mask_token_id,
                mask_prob=args.mask_prob,
            )

            improved = val_loss < best_val - 1e-6
            if improved:
                best_val = val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "args": vars(args),
                    },
                    ckpt_path,
                )
            else:
                epochs_no_improve += 1

            msg = (
                f"Epoch {epoch:03d} | train={train_loss:.4f} (cls={train_cls:.4f}) | "
                f"val={val_loss:.4f} (cls={val_cls:.4f}, acc={val_acc:.4f}) | "
                f"best_val={best_val:.4f} @ {best_epoch}"
            )
            print(msg)
            log_f.write(msg + "\n")

            stats_writer.writerow(
                [
                    epoch,
                    args.d_model,
                    args.depth,
                    neurons,
                    train_loss,
                    train_cls,
                    val_loss,
                    val_cls,
                    val_acc,
                    best_val,
                    best_epoch,
                ]
            )
            stats_f.flush()

            if epochs_no_improve >= args.patience:
                stop_msg = f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)"
                print(stop_msg)
                log_f.write(stop_msg + "\n")
                break
    finally:
        log_f.flush()
        stats_f.flush()

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
    test_acc = eval_accuracy(
        model,
        test_loader,
        device,
        mask_token_id=mask_token_id,
        mask_prob=args.mask_prob,
    )

    report = {
        "num_samples": args.num_samples,
        "seq_len": args.seq_len,
        "vocab_size": args.vocab_size,
        "num_classes": args.num_classes,
        "d_model": args.d_model,
        "depth": args.depth,
        "neurons_metric": neurons,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "test_acc": test_acc,
        "lambda_recon": args.lambda_recon,
        "mask_prob": args.mask_prob,
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log_f.write("\n" + json.dumps(report, indent=2) + "\n")
    log_f.close()
    stats_f.close()


if __name__ == "__main__":
    main()

