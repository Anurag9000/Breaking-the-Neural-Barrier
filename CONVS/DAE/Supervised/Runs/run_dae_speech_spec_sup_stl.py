import argparse
import csv
import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from CONVS.DAE.Supervised.Models.dae_speech_spec_sup_stl import (
    SupDAESpeechSpec,
    sup_dae_speech_total_neurons,
)
from utils.audio_benchmarks import make_librispeech_ctc_loaders


def make_loaders(
    batch_size: int,
    num_workers: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, int]:
    dl_train, dl_val, _, vocab_size = make_librispeech_ctc_loaders(
        root="./data/LibriSpeech",
        batch_size=batch_size,
        download=True,
        n_mfcc=64,
        num_workers=num_workers,
    )
    return dl_train, dl_val, vocab_size


def train_one_epoch(
    model: SupDAESpeechSpec,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    lambda_recon: float,
) -> float:
    model.train()
    mse = nn.MSELoss()
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    total, n = 0.0, 0
    for spec, labels, lab_len in loader:
        spec = spec.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        lab_len = lab_len.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        spec_rec, logits = model(spec)
        loss_recon = mse(spec_rec, spec)

        # CTC expects (T,B,C)
        B, T, V = logits.shape
        logp = logits.log_softmax(-1).transpose(0, 1)
        input_len = torch.full((B,), T, dtype=torch.long, device=device)
        flat_labels = labels.view(-1)
        loss_ctc = ctc(logp, flat_labels, input_len, lab_len)

        loss = loss_ctc + lambda_recon * loss_recon
        loss.backward()
        opt.step()

        total += float(loss.item()) * B
        n += B
    return total / max(n, 1)


def eval_epoch(
    model: SupDAESpeechSpec,
    loader: DataLoader,
    device: torch.device,
    lambda_recon: float,
) -> float:
    model.eval()
    mse = nn.MSELoss(reduction="sum")
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    total, n = 0.0, 0
    with torch.no_grad():
        for spec, labels, lab_len in loader:
            spec = spec.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            lab_len = lab_len.to(device, non_blocking=True)
            spec_rec, logits = model(spec)
            loss_recon = mse(spec_rec, spec) / spec.size(0)

            B, T, V = logits.shape
            logp = logits.log_softmax(-1).transpose(0, 1)
            input_len = torch.full((B,), T, dtype=torch.long, device=device)
            flat_labels = labels.view(-1)
            loss_ctc = ctc(logp, flat_labels, input_len, lab_len)

            loss = loss_ctc + lambda_recon * loss_recon
            total += float(loss.item()) * B
            n += B
    return total / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="runs/dae_speech_spec_sup_stl")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--base", type=int, default=64)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--vocab-size", type=int, default=30)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lambda-recon", type=float, default=1.0)

    args = p.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dl_train, dl_val, vocab_size = make_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = SupDAESpeechSpec(vocab_size=vocab_size, base=args.base, depth=args.depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log_path = out_dir / "training_log.txt"
    stats_path = out_dir / "training_stats.csv"
    log_f = log_path.open("w", encoding="utf-8")
    stats_f = stats_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(stats_f)
    neurons = sup_dae_speech_total_neurons(args.base, args.depth, args.vocab_size)
    writer.writerow(
        ["epoch", "base", "depth", "neurons", "train_loss", "val_loss", "best_val", "best_epoch"]
    )

    best_val = float("inf")
    best_epoch = -1
    es_counter = 0
    ckpt_path = out_dir / "best.pt"

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model, dl_train, opt, device, args.lambda_recon
            )
            val_loss = eval_epoch(
                model, dl_val, device, args.lambda_recon
            )

            improved = val_loss < best_val - 1e-6
            if improved:
                best_val = val_loss
                best_epoch = epoch
                es_counter = 0
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
                es_counter += 1

            msg = (
                f"Epoch {epoch:03d} | train={train_loss:.6f} | "
                f"val={val_loss:.6f} | best_val={best_val:.6f} @ {best_epoch}"
            )
            print(msg)
            log_f.write(msg + "\n")

            writer.writerow(
                [epoch, args.base, args.depth, neurons, train_loss, val_loss, best_val, best_epoch]
            )
            stats_f.flush()

            if es_counter >= args.patience:
                stop_msg = (
                    f"Early stopping at epoch {epoch} "
                    f"(no improvement for {args.patience} epochs)"
                )
                print(stop_msg)
                log_f.write(stop_msg + "\n")
                break
    finally:
        log_f.flush()
        stats_f.flush()

    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
    test_loss = eval_epoch(model, dl_val, device, args.lambda_recon)

    report = {
        "base": args.base,
        "depth": args.depth,
        "neurons": neurons,
        "best_val": best_val,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
