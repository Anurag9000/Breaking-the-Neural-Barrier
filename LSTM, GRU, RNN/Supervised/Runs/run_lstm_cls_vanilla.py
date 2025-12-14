import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple
import random

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger.nn as nn
from torch.utils.data import Dataset, DataLoader

from lstm_cls_vanilla import LSTMClassifier, LSTMClassifierConfig


# ---------------------------- Synthetic dataset (replace with real) ----------------------------
class SyntheticTextCls(Dataset):
    """Generates synthetic integer token sequences with a parity-based label.
    Replace this with a real dataset for practical use.
    """
    def __init__(self, n: int, vocab_size: int, min_len: int, max_len: int, pad_idx: int = 0):
        self.samples = []
        rng = random.Random(1337)
        for _ in range(n):
            L = rng.randint(min_len, max_len)
            x = [rng.randint(1, vocab_size - 1) for _ in range(L)]
            # Label = 1 if sum tokens even else 0 (arbitrary)
            y = int(sum(x) % 2 == 0)
            self.samples.append((x, y))
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_pad(batch: List[Tuple[List[int], int]], pad_idx: int):
    lengths = torch.tensor([len(x) for x, _ in batch], dtype=torch.long)
    max_len = int(lengths.max())
    tokens = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
    labels = torch.tensor([y for _, y in batch], dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        tokens[i, : len(x)] = torch.tensor(x, dtype=torch.long)
    return tokens, lengths, labels


# ------------------------------ Training utilities ------------------------------
@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 64
    max_epochs: int = 30
    patience: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class EarlyStopper:
    def __init__(self, patience: int):
        self.patience = patience
        self.best = float("inf")
        self.bad_epochs = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module):
        improved = val_loss < self.best - 1e-7
        if improved:
            self.best = val_loss
            self.bad_epochs = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.bad_epochs += 1
        return improved

    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ------------------------------ Train / Eval loops ------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    count = 0
    for tokens, lengths, labels in loader:
        tokens, lengths, labels = tokens.to(device), lengths.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.item()) * labels.size(0)
        count += labels.size(0)
    return total / max(1, count)


def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    count = 0
    correct = 0
    with torch.no_grad():
        for tokens, lengths, labels in loader:
            tokens, lengths, labels = tokens.to(device), lengths.to(device), labels.to(device)
            logits = model(tokens, lengths)
            loss = criterion(logits, labels)
            total += float(loss.item()) * labels.size(0)
            count += labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += int((preds == labels).sum().item())
    return total / max(1, count), correct / max(1, count)


# ------------------------------------- Main -------------------------------------

def main():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--vocab-size", type=int, default=20000)
    p.add_argument("--min-len", type=int, default=20)
    p.add_argument("--max-len", type=int, default=80)
    p.add_argument("--train-n", type=int, default=8000)
    p.add_argument("--val-n", type=int, default=1000)
    p.add_argument("--test-n", type=int, default=1000)

    # Model
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--pad-idx", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)

    # Train
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--outdir", type=str, default="results_lstm")

    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.outdir, exist_ok=True)

    # Build datasets
    train_ds = SyntheticTextCls(args.train_n, args.vocab_size, args.min_len, args.max_len, pad_idx=args.pad_idx)
    val_ds = SyntheticTextCls(args.val_n, args.vocab_size, args.min_len, args.max_len, pad_idx=args.pad_idx)
    test_ds = SyntheticTextCls(args.test_n, args.vocab_size, args.min_len, args.max_len, pad_idx=args.pad_idx)

    collate = lambda batch: collate_pad(batch, pad_idx=args.pad_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # Build model
    cfg = LSTMClassifierConfig(
        vocab_size=args.vocab_size,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=args.num_classes,
        pad_idx=args.pad_idx,
        bidirectional=False,
    )
    model = LSTMClassifier(cfg).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    stopper = EarlyStopper(patience=args.patience)

    best_val_acc = 0.0

    # Init Logger

    logger = ContinuousLogger(Path('results_run_lstm_cls_vanilla'), 'run_lstm_cls_vanilla', 'train')

    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, args.device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, args.device)
        stopper.step(val_loss, model)
        best_val_acc = max(best_val_acc, val_acc)
        # Log

        msg = f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"

        logger.log_console(msg)

        logger.log_epoch_stats({

            "epoch": epoch,

            "val_loss": val_loss if 'val_loss' in locals() else (loss.item() if 'loss' in locals() else 0),

            "train_loss": loss.item() if 'loss' in locals() else 0

        })
        if stopper.should_stop():
            print("Early stopping triggered.")
            break

    stopper.restore_best(model)

    # Final test
    test_loss, test_acc = evaluate(model, test_loader, criterion, args.device)
    print(f"TEST | loss={test_loss:.4f} | acc={test_acc:.4f}")

    # Save
    torch.save({
        "model_state": model.state_dict(),
        "config": cfg.__dict__,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }, os.path.join(args.outdir, "lstm_cls_vanilla.pt"))


if __name__ == "__main__":
    main()