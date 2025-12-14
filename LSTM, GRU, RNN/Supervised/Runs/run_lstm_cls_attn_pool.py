import argparse
import os
import random
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import List, Tuple

from lstm_cls_attn_pool import LSTMAttnPoolClassifier, LSTMAttnPoolConfig


class SyntheticTextCls(Dataset):
    def __init__(self, n: int, vocab_size: int, min_len: int, max_len: int, pad_idx: int = 0):
        rng = random.Random(555)
        self.samples = []
        for _ in range(n):
            L = rng.randint(min_len, max_len)
            x = [rng.randint(1, vocab_size - 1) for _ in range(L)]
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


class EarlyStopper:
    def __init__(self, patience: int):
        self.patience = patience
        self.best = float("inf")
        self.bad = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module):
        if val_loss < self.best - 1e-7:
            self.best = val_loss
            self.bad = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.bad += 1
        return self.bad >= self.patience

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def train_epoch(model, loader, opt, crit, device):
    model.train()
    total = 0.0
    n = 0
    for tokens, lengths, labels in loader:
        tokens, lengths, labels = tokens.to(device), lengths.to(device), labels.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(tokens, lengths)
        loss = crit(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += float(loss.item()) * labels.size(0)
        n += labels.size(0)
    return total / max(1, n)


def eval_epoch(model, loader, crit, device):
    model.eval()
    total = 0.0
    n = 0
    corr = 0
    with torch.no_grad():
        for tokens, lengths, labels in loader:
            tokens, lengths, labels = tokens.to(device), lengths.to(device), labels.to(device)
            logits = model(tokens, lengths)
            loss = crit(logits, labels)
            total += float(loss.item()) * labels.size(0)
            n += labels.size(0)
            preds = logits.argmax(dim=-1)
            corr += int((preds == labels).sum().item())
    return total / max(1, n), corr / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=20000)
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=80)
    ap.add_argument("--train-n", type=int, default=8000)
    ap.add_argument("--val-n", type=int, default=1000)
    ap.add_argument("--test-n", type=int, default=1000)

    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--attn-hidden", type=int, default=128)
    ap.add_argument("--num-classes", type=int, default=2)
    ap.add_argument("--pad-idx", type=int, default=0)

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--outdir", type=str, default="results_lstm")
    args = ap.parse_args()

    random.seed(555)
    torch.manual_seed(555)

    os.makedirs(args.outdir, exist_ok=True)

    train_ds = SyntheticTextCls(args.train_n, args.vocab_size, args.min_len, args.max_len, pad_idx=args.pad_idx)
    val_ds = SyntheticTextCls(args.val_n, args.vocab_size, args.min_len, args.max_len, pad_idx=args.pad_idx)
    test_ds = SyntheticTextCls(args.test_n, args.vocab_size, args.min_len, args.max_len, pad_idx=args.pad_idx)

    collate = lambda batch: collate_pad(batch, pad_idx=args.pad_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    cfg = LSTMAttnPoolConfig(
        vocab_size=args.vocab_size,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=args.num_classes,
        pad_idx=args.pad_idx,
        attn_hidden=args.attn_hidden,
        bidirectional=False,
    )
    model = LSTMAttnPoolClassifier(cfg).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()
    es = EarlyStopper(args.patience)


    # Init Logger


    logger = ContinuousLogger(Path('results_run_lstm_cls_attn_pool'), 'run_lstm_cls_attn_pool', 'train')


    for epoch in range(1, args.max_epochs + 1):
        tr = train_epoch(model, train_loader, opt, crit, args.device)
        vl, vacc = eval_epoch(model, val_loader, crit, args.device)
        stop = es.step(vl, model)
        # Log

        msg = f"Epoch {epoch:03d} | train_loss={tr:.4f} | val_loss={vl:.4f} | val_acc={vacc:.4f}"

        logger.log_console(msg)

        logger.log_epoch_stats({

            "epoch": epoch,

            "val_loss": val_loss if 'val_loss' in locals() else (loss.item() if 'loss' in locals() else 0),

            "train_loss": loss.item() if 'loss' in locals() else 0

        })
        if stop:
            print("Early stopping.")
            break

    es.restore_best(model)
    tl, ta = eval_epoch(model, test_loader, crit, args.device)
    print(f"TEST | loss={tl:.4f} | acc={ta:.4f}")

    torch.save({
        "model_state": model.state_dict(),
        "config": cfg.__dict__,
        "test_loss": tl,
        "test_acc": ta,
    }, os.path.join(args.outdir, "lstm_cls_attn_pool.pt"))


if __name__ == "__main__":
    main()