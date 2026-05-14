from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

from _common_real_text import make_20newsgroups_loaders
from lstm_cls_milstm import MILSTMClassifier, MILSTMConfig


class EarlyStopper:
    def __init__(self, patience: int):
        self.patience = patience
        self.best = float("inf")
        self.bad = 0
        self.state = None

    def step(self, value: float, model: nn.Module) -> bool:
        if value < self.best - 1e-7:
            self.best = value
            self.bad = 0
            self.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.bad += 1
        return self.bad >= self.patience

    def restore(self, model: nn.Module):
        if self.state is not None:
            model.load_state_dict(self.state)


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
            corr += int((logits.argmax(-1) == labels).sum().item())
    return total / max(1, n), corr / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=20000)
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", type=str, default="results_lstm")
    ap.add_argument("--seed", type=int, default=717)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)

    train_loader, val_loader, test_loader, vocab_size, num_classes = make_20newsgroups_loaders(
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )
    cfg = MILSTMConfig(
        vocab_size=vocab_size,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=num_classes,
        pad_idx=0,
    )
    model = MILSTMClassifier(cfg).to(args.device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()
    es = EarlyStopper(args.patience)

    for epoch in range(1, args.max_epochs + 1):
        tr = train_epoch(model, train_loader, opt, crit, args.device)
        vl, vacc = eval_epoch(model, val_loader, crit, args.device)
        print(f"Epoch {epoch:03d} | train_loss {tr:.4f} | val_loss {vl:.4f} | val_acc {vacc:.4f}")
        if es.step(vl, model):
            print("Early stopping.")
            break

    es.restore(model)
    tl, ta = eval_epoch(model, test_loader, crit, args.device)
    print(f"TEST | loss {tl:.4f} | acc {ta:.4f}")
    torch.save({"model_state": model.state_dict(), "config": cfg.__dict__, "test_loss": tl, "test_acc": ta}, os.path.join(args.outdir, "lstm_cls_milstm.pt"))


if __name__ == "__main__":
    main()
