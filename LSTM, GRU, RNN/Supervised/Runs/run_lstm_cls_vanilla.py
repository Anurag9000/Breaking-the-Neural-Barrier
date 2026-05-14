from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn

from lstm_cls_vanilla import LSTMClassifier, LSTMClassifierConfig
from _common_real_text import make_20newsgroups_loaders


def train_epoch(model, loader, optimizer, criterion, device):
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
            correct += int((logits.argmax(dim=-1) == labels).sum().item())
    return total / max(1, count), correct / max(1, count)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab-size", type=int, default=20000)
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pad-idx", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", type=str, default="results_lstm")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    train_loader, val_loader, test_loader, vocab_size, num_classes = make_20newsgroups_loaders(
        batch_size=args.batch_size, vocab_size=args.vocab_size, seed=args.seed
    )
    cfg = LSTMClassifierConfig(
        vocab_size=vocab_size,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_classes=num_classes,
        pad_idx=args.pad_idx,
        bidirectional=False,
    )
    model = LSTMClassifier(cfg).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, args.device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, args.device)
        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, args.device)
    print(f"TEST | loss={test_loss:.4f} | acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
