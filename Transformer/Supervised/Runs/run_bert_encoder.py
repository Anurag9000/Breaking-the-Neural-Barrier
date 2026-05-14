import argparse
from pathlib import Path

import torch
import torch.nn as nn

from utils.adp_logging import ContinuousLogger
from utils.text_benchmarks import make_ag_news_classification_loaders

from model_bert_encoder import BERTEncoder


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for ids, labels in loader:
            ids = ids.to(device)
            labels = labels.to(device)
            logits = model(ids)
            pred = logits.argmax(-1)
            correct += int((pred == labels).sum().item())
            total += int(labels.numel())
    return correct / max(total, 1)


def main():
    p = argparse.ArgumentParser(description="BERT-style encoder on AG News")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_len", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--min_freq", type=int, default=2)
    p.add_argument("--max_vocab", type=int, default=50000)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    train_loader, val_loader, test_loader, vocab, num_classes = make_ag_news_classification_loaders(
        batch_size=args.batch_size,
        max_len=args.max_len,
        seed=args.seed,
        val_fraction=args.val_fraction,
        min_freq=args.min_freq,
        max_vocab=args.max_vocab,
    )

    device = torch.device(args.device)
    model = BERTEncoder(
        vocab_size=len(vocab),
        num_classes=num_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_feedforward=args.ff,
        dropout=args.dropout,
        max_len=args.max_len,
        pad_id=vocab.pad_idx(),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best, bad = 0.0, 0
    patience = 2

    logger = ContinuousLogger(Path("results_run_bert_encoder"), "run_bert_encoder", "train")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        total = 0
        for ids, labels in train_loader:
            ids = ids.to(device)
            labels = labels.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(ids)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss += float(loss.item()) * labels.size(0)
            total += int(labels.size(0))

        val_acc = evaluate(model, val_loader, device)
        msg = f"Epoch {epoch:03d}: train_loss={train_loss / max(total, 1):.4f} val_acc={val_acc:.4f}"
        print(msg)
        logger.log_console(msg)
        logger.log_epoch_stats({"epoch": epoch, "train_loss": train_loss / max(total, 1), "val_acc": val_acc})

        if val_acc > best + 1e-6:
            best = val_acc
            bad = 0
            torch.save({"model": model.state_dict(), "vocab": vocab}, "BERTEncoder_best.pth")
        else:
            bad += 1
            if bad >= patience:
                break

    test_acc = evaluate(model, test_loader, device)
    print(f"Done. Best val acc: {best:.4f}, test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
