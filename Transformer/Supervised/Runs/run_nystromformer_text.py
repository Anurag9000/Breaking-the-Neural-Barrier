import argparse
from pathlib import Path

import torch
import torch.nn as nn

from utils.adp_logging import ContinuousLogger
from utils.text_benchmarks import make_ag_news_classification_loaders

from model_nystromformer import NystromformerEncoder


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for ids, y in loader:
            ids = ids.to(device)
            y = y.to(device)
            logits = model(ids)
            pred = logits.argmax(-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser(description="Nystromformer text classifier on AG News")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_fraction", type=float, default=0.1)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--max_vocab", type=int, default=50000)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    train_loader, val_loader, test_loader, vocab, num_classes = make_ag_news_classification_loaders(
        batch_size=args.batch_size,
        max_len=args.max_len,
        seed=args.seed,
        val_fraction=args.val_fraction,
        min_freq=args.min_freq,
        max_vocab=args.max_vocab,
    )

    model = NystromformerEncoder(vocab=len(vocab), num_classes=num_classes, dim=256, depth=6, heads=8, landmarks=64).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best, bad, patience = 0.0, 0, 4
    logger = ContinuousLogger(Path("results_run_nystromformer_text"), "run_nystromformer_text", "train")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        total = 0
        for ids, y in train_loader:
            ids = ids.to(args.device)
            y = y.to(args.device)
            opt.zero_grad(set_to_none=True)
            logits = model(ids)
            loss = crit(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += float(loss.item()) * y.size(0)
            total += int(y.size(0))

        acc = evaluate(model, val_loader, args.device)
        msg = f"Epoch {epoch:03d}: train_loss={train_loss / max(total, 1):.4f} val_acc={acc:.4f}"
        print(msg)
        logger.log_console(msg)
        logger.log_epoch_stats({"epoch": epoch, "train_loss": train_loss / max(total, 1), "val_acc": acc})

        if acc > best + 1e-6:
            best = acc
            bad = 0
            torch.save({"model": model.state_dict()}, "Nystromformer_best.pth")
        else:
            bad += 1
            if bad >= patience:
                break

    test_acc = evaluate(model, test_loader, args.device)
    print(f"Done. Best val acc: {best:.4f}, test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
