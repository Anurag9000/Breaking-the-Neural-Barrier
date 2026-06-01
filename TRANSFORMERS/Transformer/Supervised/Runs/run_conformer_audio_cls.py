import argparse
from pathlib import Path

import torch
import torch.nn as nn

from utils.adp_logging import ContinuousLogger
from utils.audio_benchmarks import make_speechcommands_loaders

from model_conformer import ConformerEncoder


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for feats, y in loader:
            feats = feats.to(device)
            y = y.to(device)
            logits = model(feats)
            pred = logits.argmax(-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser(description="Conformer audio classification on SpeechCommands")
    ap.add_argument("--data-root", type=str, default="./data/SpeechCommands")
    ap.add_argument("--download", action="store_true", help="Download SpeechCommands if missing")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n_mfcc", type=int, default=80)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    train_loader, val_loader, test_loader, num_classes = make_speechcommands_loaders(
        root=args.data_root,
        batch_size=args.batch_size,
        download=args.download,
        n_mfcc=args.n_mfcc,
    )

    device = torch.device(args.device)
    model = ConformerEncoder(
        in_feats=args.n_mfcc,
        num_classes=num_classes,
        d_model=args.d_model,
        nhead=args.nhead,
        layers=args.layers,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best_val = 0.0
    best_epoch = 0
    bad = 0
    patience = 4

    logger = ContinuousLogger(Path("results_run_conformer_audio_cls"), "run_conformer_audio_cls", "train")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        total = 0
        for feats, y in train_loader:
            feats = feats.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(feats)
            loss = crit(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += float(loss.item()) * y.size(0)
            total += int(y.size(0))

        val_acc = evaluate(model, val_loader, device)
        msg = f"Epoch {epoch:03d}: train_loss={train_loss / max(total, 1):.4f} val_acc={val_acc:.4f}"
        print(msg)
        logger.log_console(msg)
        logger.log_epoch_stats({"epoch": epoch, "train_loss": train_loss / max(total, 1), "val_acc": val_acc})

        if val_acc > best_val + 1e-6:
            best_val = val_acc
            best_epoch = epoch
            bad = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_acc": val_acc}, "Conformer_best.pth")
        else:
            bad += 1
            if bad >= patience:
                break

    test_acc = evaluate(model, test_loader, device)
    print(f"Done. Best val acc: {best_val:.4f} @ epoch {best_epoch}, test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
