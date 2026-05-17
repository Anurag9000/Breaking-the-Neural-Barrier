import argparse
import os
import random

import torch
from torch.utils.data import DataLoader

from nlp_ssl_stl import MLPTextSSL
from utils.text_benchmarks import make_ag_news_ssl_loaders


def evaluate(model, val_loader, device, temperature):
    model.eval()
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for v1, v2 in val_loader:
            (i1, l1), (i2, l2) = v1, v2
            i1, l1, i2, l2 = i1.to(device), l1.to(device), i2.to(device), l2.to(device)
            loss = model((i1, l1), (i2, l2), temperature=temperature)
            loss_sum += loss.item() * i1.size(0)
            n += i1.size(0)
    return loss_sum / max(n, 1)


def main():
    p = argparse.ArgumentParser(description="AG News contrastive text SSL")
    p.add_argument("--emb_dim", type=int, default=256)
    p.add_argument("--hidden", type=int, nargs="+", default=[1024, 512])
    p.add_argument("--rep_dim", type=int, default=256)
    p.add_argument("--proj_dim", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--min_freq", type=int, default=2)
    p.add_argument("--max_size", type=int, default=50000)
    p.add_argument("--word_dropout", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--val_fraction", type=float, default=0.1)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trl, val, test, vocab = make_ag_news_ssl_loaders(
        batch_size=args.batch_size,
        max_len=args.max_len,
        seed=args.seed,
        val_fraction=args.val_fraction,
        word_dropout=args.word_dropout,
        min_freq=args.min_freq,
        max_vocab=args.max_size,
    )

    model = MLPTextSSL(len(vocab), args.emb_dim, args.hidden, args.rep_dim, args.proj_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val, best_state, bad = float("inf"), None, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        tr_sum, tr_n = 0.0, 0
        for v1, v2 in trl:
            (i1, l1), (i2, l2) = v1, v2
            i1, l1, i2, l2 = i1.to(device), l1.to(device), i2.to(device), l2.to(device)
            loss = model((i1, l1), (i2, l2), temperature=args.temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_sum += loss.item() * i1.size(0)
            tr_n += i1.size(0)
        tr_loss = tr_sum / max(tr_n, 1)

        va_loss = evaluate(model, val, device, args.temperature)
        if va_loss < best_val:
            best_val, best_state, bad = va_loss, {k: v.detach().cpu() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        print(f"[{ep:03d}] train_ssl={tr_loss:.6f} val_ssl={va_loss:.6f} | hidden={args.hidden} rep={args.rep_dim} proj={args.proj_dim}")
        if bad >= args.patience:
            break

    if best_state is not None:
        os.makedirs("checkpoints", exist_ok=True)
        path = os.path.join("checkpoints", "nlp_ssl_stl.pt")
        torch.save({"model": best_state, "val_ssl": best_val, "config": vars(args)}, path)
        print(f"Saved best checkpoint to: {path} (val_ssl={best_val:.6f})")


if __name__ == "__main__":
    main()
