import argparse
import random
from typing import List

import torch
from torch.utils.data import DataLoader

from lstm_ssl_bilm import WhitespaceTokenizer, LMDataset, BiLSTMLM, TrainConfig, train, evaluate


def read_corpus(paths: List[str]):
    texts = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
    return texts


def make_splits(texts: List[str], val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 1337):
    rng = random.Random(seed)
    rng.shuffle(texts)
    n = len(texts)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    test = texts[:n_test]
    val = texts[n_test:n_test + n_val]
    train = texts[n_test + n_val:]
    return train, val, test


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, nargs="+", required=True)
    ap.add_argument("--val", type=str, nargs="*")
    ap.add_argument("--test", type=str, nargs="*")
    ap.add_argument("--bptt", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    texts = read_corpus(args.train)
    if args.val and args.test:
        val_texts = read_corpus(args.val)
        test_texts = read_corpus(args.test)
        tr, va, te = texts, val_texts, test_texts
    else:
        tr, va, te = make_splits(texts, 0.1, 0.1, seed=args.seed)

    tok = WhitespaceTokenizer(min_freq=1)
    tok.build_vocab(tr)

    def flatten_lines(lines):
        ids = []
        for s in lines:
            ids.extend(tok.encode(s))
        return ids

    train_ids = flatten_lines(tr)
    val_ids = flatten_lines(va)
    test_ids = flatten_lines(te)

    train_ds = LMDataset(train_ids, bptt=args.bptt)
    val_ds = LMDataset(val_ids, bptt=args.bptt)
    test_ds = LMDataset(test_ids, bptt=args.bptt)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    cfg = TrainConfig(
        emb_dim=args.emb_dim,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        lr=args.lr,
        bptt=args.bptt,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )

    model = BiLSTMLM(vocab=len(tok.itos), emb_dim=cfg.emb_dim, hidden=cfg.hidden, layers=cfg.layers, dropout=cfg.dropout).to(device)
    best_val = train(model, train_loader, val_loader, cfg, device)
    test_loss, test_ppl = evaluate(model, test_loader, device)
    print({
        "vocab": len(tok.itos),
        "best_val_loss": round(best_val, 4),
        "test_loss": round(test_loss, 4),
        "test_ppl": round(test_ppl, 2),
        "params": sum(p.numel() for p in model.parameters()),
    })
