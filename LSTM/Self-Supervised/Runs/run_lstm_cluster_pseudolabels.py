import argparse
import random
from typing import List

import torch
from torch.utils.data import DataLoader

from lstm_cluster_pseudolabels import (
    WhitespaceTokenizer,
    SeqDataset,
    pad_collate,
    LSTMClust,
    TrainConfig,
    train,
    evaluate,
)


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
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--n_clusters", type=int, default=100)
    ap.add_argument("--kmeans_iters", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
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

    def to_ids(lines: List[str]):
        return [tok.encode(s) for s in lines]

    train_ids, val_ids, test_ids = to_ids(tr), to_ids(va), to_ids(te)

    train_ds = SeqDataset(train_ids, max_len=args.max_len)
    val_ds = SeqDataset(val_ids, max_len=args.max_len)
    test_ds = SeqDataset(test_ids, max_len=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)

    cfg = TrainConfig(
        emb_dim=args.emb_dim,
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        n_clusters=args.n_clusters,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        kmeans_iters=args.kmeans_iters,
    )

    model = LSTMClust(vocab=len(tok.itos), emb_dim=cfg.emb_dim, hidden=cfg.hidden, layers=cfg.layers, dropout=cfg.dropout, n_clusters=cfg.n_clusters).to(device)
    best = train(model, train_loader, val_loader, cfg, device)
    # proxy evaluation on test: k-means + CE like validation
    from lstm_cluster_pseudolabels import kmeans_on_loader
    pseudo_test, _ = kmeans_on_loader(model, test_loader, device, cfg.n_clusters, iters=max(1, cfg.kmeans_iters//2))
    # build test buf
    Xtest = []
    for x, _ in test_loader:
        Xtest.append(x)
    Xtest = torch.cat(Xtest, dim=0)
    class _Buf(torch.utils.data.Dataset):
        def __init__(self, X, y): self.X, self.y = X, y
        def __len__(self): return self.X.size(0)
        def __getitem__(self, i): return self.X[i], self.y[i]
    test_buf = _Buf(Xtest, pseudo_test)
    test_buf_loader = DataLoader(test_buf, batch_size=cfg.batch_size, shuffle=False)
    test_loss = evaluate(model, test_buf_loader, device)

    print({
        "vocab": len(tok.itos),
        "best_val_proxy_ce": round(best, 4),
        "test_proxy_ce": round(test_loss, 4),
        "params": sum(p.numel() for p in model.parameters()),
    })
