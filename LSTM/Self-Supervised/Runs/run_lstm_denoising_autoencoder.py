import argparse
import random
from typing import List

import torch
from torch.utils.data import DataLoader

from lstm_denoising_autoencoder import (
    WhitespaceTokenizer,
    DenoiseDataset,
    pad_collate,
    LSTMDenoisingAutoencoder,
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
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--emb_dim", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--enc_layers", type=int, default=1)
    ap.add_argument("--dec_layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max_epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--mask_prob", type=float, default=0.15)
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

    def to_ids(blocks: List[str]):
        return [tok.encode_with_bos_eos(s) for s in blocks]

    train_ids = to_ids(tr)
    val_ids = to_ids(va)
    test_ids = to_ids(te)

    train_ds = DenoiseDataset(train_ids, max_len=args.max_len, mask_prob=args.mask_prob, seed=args.seed)
    val_ds = DenoiseDataset(val_ids, max_len=args.max_len, mask_prob=args.mask_prob, seed=args.seed)
    test_ds = DenoiseDataset(test_ids, max_len=args.max_len, mask_prob=args.mask_prob, seed=args.seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)

    cfg = TrainConfig(
        emb_dim=args.emb_dim,
        hidden=args.hidden,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        dropout=args.dropout,
        lr=args.lr,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        mask_prob=args.mask_prob,
    )

    model = LSTMDenoisingAutoencoder(vocab=len(tok.itos), emb_dim=cfg.emb_dim, hidden=cfg.hidden, enc_layers=cfg.enc_layers, dec_layers=cfg.dec_layers, dropout=cfg.dropout).to(device)
    best = train(model, train_loader, val_loader, cfg, device)
    test_loss, _ = evaluate(model, test_loader, device)
    print({
        "vocab": len(tok.itos),
        "best_val_loss": round(best, 4),
        "test_loss": round(test_loss, 4),
        "params": sum(p.numel() for p in model.parameters()),
    })
