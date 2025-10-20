import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model_bert_encoder import BERTEncoder

# Simple text classification dataset (TSV: "label\ttext") with whitespace tokenization
class TSVTextCls(Dataset):
    def __init__(self, path: Path, vocab: dict, max_len: int = 256):
        self.samples = []
        self.vocab = vocab
        self.max_len = max_len
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.rstrip('\n').split('\t')
                if len(s) < 2:
                    continue
                label = int(s[0])
                text = s[1]
                self.samples.append((label, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def build_vocab(paths, min_freq=1):
    from collections import Counter
    cnt = Counter()
    for p in paths:
        if not Path(p).exists():
            continue
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.rstrip('\n').split('\t')
                if len(s) < 2:
                    continue
                text = s[1]
                cnt.update(text.split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, c in cnt.items():
        if c >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def collate(batch, vocab: dict, max_len: int):
    labels, ids = [], []
    for label, text in batch:
        labels.append(label)
        toks = [vocab.get(tok, vocab['<unk>']) for tok in text.split()]
        toks = toks[:max_len]
        ids.append(toks)
    S = max(1, max(len(x) for x in ids))
    pad = vocab['<pad>']
    ids = torch.tensor([x + [pad]*(S-len(x)) for x in ids], dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    return ids, labels


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for ids, labels in loader:
            ids, labels = ids.to(device), labels.to(device)
            logits = model(ids)
            pred = logits.argmax(-1)
            correct += (pred == labels).sum().item()
            total += labels.numel()
    return correct / max(total, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train_tsv', type=str)
    p.add_argument('--val_tsv', type=str)
    p.add_argument('--num_classes', type=int, default=2)
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--nhead', type=int, default=8)
    p.add_argument('--layers', type=int, default=6)
    p.add_argument('--ff', type=int, default=1024)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--max_len', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--patience', type=int, default=2)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    # Build vocabulary
    vocab = build_vocab([args.train_tsv, args.val_tsv]) if args.train_tsv else {"<pad>": 0, "<unk>": 1, "hello": 2, "world": 3}

    # Datasets (fallback synthetic if TSVs not provided)
    if args.train_tsv and Path(args.train_tsv).exists():
        train_ds = TSVTextCls(Path(args.train_tsv), vocab, args.max_len)
    else:
        tmp = Path('train_text.tsv')
        tmp.write_text('\n'.join(['0\thello world']*500 + ['1\tworld hello']*500), encoding='utf-8')
        train_ds = TSVTextCls(tmp, vocab, args.max_len)

    if args.val_tsv and Path(args.val_tsv).exists():
        val_ds = TSVTextCls(Path(args.val_tsv), vocab, args.max_len)
    else:
        tmpv = Path('val_text.tsv')
        tmpv.write_text('\n'.join(['0\thello world']*100 + ['1\tworld hello']*100), encoding='utf-8')
        val_ds = TSVTextCls(tmpv, vocab, args.max_len)

    coll = lambda b: collate(b, vocab, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=coll)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll)

    model = BERTEncoder(vocab_size=len(vocab), num_classes=args.num_classes, d_model=args.d_model,
                        nhead=args.nhead, num_layers=args.layers, dim_feedforward=args.ff,
                        dropout=args.dropout, max_len=args.max_len, pad_id=0).to(args.device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best, bad = 0.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for ids, labels in train_loader:
            ids, labels = ids.to(args.device), labels.to(args.device)
            optim.zero_grad()
            logits = model(ids)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        acc = evaluate(model, val_loader, args.device)
        print(f"Epoch {epoch}: val_acc={acc:.4f}")
        if acc > best + 1e-6:
            best = acc
            bad = 0
            torch.save({'model': model.state_dict(), 'vocab': vocab}, 'BERTEncoder_best.pth')
        else:
            bad += 1
            if bad >= args.patience:
                print('Early stopping.')
                break
    print('Done. Best val acc:', best)

if __name__ == '__main__':
    main()
