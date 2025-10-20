import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model_causal_transformer import CausalTransformer

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
                label = int(s[0]); text = s[1]
                self.samples.append((label, text))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]

def build_vocab(paths, min_freq=1):
    from collections import Counter
    cnt = Counter()
    for p in paths:
        if not p or not Path(p).exists():
            continue
        with open(p, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.rstrip('\n').split('\t')
                if len(s) < 2: continue
                cnt.update(s[1].split())
    vocab = {"<pad>":0, "<unk>":1}
    for tok, c in cnt.items():
        if c >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)
    if len(vocab) == 2:
        vocab.update({"hello":2, "world":3})
    return vocab

def collate(batch, vocab, max_len):
    labels, ids = [], []
    for y, text in batch:
        labels.append(y)
        toks = [vocab.get(t, vocab['<unk>']) for t in text.split()][:max_len]
        ids.append(toks)
    S = max(1, max(len(x) for x in ids))
    pad = vocab['<pad>']
    ids = torch.tensor([x + [pad]*(S-len(x)) for x in ids], dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    return ids, labels

def evaluate(model, loader, device):
    model.eval(); correct=total=0
    with torch.no_grad():
        for ids, labels in loader:
            ids, labels = ids.to(device), labels.to(device)
            logits = model(ids)
            pred = logits.argmax(-1)
            correct += (pred==labels).sum().item(); total += labels.numel()
    return correct/max(total,1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_tsv', type=str)
    ap.add_argument('--val_tsv', type=str)
    ap.add_argument('--num_classes', type=int, default=2)
    ap.add_argument('--d_model', type=int, default=256)
    ap.add_argument('--nhead', type=int, default=8)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--ff', type=int, default=1024)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--max_len', type=int, default=256)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--patience', type=int, default=2)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    vocab = build_vocab([args.train_tsv, args.val_tsv])
    if args.train_tsv and Path(args.train_tsv).exists():
        train_ds = TSVTextCls(Path(args.train_tsv), vocab, args.max_len)
    else:
        tmp = Path('train_ct.tsv'); tmp.write_text('\n'.join(['0\thello world']*500 + ['1\tworld hello']*500), encoding='utf-8')
        train_ds = TSVTextCls(tmp, vocab, args.max_len)
    if args.val_tsv and Path(args.val_tsv).exists():
        val_ds = TSVTextCls(Path(args.val_tsv), vocab, args.max_len)
    else:
        tmpv = Path('val_ct.tsv'); tmpv.write_text('\n'.join(['0\thello world']*100 + ['1\tworld hello']*100), encoding='utf-8')
        val_ds = TSVTextCls(tmpv, vocab, args.max_len)

    coll = lambda b: collate(b, vocab, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=coll)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll)

    model = CausalTransformer(vocab_size=len(vocab), num_classes=args.num_classes, d_model=args.d_model,
                              nhead=args.nhead, num_layers=args.layers, dim_ff=args.ff,
                              dropout=args.dropout, max_len=args.max_len, pad_id=0).to(args.device)

    crit = nn.CrossEntropyLoss(); opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best, bad = 0.0, 0
    for epoch in range(1, args.epochs+1):
        model.train()
        for ids, labels in train_loader:
            ids, labels = ids.to(args.device), labels.to(args.device)
            opt.zero_grad(); logits = model(ids); loss = crit(logits, labels); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        acc = evaluate(model, val_loader, args.device)
        print(f"Epoch {epoch}: val_acc={acc:.4f}")
        if acc > best + 1e-6:
            best = acc; bad = 0
            torch.save({'model': model.state_dict(), 'vocab': vocab}, 'CausalTransformer_best.pth')
        else:
            bad += 1
            if bad >= args.patience:
                print('Early stopping.'); break
    print('Done. Best val acc:', best)

if __name__ == '__main__':
    main()
