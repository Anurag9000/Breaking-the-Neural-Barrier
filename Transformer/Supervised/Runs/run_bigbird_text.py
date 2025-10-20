import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model_bigbird_encoder import BigBirdEncoder

class TSVLongText(Dataset):
    def __init__(self, path: Path, vocab: dict, max_len: int = 2048):
        self.samples=[]; self.vocab=vocab; self.max_len=max_len
        for line in open(path,'r',encoding='utf-8'):
            s=line.rstrip('\n').split('\t');
            if len(s)<2: continue
            self.samples.append((int(s[0]), s[1]))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]

def build_vocab(paths):
    from collections import Counter
    cnt=Counter()
    for p in paths:
        if not p or not Path(p).exists(): continue
        for line in open(p,'r',encoding='utf-8'):
            s=line.rstrip('\n').split('\t');
            if len(s)<2: continue
            cnt.update(s[1].split())
    vocab={"<pad>":0,"<unk>":1,"<cls>":2}
    for t,_ in cnt.items():
        if t not in vocab: vocab[t]=len(vocab)
    return vocab

def collate(batch, vocab, max_len):
    labels, ids = [], []
    for y, text in batch:
        labels.append(y)
        toks=[vocab['<cls>']] + [vocab.get(t, vocab['<unk>']) for t in text.split()]
        ids.append(toks[:max_len])
    S = max(1, max(len(x) for x in ids)); pad=vocab['<pad>']
    ids = torch.tensor([x+[pad]*(S-len(x)) for x in ids]); labels=torch.tensor(labels)
    return ids, labels

def evaluate(model, loader, device):
    model.eval(); correct=total=0
    with torch.no_grad():
        for ids, y in loader:
            ids,y = ids.to(device), y.to(device)
            logits = model(ids)
            pred = logits.argmax(-1)
            correct += (pred==y).sum().item(); total += y.numel()
    return correct/max(total,1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_tsv', type=str)
    ap.add_argument('--val_tsv', type=str)
    ap.add_argument('--num_classes', type=int, default=2)
    ap.add_argument('--d_model', type=int, default=256)
    ap.add_argument('--nhead', type=int, default=8)
    ap.add_argument('--layers', type=int, default=6)
    ap.add_argument('--block', type=int, default=32)
    ap.add_argument('--max_len', type=int, default=2048)
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--patience', type=int, default=2)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    if args.train_tsv and Path(args.train_tsv).exists():
        vocab = build_vocab([args.train_tsv, args.val_tsv])
    else:
        vocab = {"<pad>":0,"<unk>":1,"<cls>":2,"dolor":3,"amet":4}
        tmp=Path('train_bb.tsv'); tmp.write_text('\n'.join(['0\t'+'dolor '*1500]*100 + ['1\t'+'amet '*1500]*100), encoding='utf-8'); args.train_tsv=str(tmp)
        tmpv=Path('val_bb.tsv'); tmpv.write_text('\n'.join(['0\t'+'dolor '*1000]*20 + ['1\t'+'amet '*1000]*20), encoding='utf-8'); args.val_tsv=str(tmpv)

    from functools import partial
    train_loader = DataLoader(TSVLongText(Path(args.train_tsv), vocab, args.max_len), batch_size=args.batch_size, shuffle=True, collate_fn=partial(collate, vocab=vocab, max_len=args.max_len))
    val_loader = DataLoader(TSVLongText(Path(args.val_tsv), vocab, args.max_len), batch_size=args.batch_size, shuffle=False, collate_fn=partial(collate, vocab=vocab, max_len=args.max_len))

    model = BigBirdEncoder(vocab=len(vocab), num_classes=args.num_classes, d_model=args.d_model, nhead=args.nhead, layers=args.layers, block=args.block, max_len=args.max_len).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr); crit = nn.CrossEntropyLoss()

    best, bad=0.0,0
    for epoch in range(1, args.epochs+1):
        model.train()
        for ids, y in train_loader:
            ids,y = ids.to(args.device), y.to(args.device)
            opt.zero_grad(); logits = model(ids); loss = crit(logits, y); loss.backward();
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        acc = evaluate(model, val_loader, args.device)
        print(f"Epoch {epoch}: val_acc={acc:.4f}")
        if acc > best + 1e-6:
            best=acc; bad=0; torch.save({'model': model.state_dict(),'vocab':vocab}, 'BigBirdEncoder_best.pth')
        else:
            bad+=1
            if bad>=args.patience:
                print('Early stopping.'); break
    print('Done. Best val acc:', best)

if __name__ == '__main__':
    main()
