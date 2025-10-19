import argparse, os, random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

from lstm_mhsa_top_cls import LSTMMHSAClassifier, MHSAonLSTMConfig


class SyntheticTextCls(Dataset):
    def __init__(self, n: int, vocab_size: int, min_len: int, max_len: int, pad_idx: int = 0):
        rng = random.Random(939)
        self.samples=[]
        for _ in range(n):
            L=rng.randint(min_len, max_len)
            x=[rng.randint(1, vocab_size-1) for _ in range(L)]
            y=int(sum(x)%2==0)
            self.samples.append((x,y))
        self.pad_idx=pad_idx
    def __len__(self): return len(self.samples)
    def __getitem__(self,i): return self.samples[i]


def collate_pad(batch: List[Tuple[List[int], int]], pad_idx: int):
    lengths = torch.tensor([len(x) for x,_ in batch], dtype=torch.long)
    max_len=int(lengths.max())
    tokens=torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
    labels=torch.tensor([y for _,y in batch], dtype=torch.long)
    for i,(x,y) in enumerate(batch):
        tokens[i,:len(x)] = torch.tensor(x, dtype=torch.long)
    return tokens, lengths, labels


class EarlyStopper:
    def __init__(self, patience:int):
        self.p=patience; self.best=float('inf'); self.bad=0; self.state=None
    def step(self, v, m):
        if v < self.best - 1e-7:
            self.best=v; self.bad=0; self.state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        else: self.bad+=1
        return self.bad>=self.p
    def restore(self, m):
        if self.state is not None: m.load_state_dict(self.state)


def train_epoch(model, loader, opt, crit, device):
    model.train(); tot=0.0; n=0
    for tokens, lengths, labels in loader:
        tokens, lengths, labels = tokens.to(device), lengths.to(device), labels.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(tokens, lengths)
        loss = crit(logits, labels)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tot += float(loss.item()) * labels.size(0); n += labels.size(0)
    return tot/max(1,n)


def eval_epoch(model, loader, crit, device):
    model.eval(); tot=0.0; n=0; corr=0
    with torch.no_grad():
        for tokens, lengths, labels in loader:
            tokens, lengths, labels = tokens.to(device), lengths.to(device), labels.to(device)
            logits = model(tokens, lengths)
            loss = crit(logits, labels)
            tot += float(loss.item()) * labels.size(0); n += labels.size(0)
            corr += int((logits.argmax(-1)==labels).sum().item())
    return tot/max(1,n), corr/max(1,n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vocab-size', type=int, default=20000)
    ap.add_argument('--min-len', type=int, default=20)
    ap.add_argument('--max-len', type=int, default=80)
    ap.add_argument('--train-n', type=int, default=8000)
    ap.add_argument('--val-n', type=int, default=1000)
    ap.add_argument('--test-n', type=int, default=1000)

    ap.add_argument('--emb-dim', type=int, default=128)
    ap.add_argument('--hidden-dim', type=int, default=256)
    ap.add_argument('--num-layers', type=int, default=1)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--num-heads', type=int, default=4)
    ap.add_argument('--num-classes', type=int, default=2)
    ap.add_argument('--pad-idx', type=int, default=0)

    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-2)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--max-epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--outdir', type=str, default='results_lstm')
    args = ap.parse_args()

    random.seed(939); torch.manual_seed(939)
    os.makedirs(args.outdir, exist_ok=True)

    train_ds = SyntheticTextCls(args.train_n, args.vocab_size, args.min_len, args.max_len, pad_idx=args.pad_idx)
    val_ds = SyntheticTextCls(args.val_n, args.vocab_size, args.min_len, args.max_len, pad_idx=args.pad_idx)
    test_ds = SyntheticTextCls(args.test_n, args.vocab_size, args.min_len, args.max_len, pad_idx=args.pad_idx)

    collate=lambda b: collate_pad(b, pad_idx=args.pad_idx)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,collate_fn=collate)
    val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,collate_fn=collate)
    test_loader=DataLoader(test_ds,batch_size=args.batch_size,shuffle=False,collate_fn=collate)

    cfg = MHSAonLSTMConfig(vocab_size=args.vocab_size, emb_dim=args.emb_dim, hidden_dim=args.hidden_dim,
                           num_layers=args.num_layers, dropout=args.dropout, num_heads=args.num_heads,
                           num_classes=args.num_classes, pad_idx=args.pad_idx)
    model = LSTMMHSAClassifier(cfg).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()
    es = EarlyStopper(args.patience)

    for epoch in range(1, args.max_epochs+1):
        tr = train_epoch(model, train_loader, opt, crit, args.device)
        vl, vacc = eval_epoch(model, val_loader, crit, args.device)
        stop = es.step(vl, model)
        print(f'Epoch {epoch:03d} | train_loss={tr:.4f} | val_loss={vl:.4f} | val_acc={vacc:.4f}')
        if stop:
            print('Early stopping.'); break

    es.restore(model)
    tl, ta = eval_epoch(model, test_loader, crit, args.device)
    print(f'TEST | loss={tl:.4f} | acc={ta:.4f}')

    torch.save({'model_state': model.state_dict(), 'config': cfg.__dict__,
                'test_loss': tl, 'test_acc': ta}, os.path.join(args.outdir, 'lstm_mhsa_top_cls.pt'))

if __name__ == '__main__':
    main()
