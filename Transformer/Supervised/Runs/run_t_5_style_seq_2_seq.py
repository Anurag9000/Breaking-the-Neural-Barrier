import argparse, math, random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from model_t5_style import T5Style

class TSVSeq2Seq(Dataset):
    def __init__(self, path: Path, src_vocab: dict, tgt_vocab: dict, max_len: int = 128):
        self.pairs = []; self.src_vocab = src_vocab; self.tgt_vocab=tgt_vocab; self.max_len=max_len
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip().split('\t')
                if len(s)!=2: continue
                self.pairs.append((s[0], s[1]))
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i): return self.pairs[i]

def build_vocab(lines):
    from collections import Counter
    cnt = Counter(); [cnt.update(l.split()) for l in lines]
    vocab = {"<pad>":0,"<unk>":1,"<bos>":2,"<eos>":3}
    for t,c in cnt.items():
        if t not in vocab: vocab[t]=len(vocab)
    return vocab

def collate(batch, src_vocab, tgt_vocab, max_len):
    src_ids, tgt_in_ids, tgt_out_ids = [], [], []
    for src, tgt in batch:
        s = [src_vocab.get(t,src_vocab['<unk>']) for t in src.split()][:max_len]
        t = [tgt_vocab['<bos>']] + [tgt_vocab.get(t,tgt_vocab['<unk>']) for t in tgt.split()] + [tgt_vocab['<eos>']]
        t = t[:max_len]
        src_ids.append(s); tgt_in_ids.append(t[:-1]); tgt_out_ids.append(t[1:])
    S = max(1,max(len(s) for s in src_ids)); T = max(1,max(len(t) for t in tgt_in_ids))
    sp, tp = src_vocab['<pad>'], tgt_vocab['<pad>']
    pad = lambda x,L,p: x+[p]*(L-len(x))
    src = torch.tensor([pad(s,S,sp) for s in src_ids]); tgt_in = torch.tensor([pad(t,T,tp) for t in tgt_in_ids]); tgt_out = torch.tensor([pad(t,T,tp) for t in tgt_out_ids])
    return src, (src==sp), tgt_in, (tgt_in==tp), tgt_out

def evaluate(model, loader, crit, device, pad_id):
    model.eval(); tot=den=0
    with torch.no_grad():
        for src, sp, tgt_in, tp, tgt_out in loader:
            src, sp, tgt_in, tp, tgt_out = src.to(device), sp.to(device), tgt_in.to(device), tp.to(device), tgt_out.to(device)
            logits = model(src, sp, tgt_in, tp)
            loss = crit(logits.view(-1, logits.size(-1)), tgt_out.view(-1))
            m = (tgt_out.view(-1)!=pad_id)
            tot += loss.item()*m.sum().item(); den += m.sum().item()
    return tot/max(den,1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_tsv', type=str)
    ap.add_argument('--val_tsv', type=str)
    ap.add_argument('--d_model', type=int, default=256)
    ap.add_argument('--nhead', type=int, default=8)
    ap.add_argument('--enc_layers', type=int, default=6)
    ap.add_argument('--dec_layers', type=int, default=6)
    ap.add_argument('--ff', type=int, default=1024)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--max_len', type=int, default=128)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    if args.train_tsv and Path(args.train_tsv).exists():
        lines = []
        with open(args.train_tsv,'r',encoding='utf-8') as f:
            for line in f:
                s=line.strip().split('\t');
                if len(s)==2: lines.extend([s[0],s[1]])
    else:
        lines = ["translate English to Spanish: one two three", "uno dos tres"]*500
    src_vocab = build_vocab([l for i,l in enumerate(lines) if i%2==0])
    tgt_vocab = build_vocab([l for i,l in enumerate(lines) if i%2==1])

    if not (args.train_tsv and Path(args.train_tsv).exists()):
        tmp=Path('train_t5.tsv'); tmp.write_text('\n'.join(['translate English to Spanish: one two three\tuno dos tres']*600), encoding='utf-8')
        args.train_tsv=str(tmp)
    if not (args.val_tsv and Path(args.val_tsv).exists()):
        tmpv=Path('val_t5.tsv'); tmpv.write_text('\n'.join(['translate English to Spanish: one two\tuno dos']*150), encoding='utf-8')
        args.val_tsv=str(tmpv)

    from functools import partial
    def loader(path):
        ds = TSVSeq2Seq(Path(path), src_vocab, tgt_vocab, args.max_len)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=('train' in path), collate_fn=partial(collate, src_vocab=src_vocab, tgt_vocab=tgt_vocab, max_len=args.max_len))
    train_loader = loader(args.train_tsv); val_loader = loader(args.val_tsv)

    model = T5Style(src_vocab=len(src_vocab), tgt_vocab=len(tgt_vocab), d_model=args.d_model, nhead=args.nhead, enc_layers=args.enc_layers, dec_layers=args.dec_layers, ff=args.ff, dropout=args.dropout, share_embeddings=False, max_len=args.max_len).to(args.device)
    crit = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>']); opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best, bad=1e9,0
    for epoch in range(1, args.epochs+1):
        model.train()
        for src, sp, tgt_in, tp, tgt_out in train_loader:
            src, sp, tgt_in, tp, tgt_out = src.to(args.device), sp.to(args.device), tgt_in.to(args.device), tp.to(args.device), tgt_out.to(args.device)
            opt.zero_grad(); logits = model(src, sp, tgt_in, tp); loss = crit(logits.view(-1, logits.size(-1)), tgt_out.view(-1)); loss.backward();
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        val = evaluate(model, val_loader, crit, args.device, tgt_vocab['<pad>'])
        ppl = math.exp(val)
        print(f"Epoch {epoch}: val_ppl={ppl:.3f}")
        if val + 1e-6 < best:
            best=val; bad=0; torch.save({'model': model.state_dict(),'src_vocab':src_vocab,'tgt_vocab':tgt_vocab}, 'T5Style_best.pth')
        else:
            bad+=1
            if bad>=args.patience:
                print('Early stopping.'); break
    print('Done. Best val loss:', best)

if __name__ == '__main__':
    main()
