import argparse, math, random
from pathlib import Path
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger.nn as nn
from torch.utils.data import Dataset, DataLoader

from model_bart_style import BARTStyle

class TSVSeq2Seq(Dataset):
    def __init__(self, path: Path, vocab: dict, max_len: int = 128):
        self.pairs=[]; self.vocab=vocab; self.max_len=max_len
        for line in open(path,'r',encoding='utf-8'):
            s=line.strip().split('\t');
            if len(s)==2: self.pairs.append((s[0],s[1]))
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i): return self.pairs[i]

def build_vocab(lines):
    from collections import Counter
    cnt=Counter(); [cnt.update(l.split()) for l in lines]
    vocab={"<pad>":0,"<unk>":1,"<bos>":2,"<eos>":3}
    for t,_ in cnt.items():
        if t not in vocab: vocab[t]=len(vocab)
    return vocab

def collate(batch, vocab, max_len):
    src_ids, tgt_in_ids, tgt_out_ids = [], [], []
    for src,tgt in batch:
        s = [vocab.get(t,vocab['<unk>']) for t in src.split()][:max_len]
        t = [vocab['<bos>']] + [vocab.get(t,vocab['<unk>']) for t in tgt.split()] + [vocab['<eos>']]
        t = t[:max_len]
        src_ids.append(s); tgt_in_ids.append(t[:-1]); tgt_out_ids.append(t[1:])
    S = max(1,max(len(x) for x in src_ids)); T = max(1,max(len(x) for x in tgt_in_ids))
    pad=vocab['<pad>']; pad_to=lambda x,L: x+[pad]*(L-len(x))
    src=torch.tensor([pad_to(s,S) for s in src_ids]); tgt_in=torch.tensor([pad_to(t,T) for t in tgt_in_ids]); tgt_out=torch.tensor([pad_to(t,T) for t in tgt_out_ids])
    return src, (src==pad), tgt_in, (tgt_in==pad), tgt_out

def eval(model, loader, crit, device, pad_id):
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
        lines=[]
        for line in open(args.train_tsv,'r',encoding='utf-8'):
            s=line.strip().split('\t');
            if len(s)==2: lines.extend([s[0],s[1]])
    else:
        lines=["summarize: lorem ipsum dolor sit amet" , "lorem ipsum"]*500
    vocab = build_vocab(lines)

    if not (args.train_tsv and Path(args.train_tsv).exists()):
        tmp=Path('train_bart.tsv'); tmp.write_text('\n'.join(['summarize: lorem ipsum dolor sit amet\tlorem ipsum']*600), encoding='utf-8'); args.train_tsv=str(tmp)
    if not (args.val_tsv and Path(args.val_tsv).exists()):
        tmpv=Path('val_bart.tsv'); tmpv.write_text('\n'.join(['summarize: lorem ipsum dolor\tlorem ipsum']*150), encoding='utf-8'); args.val_tsv=str(tmpv)

    from functools import partial
    train_loader = DataLoader(TSVSeq2Seq(Path(args.train_tsv), vocab, args.max_len), batch_size=args.batch_size, shuffle=True, collate_fn=partial(collate, vocab=vocab, max_len=args.max_len))
    val_loader = DataLoader(TSVSeq2Seq(Path(args.val_tsv), vocab, args.max_len), batch_size=args.batch_size, shuffle=False, collate_fn=partial(collate, vocab=vocab, max_len=args.max_len))

    model = BARTStyle(vocab=len(vocab), d_model=args.d_model, nhead=args.nhead, enc_layers=args.enc_layers, dec_layers=args.dec_layers, ff=args.ff, dropout=args.dropout, max_len=args.max_len).to(args.device)
    crit = nn.CrossEntropyLoss(ignore_index=vocab['<pad>']); opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best, bad = 1e9, 0

    # Init Logger

    logger = ContinuousLogger(Path('results_run_bart_style_seq_2_seq'), 'run_bart_style_seq_2_seq', 'train')

    for epoch in range(1, args.epochs+1):
        model.train()
        for src, sp, tgt_in, tp, tgt_out in train_loader:
            src, sp, tgt_in, tp, tgt_out = src.to(args.device), sp.to(args.device), tgt_in.to(args.device), tp.to(args.device), tgt_out.to(args.device)
            opt.zero_grad(); logits = model(src, sp, tgt_in, tp); loss = crit(logits.view(-1, logits.size(-1)), tgt_out.view(-1)); loss.backward();
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        val = eval(model, val_loader, crit, args.device, vocab['<pad>'])
        ppl = math.exp(val); # Log
 msg = f"Epoch {epoch}: val_ppl={ppl:.3f}"
 logger.log_console(msg)
 logger.log_epoch_stats({
     "epoch": epoch,
     "val_loss": val_loss if 'val_loss' in locals() else (loss.item() if 'loss' in locals() else 0),
     "train_loss": loss.item() if 'loss' in locals() else 0
 })
        if val + 1e-6 < best:
            best=val; bad=0; torch.save({'model': model.state_dict(),'vocab':vocab}, 'BARTStyle_best.pth')
        else:
            bad+=1
            if bad>=args.patience:
                print('Early stopping.'); break
    print('Done. Best val loss:', best)

if __name__ == '__main__':
    main()