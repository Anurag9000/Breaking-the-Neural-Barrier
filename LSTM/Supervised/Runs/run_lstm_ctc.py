import argparse, os, random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

from lstm_ctc import LSTMCTC, LSTMCTCConfig


class SyntheticCTC(Dataset):
    """Random inputs and shorter target strings (unaligned) for CTC training.
    Targets are integers in [1..num_labels]; 0 is reserved for blank.
    """
    def __init__(self, n: int, vocab_size: int, min_len: int, max_len: int, num_labels: int, pad_idx: int=0):
        rng = random.Random(2025)
        self.samples = []
        for _ in range(n):
            T = rng.randint(min_len, max_len)
            U = rng.randint(1, max(1, T//2))
            x = [rng.randint(1, vocab_size-1) for _ in range(T)]
            y = [rng.randint(1, num_labels) for _ in range(U)]
            self.samples.append((x, y))
        self.pad_idx = pad_idx
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        return self.samples[i]


def collate_ctc(batch: List[Tuple[List[int], List[int]]], pad_idx: int):
    x_lens = torch.tensor([len(x) for x,_ in batch], dtype=torch.long)
    max_T = int(x_lens.max())
    tokens = torch.full((len(batch), max_T), pad_idx, dtype=torch.long)
    for i,(x,y) in enumerate(batch):
        tokens[i,:len(x)] = torch.tensor(x, dtype=torch.long)
    # targets flattened for CTCLoss
    targets = [torch.tensor(y, dtype=torch.long) for _,y in batch]
    flat_targets = torch.cat(targets, dim=0)
    y_lens = torch.tensor([len(y) for _,y in batch], dtype=torch.long)
    return tokens, x_lens, flat_targets, y_lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vocab-size', type=int, default=64)
    ap.add_argument('--num-labels', type=int, default=20)
    ap.add_argument('--min-len', type=int, default=40)
    ap.add_argument('--max-len', type=int, default=80)
    ap.add_argument('--train-n', type=int, default=4000)
    ap.add_argument('--val-n', type=int, default=500)
    ap.add_argument('--test-n', type=int, default=500)

    ap.add_argument('--emb-dim', type=int, default=64)
    ap.add_argument('--hidden-dim', type=int, default=128)
    ap.add_argument('--num-layers', type=int, default=2)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--blank', type=int, default=0)
    ap.add_argument('--pad-idx', type=int, default=0)

    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-3)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--outdir', type=str, default='results_lstm')
    args = ap.parse_args()

    random.seed(2025); torch.manual_seed(2025)
    os.makedirs(args.outdir, exist_ok=True)

    train_ds = SyntheticCTC(args.train_n, args.vocab_size, args.min_len, args.max_len, args.num_labels, pad_idx=args.pad_idx)
    val_ds = SyntheticCTC(args.val_n, args.vocab_size, args.min_len, args.max_len, args.num_labels, pad_idx=args.pad_idx)
    test_ds = SyntheticCTC(args.test_n, args.vocab_size, args.min_len, args.max_len, args.num_labels, pad_idx=args.pad_idx)

    collate=lambda b: collate_ctc(b, pad_idx=args.pad_idx)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,collate_fn=collate)
    val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,collate_fn=collate)
    test_loader=DataLoader(test_ds,batch_size=args.batch_size,shuffle=False,collate_fn=collate)

    cfg = LSTMCTCConfig(vocab_size=args.vocab_size, emb_dim=args.emb_dim, hidden_dim=args.hidden_dim,
                        num_layers=args.num_layers, dropout=args.dropout, num_labels=args.num_labels,
                        blank=args.blank, pad_idx=args.pad_idx)
    model = LSTMCTC(cfg).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ctc = nn.CTCLoss(blank=args.blank, zero_infinity=True)

    class EarlyStopper:
        def __init__(self, p): self.p=p; self.best=float('inf'); self.bad=0; self.state=None
        def step(self, v, m):
            if v < self.best - 1e-7:
                self.best=v; self.bad=0; self.state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
            else:
                self.bad+=1
            return self.bad>=self.p
        def restore(self, m):
            if self.state is not None: m.load_state_dict(self.state)
    es = EarlyStopper(args.patience)

    def run(loader, train: bool):
        tot=0.0; n=0
        if train: model.train()
        else: model.eval()
        with torch.set_grad_enabled(train):
            for tokens, x_lens, flat_targets, y_lens in loader:
                tokens, x_lens = tokens.to(args.device), x_lens.to(args.device)
                flat_targets, y_lens = flat_targets.to(args.device), y_lens.to(args.device)
                logits_TBC = model(tokens, x_lens)  # (T,B,C)
                log_probs = logits_TBC.log_softmax(dim=-1)
                loss = ctc(log_probs, flat_targets, x_lens, y_lens)
                if train:
                    opt.zero_grad(set_to_none=True); loss.backward();
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                tot += float(loss.item()) * tokens.size(0); n += tokens.size(0)
        return tot/max(1,n)

    for epoch in range(1, args.max_epochs+1):
        tr = run(train_loader, True)
        vl = run(val_loader, False)
        stop = es.step(vl, model)
        print(f'Epoch {epoch:03d} | train_ctc={tr:.4f} | val_ctc={vl:.4f}')
        if stop:
            print('Early stopping.'); break

    es.restore(model)
    test = run(test_loader, False)
    print(f'TEST | ctc_loss={test:.4f}')

    torch.save({'model_state': model.state_dict(), 'config': cfg.__dict__, 'test_ctc': test},
               os.path.join(args.outdir, 'lstm_ctc.pt'))

if __name__ == '__main__':
    main()
