import argparse, os, random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple

from lstm_tagger_crf import LSTMCRFTagger, LSTMCRFConfig


class SyntheticTagging(Dataset):
    def __init__(self, n: int, vocab_size: int, min_len: int, max_len: int, num_tags: int, pad_idx: int = 0):
        rng = random.Random(123)
        self.samples = []
        for _ in range(n):
            L = rng.randint(min_len, max_len)
            x = [rng.randint(1, vocab_size - 1) for _ in range(L)]
            y = [t % num_tags for t in x]
            self.samples.append((x, y))
        self.pad_idx = pad_idx
        self.num_tags = num_tags
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        return self.samples[i]


def collate_pad(batch: List[Tuple[List[int], List[int]]], pad_idx: int):
    lengths = torch.tensor([len(x) for x, _ in batch], dtype=torch.long)
    max_len = int(lengths.max())
    tokens = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
    tags = torch.full((len(batch), max_len), 0, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        L = len(x)
        tokens[i, :L] = torch.tensor(x, dtype=torch.long)
        tags[i, :L] = torch.tensor(y, dtype=torch.long)
    return tokens, lengths, tags


def token_acc(paths: List[List[int]], gold: torch.Tensor) -> float:
    # gold: (B,T) with padded positions zero; we compare only first L positions per path.
    correct = 0; total = 0
    for i, path in enumerate(paths):
        L = len(path)
        correct += int((torch.tensor(path) == gold[i, :L]).sum().item())
        total += L
    return correct / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vocab-size', type=int, default=20000)
    ap.add_argument('--num-tags', type=int, default=10)
    ap.add_argument('--min-len', type=int, default=10)
    ap.add_argument('--max-len', type=int, default=50)
    ap.add_argument('--train-n', type=int, default=8000)
    ap.add_argument('--val-n', type=int, default=1000)
    ap.add_argument('--test-n', type=int, default=1000)
    ap.add_argument('--emb-dim', type=int, default=128)
    ap.add_argument('--hidden-dim', type=int, default=256)
    ap.add_argument('--num-layers', type=int, default=1)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--pad-idx', type=int, default=0)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-2)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--max-epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--outdir', type=str, default='results_lstm')
    args = ap.parse_args()

    random.seed(123); torch.manual_seed(123)
    os.makedirs(args.outdir, exist_ok=True)

    train_ds = SyntheticTagging(args.train_n, args.vocab_size, args.min_len, args.max_len, args.num_tags, pad_idx=args.pad_idx)
    val_ds = SyntheticTagging(args.val_n, args.vocab_size, args.min_len, args.max_len, args.num_tags, pad_idx=args.pad_idx)
    test_ds = SyntheticTagging(args.test_n, args.vocab_size, args.min_len, args.max_len, args.num_tags, pad_idx=args.pad_idx)

    collate=lambda b: collate_pad(b, pad_idx=args.pad_idx)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,collate_fn=collate)
    val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,collate_fn=collate)
    test_loader=DataLoader(test_ds,batch_size=args.batch_size,shuffle=False,collate_fn=collate)

    cfg = LSTMCRFConfig(vocab_size=args.vocab_size, emb_dim=args.emb_dim, hidden_dim=args.hidden_dim,
                        num_layers=args.num_layers, dropout=args.dropout, num_tags=args.num_tags, pad_idx=args.pad_idx)
    model = LSTMCRFTagger(cfg).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    class EarlyStopper:
        def __init__(self, patience):
            self.patience=patience; self.best=float('inf'); self.bad=0; self.state=None
        def step(self, v, m):
            if v < self.best - 1e-7:
                self.best=v; self.bad=0; self.state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
            else:
                self.bad+=1
            return self.bad>=self.patience
        def restore(self, m):
            if self.state is not None: m.load_state_dict(self.state)

    es = EarlyStopper(args.patience)

    def run_epoch(loader, train: bool):
        tot=0.0; n=0
        if train: model.train()
        else: model.eval()
        with torch.set_grad_enabled(train):
            for tokens, lengths, tags in loader:
                tokens, lengths, tags = tokens.to(args.device), lengths.to(args.device), tags.to(args.device)
                emissions = model(tokens, lengths)
                loss = model.loss(emissions, tags, lengths)
                if train:
                    opt.zero_grad(set_to_none=True); loss.backward();
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                tot += float(loss.item()) * tokens.size(0); n += tokens.size(0)
        return tot/max(1,n)

    for epoch in range(1, args.max_epochs+1):
        tr = run_epoch(train_loader, True)
        vl = run_epoch(val_loader, False)
        stop = es.step(vl, model)
        print(f'Epoch {epoch:03d} | train_nll={tr:.4f} | val_nll={vl:.4f}')
        if stop:
            print('Early stopping.'); break

    es.restore(model)

    # Evaluate with Viterbi decoding
    def decode_acc(loader):
        acc_total=0.0; batches=0
        with torch.no_grad():
            for tokens, lengths, tags in loader:
                tokens, lengths, tags = tokens.to(args.device), lengths.to(args.device), tags.to(args.device)
                emissions = model(tokens, lengths)
                paths = model.decode(emissions, lengths)
                acc_total += token_acc(paths, tags.cpu())
                batches += 1
        return acc_total/max(1,batches)

    test_acc = decode_acc(test_loader)
    print(f'TEST | viterbi_token_acc={test_acc:.4f}')

    torch.save({'model_state': model.state_dict(), 'config': cfg.__dict__, 'test_token_acc': test_acc},
               os.path.join(args.outdir, 'lstm_tagger_crf.pt'))

if __name__ == '__main__':
    main()
