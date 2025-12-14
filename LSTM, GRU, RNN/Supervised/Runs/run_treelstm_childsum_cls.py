import argparse, os, random
from typing import List
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger.nn as nn
from torch.utils.data import Dataset, DataLoader

from treelstm_childsum_cls import ChildSumTreeLSTM, TreeLSTMConfig


class SyntheticTreeDataset(Dataset):
    """Generates random rooted trees with N nodes max; class is parity of sum of tokens."""
    def __init__(self, n_samples: int, max_nodes: int, vocab_size: int, seed: int = 909):
        rng = random.Random(seed)
        self.vocab_size = vocab_size
        self.samples = []
        for _ in range(n_samples):
            N = rng.randint(max_nodes//2, max_nodes)
            # build a random tree
            children = [[] for _ in range(N)]
            for v in range(1, N):
                p = rng.randrange(0, v)
                children[p].append(v)
            tokens = [rng.randint(1, vocab_size-1) for _ in range(N)]
            # pad to max_nodes for batching
            tokens_pad = tokens + [0]*(max_nodes - N)
            children_pad: List[List[int]] = []
            for u in range(max_nodes):
                if u < N:
                    children_pad.append(children[u])
                else:
                    children_pad.append([])
            label = int(sum(tokens) % 3 == 0)  # map to 0/1/2? keep 3 classes by mod 3 index
            label = sum(tokens) % 3
            self.samples.append((tokens_pad, children_pad, 0, label, N))  # root=0
        self.max_nodes = max_nodes

    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def collate_trees(batch):
    tokens = torch.tensor([b[0] for b in batch], dtype=torch.long)
    children = [b[1] for b in batch]
    roots = [b[2] for b in batch]
    labels = torch.tensor([b[3] for b in batch], dtype=torch.long)
    return tokens, children, roots, labels


def train_epoch(model, loader, opt, crit, device):
    model.train(); tot=0.0; n=0
    for tokens, children, roots, labels in loader:
        tokens, labels = tokens.to(device), labels.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(tokens, children, roots)
        loss = crit(logits, labels)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tot += float(loss.item()) * labels.size(0); n += labels.size(0)
    return tot/max(1,n)


def eval_epoch(model, loader, crit, device):
    model.eval(); tot=0.0; n=0; corr=0
    with torch.no_grad():
        for tokens, children, roots, labels in loader:
            tokens, labels = tokens.to(device), labels.to(device)
            logits = model(tokens, children, roots)
            loss = crit(logits, labels)
            tot += float(loss.item()) * labels.size(0); n += labels.size(0)
            corr += int((logits.argmax(-1)==labels).sum().item())
    return tot/max(1,n), corr/max(1,n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--vocab-size', type=int, default=5000)
    ap.add_argument('--max-nodes', type=int, default=32)
    ap.add_argument('--train-n', type=int, default=4000)
    ap.add_argument('--val-n', type=int, default=500)
    ap.add_argument('--test-n', type=int, default=500)

    ap.add_argument('--emb-dim', type=int, default=128)
    ap.add_argument('--hidden-dim', type=int, default=256)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--num-classes', type=int, default=3)

    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-2)
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--max-epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--outdir', type=str, default='results_lstm')
    args = ap.parse_args()

    random.seed(909); torch.manual_seed(909)
    os.makedirs(args.outdir, exist_ok=True)

    train_ds = SyntheticTreeDataset(args.train_n, args.max_nodes, args.vocab_size, seed=101)
    val_ds = SyntheticTreeDataset(args.val_n, args.max_nodes, args.vocab_size, seed=102)
    test_ds = SyntheticTreeDataset(args.test_n, args.max_nodes, args.vocab_size, seed=103)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_trees)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_trees)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_trees)

    cfg = TreeLSTMConfig(vocab_size=args.vocab_size, emb_dim=args.emb_dim, hidden_dim=args.hidden_dim,
                         dropout=args.dropout, num_classes=args.num_classes)
    model = ChildSumTreeLSTM(cfg).to(args.device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss()

    class EarlyStopper:
        def __init__(self, p): self.p=p; self.best=float('inf'); self.bad=0; self.state=None
        def step(self, v, m):
            if v < self.best - 1e-7:
                self.best=v; self.bad=0; self.state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
            else: self.bad+=1
            return self.bad>=self.p
        def restore(self, m):
            if self.state is not None: m.load_state_dict(self.state)

    es = EarlyStopper(args.patience)


    # Init Logger


    logger = ContinuousLogger(Path('results_run_treelstm_childsum_cls'), 'run_treelstm_childsum_cls', 'train')


    for epoch in range(1, args.max_epochs+1):
        tr = train_epoch(model, train_loader, opt, crit, args.device)
        vl, vacc = eval_epoch(model, val_loader, crit, args.device)
        stop = es.step(vl, model)
        # Log

        msg = f'Epoch {epoch:03d} | train_loss={tr:.4f} | val_loss={vl:.4f} | val_acc={vacc:.4f}'

        logger.log_console(msg)

        logger.log_epoch_stats({

            "epoch": epoch,

            "val_loss": val_loss if 'val_loss' in locals() else (loss.item() if 'loss' in locals() else 0),

            "train_loss": loss.item() if 'loss' in locals() else 0

        })
        if stop:
            print('Early stopping.'); break

    es.restore(model)
    tl, ta = eval_epoch(model, test_loader, crit, args.device)
    print(f'TEST | loss={tl:.4f} | acc={ta:.4f}')

    torch.save({'model_state': model.state_dict(), 'config': cfg.__dict__,
                'test_loss': tl, 'test_acc': ta}, os.path.join(args.outdir, 'treelstm_childsum_cls.pt'))

if __name__ == '__main__':
    main()