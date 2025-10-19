import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ------------------------------
# Tokenizer & dataset
# ------------------------------
class WhitespaceTokenizer:
    def __init__(self, min_freq: int = 1, specials=None):
        specials = specials or ["<pad>", "<unk>", "<bos>", "<eos>"]
        self.pad, self.unk, self.bos, self.eos = 0, 1, 2, 3
        self.stoi = {s: i for i, s in enumerate(specials)}
        self.itos = list(self.stoi.keys())
        self.min_freq = min_freq

    def build_vocab(self, texts: List[str]):
        from collections import Counter
        c = Counter()
        for t in texts:
            c.update(t.strip().split())
        for tok, f in c.items():
            if f >= self.min_freq and tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

    def encode(self, text: str) -> List[int]:
        toks = text.strip().split()
        return [self.bos] + [self.stoi.get(t, self.unk) for t in toks] + [self.eos]


class SeqDataset(Dataset):
    def __init__(self, sequences: List[List[int]], max_len=128):
        self.seqs = [s[:max_len] for s in sequences]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        s = self.seqs[idx]
        x = torch.tensor(s, dtype=torch.long)
        return x, torch.tensor(0, dtype=torch.long)  # placeholder label


def pad_collate(batch):
    xs, ys = zip(*batch)
    T = max(x.size(0) for x in xs)
    X = torch.full((len(xs), T), 0, dtype=torch.long)
    for i, x in enumerate(xs):
        X[i, :x.size(0)] = x
    Y = torch.zeros(len(xs), dtype=torch.long)
    return X, Y


# ------------------------------
# Model
# ------------------------------
class LSTMClust(nn.Module):
    def __init__(self, vocab, emb_dim=256, hidden=256, layers=1, dropout=0.1, n_clusters=100):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=layers, dropout=dropout if layers>1 else 0, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_clusters)
        self.hidden = hidden
        self.n_clusters = n_clusters

    def encode(self, x):
        z, _ = self.lstm(self.drop(self.emb(x)))
        h = z[:, -1, :]
        return h

    def forward(self, x):
        h = self.encode(x)
        return self.head(self.drop(h))


# ------------------------------
# Simple k-means (few iterations) on current embeddings
# ------------------------------
@torch.no_grad()
def kmeans_on_loader(model: LSTMClust, loader: DataLoader, device, n_clusters: int, iters: int = 10):
    model.eval()
    feats = []
    for x, _ in loader:
        x = x.to(device)
        h = model.encode(x)
        feats.append(h)
    X = torch.cat(feats, dim=0)
    # init centers by sampling
    idx = torch.randperm(X.size(0), device=X.device)[:n_clusters]
    centers = X[idx].clone()
    for _ in range(iters):
        # assign
        d = (X.unsqueeze(1) - centers.unsqueeze(0)).pow(2).sum(-1)  # (N, K)
        y = d.argmin(dim=1)
        # recompute centers
        for k in range(n_clusters):
            mask = (y == k)
            if mask.any():
                centers[k] = X[mask].mean(dim=0)
    return y.cpu(), centers.cpu()


@dataclass
class TrainConfig:
    emb_dim: int = 256
    hidden: int = 256
    layers: int = 1
    dropout: float = 0.1
    n_clusters: int = 100
    lr: float = 3e-4
    batch_size: int = 128
    max_epochs: int = 10
    patience: int = 3
    kmeans_iters: int = 10


def evaluate(model, loader, device):
    model.eval()
    tot, n = 0.0, 0
    ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce(logits, y)
            tot += loss.item() * x.size(0)
            n += x.size(0)
    return tot / max(1, n)


def train(model, train_loader, val_loader, cfg: TrainConfig, device):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    ce = nn.CrossEntropyLoss()
    best, bad, best_state = float('inf'), 0, None
    for epoch in range(1, cfg.max_epochs + 1):
        # E-step: run k-means on current embeddings (train split)
        pseudo, centers = kmeans_on_loader(model, train_loader, device, cfg.n_clusters, iters=cfg.kmeans_iters)
        # replace labels in a fresh loader snapshot
        def relabel(loader):
            xs = []
            for x, _ in loader:
                xs.append(x)
            X = torch.cat(xs, dim=0)
            return X
        Xtr = relabel(train_loader)
        Xva = relabel(val_loader)
        # build new datasets with pseudo labels (order must match concatenation above)
        class _Buf(Dataset):
            def __init__(self, X, y):
                self.X = X; self.y = y
            def __len__(self): return self.X.size(0)
            def __getitem__(self, i): return self.X[i], self.y[i]
        ytr = pseudo
        yva = torch.zeros(Xva.size(0), dtype=torch.long)  # dummy for val CE (no labels) -> keep original loader for val eval

        train_buf = _Buf(Xtr, ytr)
        train_buf_loader = DataLoader(train_buf, batch_size=cfg.batch_size, shuffle=True)

        # M-step: train classifier head (and encoder) using pseudo-labels
        model.train()
        for x, y in train_buf_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        # evaluate (unsupervised proxy: CE on k-means assignments of val set as well)
        v_pseudo, _ = kmeans_on_loader(model, val_loader, device, cfg.n_clusters, iters=max(1, cfg.kmeans_iters//2))
        # rebuild val buf
        Xva = relabel(val_loader)
        val_buf = _Buf(Xva, v_pseudo)
        val_buf_loader = DataLoader(val_buf, batch_size=cfg.batch_size, shuffle=False)
        v = evaluate(model, val_buf_loader, device)
        if v + 1e-9 < best:
            best, bad = v, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"Epoch {epoch}: val_proxy_ce={v:.4f}, best={best:.4f}, bad={bad}")
        if bad >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best
