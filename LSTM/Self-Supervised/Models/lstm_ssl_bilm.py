import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# Reuse exactly the same tokenization/dataset interface as unidirectional LM
class WhitespaceTokenizer:
    def __init__(self, min_freq: int = 1, specials=None):
        specials = specials or ["<pad>", "<unk>"]
        self.pad, self.unk = 0, 1
        self.stoi = {s: i for i, s in enumerate(specials)}
        self.itos = list(self.stoi.keys())
        self.min_freq = min_freq

    def build_vocab(self, texts):
        from collections import Counter
        c = Counter()
        for t in texts:
            c.update(t.strip().split())
        for tok, f in c.items():
            if f >= self.min_freq and tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)

    def encode(self, text):
        return [self.stoi.get(tok, self.unk) for tok in text.strip().split()]


class LMDataset(Dataset):
    def __init__(self, ids, bptt):
        self.ids = ids
        self.bptt = bptt

    def __len__(self):
        return max(0, len(self.ids) - self.bptt - 1)

    def __getitem__(self, idx):
        x = torch.tensor(self.ids[idx: idx + self.bptt], dtype=torch.long)
        y_fwd = torch.tensor(self.ids[idx + 1: idx + 1 + self.bptt], dtype=torch.long)
        # backward targets are previous tokens (shifted right)
        y_bwd = torch.tensor(self.ids[idx: idx + self.bptt], dtype=torch.long)
        return x, (y_fwd, y_bwd)


class BiLSTMLM(nn.Module):
    def __init__(self, vocab, emb_dim, hidden, layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb_dim, padding_idx=0)
        self.fwd = nn.LSTM(emb_dim, hidden, num_layers=layers, dropout=dropout if layers > 1 else 0, batch_first=True)
        self.bwd = nn.LSTM(emb_dim, hidden, num_layers=layers, dropout=dropout if layers > 1 else 0, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.proj_f = nn.Linear(hidden, vocab)
        self.proj_b = nn.Linear(hidden, vocab)

    def forward(self, x):
        emb = self.drop(self.embedding(x))
        out_f, _ = self.fwd(emb)
        # reverse time for backward direction
        out_b, _ = self.bwd(torch.flip(emb, dims=[1]))
        out_b = torch.flip(out_b, dims=[1])
        out_f = self.drop(out_f)
        out_b = self.drop(out_b)
        return self.proj_f(out_f), self.proj_b(out_b)


@dataclass
class TrainConfig:
    emb_dim: int = 256
    hidden: int = 256
    layers: int = 2
    dropout: float = 0.2
    lr: float = 3e-4
    bptt: int = 64
    batch_size: int = 64
    max_epochs: int = 30
    patience: int = 4


def evaluate(model, loader, device) -> Tuple[float, float]:
    model.eval()
    ce = nn.CrossEntropyLoss(ignore_index=0)
    tot, toks = 0.0, 0
    with torch.no_grad():
        for x, (yf, yb) in loader:
            x, yf, yb = x.to(device), yf.to(device), yb.to(device)
            lf, lb = model(x)
            loss = ce(lf.reshape(-1, lf.size(-1)), yf.reshape(-1)) + \
                   ce(lb.reshape(-1, lb.size(-1)), yb.reshape(-1))
            n = (yf != 0).sum().item() + (yb != 0).sum().item()
            tot += loss.item() * n
            toks += n
    ppl = math.exp(tot / max(1, toks)) if toks > 0 else float('inf')
    return tot / max(1, toks), ppl


def train(model, train_loader, val_loader, cfg: TrainConfig, device):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    ce = nn.CrossEntropyLoss(ignore_index=0)
    best, bad, best_state = float('inf'), 0, None
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        for x, (yf, yb) in train_loader:
            x, yf, yb = x.to(device), yf.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            lf, lb = model(x)
            loss = ce(lf.reshape(-1, lf.size(-1)), yf.reshape(-1)) + \
                   ce(lb.reshape(-1, lb.size(-1)), yb.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v, p = evaluate(model, val_loader, device)
        if v + 1e-9 < best:
            best, bad = v, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"Epoch {epoch}: val_loss={v:.4f}, ppl={p:.2f}, best={best:.4f}, bad={bad}")
        if bad >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best
