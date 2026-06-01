import random
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


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


class CropsDataset(Dataset):
    def __init__(self, sequences: List[List[int]], max_len=256, seed=1337):
        self.seqs = [s[:max_len] for s in sequences]
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        s = self.seqs[idx]
        n = len(s)
        # two random crops at different scales
        l1 = self.rng.randint(max(4, n//6), max(5, n//3))
        l2 = self.rng.randint(max(4, n//6), max(5, n//2))
        i1 = self.rng.randint(0, max(0, n-l1))
        i2 = self.rng.randint(0, max(0, n-l2))
        a = torch.tensor(s[i1:i1+l1], dtype=torch.long)
        b = torch.tensor(s[i2:i2+l2], dtype=torch.long)
        return a, b


def pad_pair(batch):
    a, b = zip(*batch)
    T = max(max(x.size(0), y.size(0)) for x, y in batch)
    A = torch.full((len(batch), T), 0, dtype=torch.long)
    B = torch.full((len(batch), T), 0, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        A[i, :x.size(0)] = x
        B[i, :y.size(0)] = y
    return A, B


class LSTMEncoder(nn.Module):
    def __init__(self, vocab, emb_dim=256, hidden=256, layers=1, dropout=0.1, proj_dim=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=layers, dropout=dropout if layers>1 else 0, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden, proj_dim)

    def forward(self, x):
        z, _ = self.lstm(self.drop(self.emb(x)))
        h = z[:, -1, :]
        e = nn.functional.normalize(self.proj(h), dim=-1)
        return e


@dataclass
class TrainConfig:
    emb_dim: int = 256
    hidden: int = 256
    layers: int = 1
    dropout: float = 0.1
    proj_dim: int = 128
    temperature: float = 0.1
    lr: float = 3e-4
    batch_size: int = 128
    max_epochs: int = 20
    patience: int = 5


def nt_xent(z1, z2, temperature):
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t())
    B = z1.size(0)
    mask = torch.eye(2*B, device=z.device).bool()
    sim.masked_fill_(mask, -9e15)
    targets = torch.cat([torch.arange(B, 2*B, device=z.device), torch.arange(0, B, device=z.device)], dim=0)
    logits = sim / temperature
    return nn.functional.cross_entropy(logits, targets)


def evaluate(model, loader, device, cfg: TrainConfig):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for a, b in loader:
            a, b = a.to(device), b.to(device)
            z1, z2 = model(a), model(b)
            loss = nt_xent(z1, z2, cfg.temperature)
            tot += loss.item() * a.size(0)
            n += a.size(0)
    return tot / max(1, n)


def train(model, train_loader, val_loader, cfg: TrainConfig, device):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    best, bad, best_state = float('inf'), 0, None
    for ep in range(1, cfg.max_epochs + 1):
        model.train()
        for a, b in train_loader:
            a, b = a.to(device), b.to(device)
            opt.zero_grad(set_to_none=True)
            z1, z2 = model(a), model(b)
            loss = nt_xent(z1, z2, cfg.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = evaluate(model, val_loader, device, cfg)
        if v + 1e-9 < best:
            best, bad = v, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"Epoch {ep}: val_ntxent={v:.4f}, best={best:.4f}, bad={bad}")
        if bad >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best
