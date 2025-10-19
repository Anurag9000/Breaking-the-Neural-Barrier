import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ------------------------------
# Tokenizer + simple augmentations
# ------------------------------
class SeqAugment:
    def __init__(self, drop_prob=0.1, mask_token=1, swap_prob=0.05, seed=1337):
        self.drop_prob = drop_prob
        self.mask_token = mask_token
        self.swap_prob = swap_prob
        self.rng = random.Random(seed)

    def __call__(self, ids: List[int]) -> List[int]:
        x = []
        for t in ids:
            if t == 0:
                x.append(t)
                continue
            r = self.rng.random()
            if r < self.drop_prob:
                continue
            elif r < self.drop_prob + self.swap_prob:
                x.append(self.mask_token)
            else:
                x.append(t)
        return x if x else ids[:]


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


# ------------------------------
# Datasets
# ------------------------------
class SiameseDataset(Dataset):
    def __init__(self, sequences: List[List[int]], max_len=128, augment: SeqAugment = None):
        self.seqs = [s[:max_len] for s in sequences]
        self.aug = augment or SeqAugment()

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        s = self.seqs[idx]
        a = torch.tensor(self.aug(s), dtype=torch.long)
        b = torch.tensor(self.aug(s), dtype=torch.long)
        return a, b


class TripletDataset(Dataset):
    def __init__(self, sequences: List[List[int]], max_len=128, augment: SeqAugment = None, seed=1337):
        self.seqs = [s[:max_len] for s in sequences]
        self.aug = augment or SeqAugment()
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        s_anchor = self.seqs[idx]
        j = self.rng.randrange(len(self.seqs))
        while j == idx:
            j = self.rng.randrange(len(self.seqs))
        s_neg = self.seqs[j]
        a = torch.tensor(self.aug(s_anchor), dtype=torch.long)
        p = torch.tensor(self.aug(s_anchor), dtype=torch.long)
        n = torch.tensor(self.aug(s_neg), dtype=torch.long)
        return a, p, n


def pad_pair(batch):
    a, b = zip(*batch)
    T = max(max(x.size(0), y.size(0)) for x, y in batch)
    A = torch.full((len(batch), T), 0, dtype=torch.long)
    B = torch.full((len(batch), T), 0, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        A[i, :x.size(0)] = x
        B[i, :y.size(0)] = y
    return A, B


def pad_triplet(batch):
    a, p, n = zip(*batch)
    Ta = max(x.size(0) for x in a)
    Tp = max(x.size(0) for x in p)
    Tn = max(x.size(0) for x in n)
    A = torch.full((len(batch), Ta), 0, dtype=torch.long)
    P = torch.full((len(batch), Tp), 0, dtype=torch.long)
    N = torch.full((len(batch), Tn), 0, dtype=torch.long)
    for i, (x, y, z) in enumerate(batch):
        A[i, :x.size(0)] = x
        P[i, :y.size(0)] = y
        N[i, :z.size(0)] = z
    return A, P, N


# ------------------------------
# Model + losses
# ------------------------------
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


def contrastive_nt_xent(e1, e2, temperature=0.1):
    z = torch.cat([e1, e2], dim=0)
    sim = torch.mm(z, z.t())
    B = e1.size(0)
    mask = torch.eye(2*B, device=z.device).bool()
    sim.masked_fill_(mask, -9e15)
    targets = torch.cat([torch.arange(B, 2*B, device=z.device), torch.arange(0, B, device=z.device)], dim=0)
    logits = sim / temperature
    return nn.functional.cross_entropy(logits, targets)


def triplet_loss(a, p, n, margin=0.2):
    d_ap = 1 - (a * p).sum(dim=-1)  # cosine distance (since normalized)
    d_an = 1 - (a * n).sum(dim=-1)
    return torch.clamp(d_ap - d_an + margin, min=0).mean()


@dataclass
class TrainConfig:
    emb_dim: int = 256
    hidden: int = 256
    layers: int = 1
    dropout: float = 0.1
    proj_dim: int = 128
    temperature: float = 0.1
    margin: float = 0.2
    use_triplet: bool = False  # if False, use NT-Xent
    lr: float = 3e-4
    batch_size: int = 128
    max_epochs: int = 20
    patience: int = 5


def evaluate(model, loader, device, cfg: TrainConfig):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        if cfg.use_triplet:
            for a, p, neg in loader:
                a, p, neg = a.to(device), p.to(device), neg.to(device)
                ea, ep, en = model(a), model(p), model(neg)
                loss = triplet_loss(ea, ep, en, margin=cfg.margin)
                tot += loss.item() * a.size(0)
                n += a.size(0)
        else:
            for a, b in loader:
                a, b = a.to(device), b.to(device)
                ea, eb = model(a), model(b)
                loss = contrastive_nt_xent(ea, eb, cfg.temperature)
                tot += loss.item() * a.size(0)
                n += a.size(0)
    return tot / max(1, n)


def train(model, train_loader, val_loader, cfg: TrainConfig, device):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    best, bad, best_state = float('inf'), 0, None
    for ep in range(1, cfg.max_epochs + 1):
        model.train()
        if cfg.use_triplet:
            for a, p, neg in train_loader:
                a, p, neg = a.to(device), p.to(device), neg.to(device)
                opt.zero_grad(set_to_none=True)
                ea, ep, en = model(a), model(p), model(neg)
                loss = triplet_loss(ea, ep, en, margin=cfg.margin)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        else:
            for a, b in train_loader:
                a, b = a.to(device), b.to(device)
                opt.zero_grad(set_to_none=True)
                ea, eb = model(a), model(b)
                loss = contrastive_nt_xent(ea, eb, cfg.temperature)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        v = evaluate(model, val_loader, device, cfg)
        if v + 1e-9 < best:
            best, bad = v, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"Epoch {ep}: val_metric={v:.4f}, best={best:.4f}, bad={bad}")
        if bad >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best
