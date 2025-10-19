import random
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ------------------------------
# Light text augmentations for sequences (token-level)
# ------------------------------
class SeqAugment:
    def __init__(self, drop_prob=0.1, mask_token=1, swap_prob=0.05):
        self.drop_prob = drop_prob
        self.mask_token = mask_token  # use <unk> as mask by default
        self.swap_prob = swap_prob
        self.rng = random.Random(1337)

    def __call__(self, ids: List[int]) -> List[int]:
        x = []
        for t in ids:
            if t == 0:  # keep pad
                x.append(t)
                continue
            r = self.rng.random()
            if r < self.drop_prob:
                continue
            elif r < self.drop_prob + self.swap_prob:
                x.append(self.mask_token)
            else:
                x.append(t)
        if not x:
            x = ids[:]
        return x


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


class SimCLRSeqDataset(Dataset):
    def __init__(self, sequences: List[List[int]], max_len=128, augment: SeqAugment = None):
        self.seqs = [s[:max_len] for s in sequences]
        self.aug = augment or SeqAugment()

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        s = self.seqs[idx]
        v1 = torch.tensor(self.aug(s), dtype=torch.long)
        v2 = torch.tensor(self.aug(s), dtype=torch.long)
        return v1, v2


def pad_pair(batch):
    v1s, v2s = zip(*batch)
    T = max(max(a.size(0), b.size(0)) for a, b in batch)
    A = torch.full((len(batch), T), 0, dtype=torch.long)
    B = torch.full((len(batch), T), 0, dtype=torch.long)
    for i, (a, b) in enumerate(batch):
        A[i, :a.size(0)] = a
        B[i, :b.size(0)] = b
    return A, B


class LSTMEncoder(nn.Module):
    def __init__(self, vocab, emb_dim=256, hidden=256, layers=1, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=layers, dropout=dropout if layers>1 else 0, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.hidden = hidden

    def forward(self, x):
        z, _ = self.lstm(self.dropout(self.emb(x)))
        # use last valid timestep (approx via last index, pads at end)
        return z[:, -1, :]


class ProjectionMLP(nn.Module):
    def __init__(self, dim, proj=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, proj)
        )
    def forward(self, x):
        return self.net(x)


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


class SimCLRSeq(nn.Module):
    def __init__(self, vocab, cfg: TrainConfig):
        super().__init__()
        self.enc = LSTMEncoder(vocab, cfg.emb_dim, cfg.hidden, cfg.layers, cfg.dropout)
        self.proj = ProjectionMLP(cfg.hidden, cfg.proj_dim)

    def forward(self, x):
        h = self.enc(x)
        z = nn.functional.normalize(self.proj(h), dim=-1)
        return z


def nt_xent(z1, z2, temperature):
    # concatenate and compute pairwise logits
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    sim = torch.mm(z, z.t())        # (2B, 2B)
    B = z1.size(0)
    # mask self-similarity
    mask = torch.eye(2*B, device=z.device).bool()
    sim.masked_fill_(mask, -9e15)
    # positives are (i, i+B) and (i+B, i)
    targets = torch.cat([torch.arange(B, 2*B, device=z.device), torch.arange(0, B, device=z.device)], dim=0)
    logits = sim / temperature
    loss = nn.functional.cross_entropy(logits, targets)
    return loss


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
