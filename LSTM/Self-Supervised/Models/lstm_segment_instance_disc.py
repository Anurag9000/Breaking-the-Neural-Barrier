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


class SegmentDataset(Dataset):
    def __init__(self, sequences: List[List[int]], seg_len=64, views=2, seed=1337):
        self.seqs = sequences
        self.seg_len = seg_len
        self.views = views
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        s = self.seqs[idx]
        n = len(s)
        def sample_segment():
            if n <= self.seg_len:
                return torch.tensor(s, dtype=torch.long)
            start = self.rng.randint(0, n - self.seg_len)
            return torch.tensor(s[start:start+self.seg_len], dtype=torch.long)
        views = [sample_segment() for _ in range(self.views)]
        return tuple(views)


def pad_views(batch):
    V = len(batch[0])
    maxT = [0]*V
    for sample in batch:
        for v in range(V):
            maxT[v] = max(maxT[v], sample[v].size(0))
    out = []
    for v in range(V):
        T = maxT[v]
        X = torch.full((len(batch), T), 0, dtype=torch.long)
        for i, sample in enumerate(batch):
            x = sample[v]
            X[i, :x.size(0)] = x
        out.append(X)
    return tuple(out)


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


def nt_xent_many(embs: List[torch.Tensor], temperature: float):
    # embs: list of tensors [(B, D), (B, D), ...] length V (views)
    # positives are across views for the same index; negatives are others
    z = torch.cat(embs, dim=0)  # (V*B, D)
    sim = torch.mm(z, z.t())
    B = embs[0].size(0)
    V = len(embs)
    mask = torch.eye(V*B, device=z.device).bool()
    sim.masked_fill_(mask, -9e15)
    # For each view i*B..(i+1)B-1, positives are the same index in other views
    targets = []
    for i in range(V):
        for j in range(B):
            pos_row = ((i+1) % V) * B + j
            targets.append(pos_row)
    targets = torch.tensor(targets, device=z.device)
    logits = sim / temperature
    return nn.functional.cross_entropy(logits, targets)


def evaluate(model, loader, device, cfg: TrainConfig):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for views in loader:
            views = [v.to(device) for v in views]
            embs = [model(v) for v in views]
            loss = nt_xent_many(embs, cfg.temperature)
            tot += loss.item() * views[0].size(0)
            n += views[0].size(0)
    return tot / max(1, n)


def train(model, train_loader, val_loader, cfg: TrainConfig, device):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    best, bad, best_state = float('inf'), 0, None
    for ep in range(1, cfg.max_epochs + 1):
        model.train()
        for views in train_loader:
            views = [v.to(device) for v in views]
            opt.zero_grad(set_to_none=True)
            embs = [model(v) for v in views]
            loss = nt_xent_many(embs, cfg.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = evaluate(model, val_loader, device, cfg)
        if v + 1e-9 < best:
            best, bad = v, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"Epoch {ep}: val_loss={v:.4f}, best={best:.4f}, bad={bad}")
        if bad >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best
