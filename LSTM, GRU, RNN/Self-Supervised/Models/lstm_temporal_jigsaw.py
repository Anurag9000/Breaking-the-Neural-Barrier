import itertools
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


PERMS = list(itertools.permutations([0,1,2]))  # 6 classes


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


class JigsawDataset(Dataset):
    def __init__(self, sequences: List[List[int]], max_len=180, seed=1337):
        self.seqs = [s[:max_len] for s in sequences]
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        s = self.seqs[idx]
        n = len(s)
        third = max(1, n // 3)
        segs = [s[:third], s[third:2*third], s[2*third:]]
        perm_idx = self.rng.randrange(len(PERMS))
        perm = PERMS[perm_idx]
        flat = []
        for i in perm:
            flat.extend(segs[i])
        x = torch.tensor(flat, dtype=torch.long)
        y = torch.tensor(perm_idx, dtype=torch.long)
        return x, y


def pad_collate(batch):
    xs, ys = zip(*batch)
    T = max(x.size(0) for x in xs)
    X = torch.full((len(xs), T), 0, dtype=torch.long)
    Y = torch.stack(ys)
    for i, x in enumerate(xs):
        X[i, :x.size(0)] = x
    return X, Y


class LSTMJigsaw(nn.Module):
    def __init__(self, vocab, emb_dim=256, hidden=256, layers=1, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=layers, dropout=dropout if layers>1 else 0, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Linear(hidden, len(PERMS))

    def forward(self, x):
        z, _ = self.lstm(self.drop(self.emb(x)))
        h = z[:, -1, :]
        return self.cls(self.drop(h))


@dataclass
class TrainConfig:
    emb_dim: int = 256
    hidden: int = 256
    layers: int = 1
    dropout: float = 0.1
    lr: float = 3e-4
    batch_size: int = 128
    max_epochs: int = 15
    patience: int = 4


def evaluate(model, loader, device) -> Tuple[float, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    tot_loss, tot_correct, tot = 0.0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce(logits, y)
            tot_loss += loss.item() * x.size(0)
            tot_correct += (logits.argmax(-1) == y).sum().item()
            tot += x.size(0)
    return tot_loss / max(1, tot), tot_correct / max(1, tot)


def train(model, train_loader, val_loader, cfg: TrainConfig, device):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    ce = nn.CrossEntropyLoss()
    best, bad, best_state = float('inf'), 0, None
    for ep in range(1, cfg.max_epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = ce(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v, acc = evaluate(model, val_loader, device)
        if v + 1e-9 < best:
            best, bad = v, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"Epoch {ep}: val_loss={v:.4f}, acc={acc:.3f}, best={best:.4f}, bad={bad}")
        if bad >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best
