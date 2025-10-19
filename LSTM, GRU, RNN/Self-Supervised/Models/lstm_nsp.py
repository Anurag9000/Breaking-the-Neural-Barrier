import random
from dataclasses import dataclass
from typing import List, Tuple

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


class NSPDataset(Dataset):
    """Builds pairs (s1, s2) where s2 is next sentence (label=1) or a random sentence (label=0)."""
    def __init__(self, texts: List[str], tokenizer: WhitespaceTokenizer, max_len=128, seed=1337):
        self.rng = random.Random(seed)
        self.tok = tokenizer
        self.max_len = max_len
        # make sentence list
        self.sents = [t.strip() for t in texts if t.strip()]

    def __len__(self):
        return max(0, len(self.sents) - 1)

    def __getitem__(self, idx):
        s1 = self.tok.encode(self.sents[idx])[:self.max_len]
        if self.rng.random() < 0.5 and idx + 1 < len(self.sents):
            s2 = self.tok.encode(self.sents[idx + 1])[:self.max_len]
            y = 1
        else:
            j = self.rng.randrange(len(self.sents))
            s2 = self.tok.encode(self.sents[j])[:self.max_len]
            y = 0
        a = torch.tensor(s1, dtype=torch.long)
        b = torch.tensor(s2, dtype=torch.long)
        return a, b, torch.tensor(y, dtype=torch.long)


def pad_pair(batch):
    a, b, y = zip(*batch)
    Ta = max(x.size(0) for x in a)
    Tb = max(x.size(0) for x in b)
    A = torch.full((len(batch), Ta), 0, dtype=torch.long)
    B = torch.full((len(batch), Tb), 0, dtype=torch.long)
    Y = torch.stack(y)
    for i, (x, z) in enumerate(zip(a, b)):
        A[i, :x.size(0)] = x
        B[i, :z.size(0)] = z
    return A, B, Y


class LSTMNSP(nn.Module):
    def __init__(self, vocab, emb_dim=256, hidden=256, layers=1, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=layers, dropout=dropout if layers>1 else 0, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Linear(2*hidden, 2)

    def encode(self, x):
        z, _ = self.lstm(self.drop(self.emb(x)))
        h = z[:, -1, :]
        return h

    def forward(self, a, b):
        ha = self.encode(a)
        hb = self.encode(b)
        h = torch.cat([ha, hb], dim=-1)
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
    ce = nn.CrossEntropyLoss()
    model.eval()
    tot_loss, tot_correct, tot = 0.0, 0, 0
    with torch.no_grad():
        for a, b, y in loader:
            a, b, y = a.to(device), b.to(device), y.to(device)
            logits = model(a, b)
            loss = ce(logits, y)
            tot_loss += loss.item() * a.size(0)
            tot_correct += (logits.argmax(-1) == y).sum().item()
            tot += a.size(0)
    return tot_loss / max(1, tot), tot_correct / max(1, tot)


def train(model, train_loader, val_loader, cfg: TrainConfig, device):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    ce = nn.CrossEntropyLoss()
    best, bad, best_state = float('inf'), 0, None
    for ep in range(1, cfg.max_epochs + 1):
        model.train()
        for a, b, y in train_loader:
            a, b, y = a.to(device), b.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = ce(model(a, b), y)
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
