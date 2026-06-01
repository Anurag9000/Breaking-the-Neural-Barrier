import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ------------------------------
# Tokenizer & Dataset
# ------------------------------
class WhitespaceTokenizer:
    def __init__(self, min_freq: int = 1, specials: List[str] = None):
        self.min_freq = min_freq
        self.specials = specials or ["<pad>", "<unk>"]
        self.pad, self.unk = 0, 1
        self.stoi = {s: i for i, s in enumerate(self.specials)}
        self.itos = list(self.stoi.keys())

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
        return [self.stoi.get(tok, self.unk) for tok in text.strip().split()]


class LMDataset(Dataset):
    def __init__(self, ids: List[int], bptt: int):
        self.ids = ids
        self.bptt = bptt

    def __len__(self):
        return max(0, len(self.ids) - self.bptt)

    def __getitem__(self, idx):
        x = torch.tensor(self.ids[idx: idx + self.bptt], dtype=torch.long)
        y = torch.tensor(self.ids[idx + 1: idx + 1 + self.bptt], dtype=torch.long)
        return x, y


# ------------------------------
# Model
# ------------------------------
class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden: int, num_layers: int, dropout: float, tie_weights: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=num_layers, dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden, vocab_size)
        if tie_weights:
            if hidden != emb_dim:
                raise ValueError("To tie weights, hidden must equal emb_dim")
            self.proj.weight = self.embedding.weight

    def forward(self, x):  # x: (B, T)
        emb = self.dropout(self.embedding(x))
        out, _ = self.lstm(emb)
        out = self.dropout(out)
        logits = self.proj(out)  # (B, T, V)
        return logits


# ------------------------------
# Utilities
# ------------------------------
@dataclass
class TrainConfig:
    emb_dim: int = 256
    hidden: int = 256
    layers: int = 2
    dropout: float = 0.2
    lr: float = 3e-4
    weight_decay: float = 0.0
    bptt: int = 64
    batch_size: int = 64
    max_epochs: int = 30
    patience: int = 4
    tie_weights: bool = True
    seed: int = 1337


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model: nn.Module, loader: DataLoader, device) -> Tuple[float, float]:
    model.eval()
    total_loss, total_tokens = 0.0, 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            n = (y != 0).sum().item()
            total_loss += loss.item() * n
            total_tokens += n
    ppl = math.exp(total_loss / max(1, total_tokens)) if total_tokens > 0 else float('inf')
    return total_loss / max(1, total_tokens), ppl


def train_lm(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, cfg: TrainConfig, device):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    best_val = float('inf')
    bad = 0
    best_state = None

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val_loss, val_ppl = evaluate(model, val_loader, device)
        if val_loss + 1e-9 < best_val:
            best_val = val_loss
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"Epoch {epoch}: val_loss={val_loss:.4f}, ppl={val_ppl:.2f}, best={best_val:.4f}, bad={bad}")
        if bad >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_val
