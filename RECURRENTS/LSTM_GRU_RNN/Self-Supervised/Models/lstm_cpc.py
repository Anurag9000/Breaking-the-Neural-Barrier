import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ------------------------------
# Tokenizer & simple sequence dataset
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


class CPCDataset(Dataset):
    """Produces subsequences for CPC: context window and K future targets from the same sequence.
    Also returns a shuffled pool of negatives from other positions in the batch (handled in loss)."""
    def __init__(self, sequences: List[List[int]], max_len=128, context_len=32, k_future=3):
        self.seqs = [s[:max_len] for s in sequences if len(s) >= context_len + k_future + 2]
        self.context_len = context_len
        self.k_future = k_future

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        s = self.seqs[idx]
        # choose a start index ensuring we have context_len + k_future tokens
        max_start = max(1, len(s) - (self.context_len + self.k_future))
        start = random.randint(0, max_start - 1)
        ctx = torch.tensor(s[start : start + self.context_len], dtype=torch.long)
        futures = []
        for k in range(1, self.k_future + 1):
            futures.append(torch.tensor(s[start + self.context_len + k - 1], dtype=torch.long))
        futures = torch.stack(futures)  # (K)
        return ctx, futures


def pad_ctx(batch):
    ctxs, futures = zip(*batch)
    T = max(x.size(0) for x in ctxs)
    X = torch.full((len(ctxs), T), 0, dtype=torch.long)
    Y = torch.stack(futures)  # (B, K)
    for i, x in enumerate(ctxs):
        X[i, :x.size(0)] = x
    return X, Y


# ------------------------------
# Model: encoder + AR LSTM + K linear predictors, InfoNCE loss
# ------------------------------
class CPCEncoder(nn.Module):
    def __init__(self, vocab, emb_dim):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=0)
    def forward(self, x):
        return self.emb(x)  # (B, T, D)


class CPCLSTM(nn.Module):
    def __init__(self, vocab, emb_dim=256, ar_hidden=256, layers=1, k_future=3, dropout=0.1):
        super().__init__()
        self.encoder = CPCEncoder(vocab, emb_dim)
        self.ar = nn.LSTM(emb_dim, ar_hidden, num_layers=layers, dropout=dropout if layers>1 else 0, batch_first=True)
        self.projs = nn.ModuleList([nn.Linear(ar_hidden, emb_dim) for _ in range(k_future)])
        self.dropout = nn.Dropout(dropout)
        self.k_future = k_future
        self.emb_dim = emb_dim

    def forward(self, ctx_tokens):
        z = self.encoder(ctx_tokens)  # (B, T, D)
        z = self.dropout(z)
        c, _ = self.ar(z)            # (B, T, H)
        cT = c[:, -1, :]             # (B, H) context summary at end
        preds = [proj(cT) for proj in self.projs]  # list of (B, D)
        return z, preds  # return token embeddings (for positives) and predictions


def info_nce_logits(pred: torch.Tensor, positives: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
    """pred: (B, D), positives: (B, D), bank: (B, M, D) negatives/positives pool
       Return logits (B, M) where the diagonal corresponds to positives.
    """
    # Build big matrix of candidates: take the positives from each sample to act as positives/negatives across batch
    candidates = bank  # (B, M, D)
    # compute dot products with each row's candidate set
    # (B, D) @ (B, D, M) -> (B, M)
    logits = torch.bmm(candidates, pred.unsqueeze(-1)).squeeze(-1)  # (B, M)
    return logits


@dataclass
class TrainConfig:
    emb_dim: int = 256
    ar_hidden: int = 256
    layers: int = 1
    dropout: float = 0.1
    k_future: int = 3
    temperature: float = 0.1
    lr: float = 3e-4
    batch_size: int = 64
    max_epochs: int = 20
    patience: int = 5


def cpc_step(model: CPCLSTM, x, fut, device, temperature):
    # x: (B, T)  fut: (B, K) tokens for future positions
    z, preds = model(x)
    # gather true future embeddings from encoder's token space
    B, K = fut.size(0), fut.size(1)
    # Build candidate banks per step from the batch's future embeddings
    loss = 0.0
    for k in range(K):
        # positives embedding for this k from each sample
        pos_emb = model.encoder.emb(fut[:, k])  # (B, D)
        # bank = embeddings of all B samples' k-th future (acts as pos for its own row, negs for others)
        bank = pos_emb.unsqueeze(1).repeat(1, B, 1)  # (B, B, D)
        logits = info_nce_logits(preds[k], pos_emb, bank) / temperature
        targets = torch.arange(B, device=logits.device)
        loss += nn.functional.cross_entropy(logits, targets)
    return loss / K


def evaluate(model, loader, device, cfg: TrainConfig) -> float:
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for x, fut in loader:
            x, fut = x.to(device), fut.to(device)
            loss = cpc_step(model, x, fut, device, cfg.temperature)
            tot += loss.item() * x.size(0)
            n += x.size(0)
    return tot / max(1, n)


def train(model, train_loader, val_loader, cfg: TrainConfig, device):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    best, bad, best_state = float('inf'), 0, None
    for ep in range(1, cfg.max_epochs + 1):
        model.train()
        for x, fut in train_loader:
            x, fut = x.to(device), fut.to(device)
            opt.zero_grad(set_to_none=True)
            loss = cpc_step(model, x, fut, device, cfg.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = evaluate(model, val_loader, device, cfg)
        if v + 1e-9 < best:
            best, bad = v, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"Epoch {ep}: val_cpc={v:.4f}, best={best:.4f}, bad={bad}")
        if bad >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best
