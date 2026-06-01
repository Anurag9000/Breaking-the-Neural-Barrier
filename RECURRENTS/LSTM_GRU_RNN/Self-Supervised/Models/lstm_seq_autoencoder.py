from dataclasses import dataclass
from typing import Tuple

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
        toks = text.strip().split()
        ids = [self.bos] + [self.stoi.get(t, self.unk) for t in toks] + [self.eos]
        return ids


class SeqDataset(Dataset):
    def __init__(self, sequences, max_len=128):
        self.sequences = [s[:max_len] for s in sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        s = self.sequences[idx]
        x = torch.tensor(s[:-1], dtype=torch.long)  # input w/ BOS..last-1
        y = torch.tensor(s[1:], dtype=torch.long)   # predict next token upto EOS
        return x, y


def pad_collate(batch):
    xs, ys = zip(*batch)
    maxT = max(x.size(0) for x in xs)
    xpad = torch.full((len(xs), maxT), 0, dtype=torch.long)
    ypad = torch.full((len(xs), maxT), 0, dtype=torch.long)
    for i, (x, y) in enumerate(zip(xs, ys)):
        xpad[i, :x.size(0)] = x
        ypad[i, :y.size(0)] = y
    return xpad, ypad


class LSTMSeqAutoencoder(nn.Module):
    def __init__(self, vocab, emb_dim, hidden, enc_layers, dec_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb_dim, padding_idx=0)
        self.encoder = nn.LSTM(emb_dim, hidden, num_layers=enc_layers, dropout=dropout if enc_layers > 1 else 0, batch_first=True)
        self.decoder = nn.LSTM(emb_dim, hidden, num_layers=dec_layers, dropout=dropout if dec_layers > 1 else 0, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden, vocab)

    def forward(self, x):  # teacher forcing reconstruction
        emb = self.drop(self.embedding(x))
        _, (h, c) = self.encoder(emb)
        out, _ = self.decoder(emb, (h, c))
        out = self.drop(out)
        return self.proj(out)


@dataclass
class TrainConfig:
    emb_dim: int = 256
    hidden: int = 256
    enc_layers: int = 1
    dec_layers: int = 1
    dropout: float = 0.1
    lr: float = 3e-4
    batch_size: int = 64
    max_epochs: int = 20
    patience: int = 4


def evaluate(model, loader, device) -> Tuple[float, float]:
    ce = nn.CrossEntropyLoss(ignore_index=0)
    model.eval()
    tot, toks = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            n = (y != 0).sum().item()
            tot += loss.item() * n
            toks += n
    return tot / max(1, toks), tot / max(1, toks)


def train(model, train_loader, val_loader, cfg: TrainConfig, device):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    ce = nn.CrossEntropyLoss(ignore_index=0)
    best, bad, best_state = float('inf'), 0, None
    for ep in range(1, cfg.max_epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = ce(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v, _ = evaluate(model, val_loader, device)
        if v + 1e-9 < best:
            best, bad = v, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"Epoch {ep}: val_xent={v:.4f}, best={best:.4f}, bad={bad}")
        if bad >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best
