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


class StepDataset(Dataset):
    def __init__(self, sequences: List[List[int]], max_len=128, k_future=3):
        self.seqs = [s[:max_len] for s in sequences]
        self.k = k_future

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        s = self.seqs[idx]
        x = torch.tensor(s[:-self.k], dtype=torch.long) if len(s) > self.k else torch.tensor(s, dtype=torch.long)
        y = torch.tensor(s[-1], dtype=torch.long)  # dummy label not used
        return x, y


def pad_collate(batch):
    xs, _ = zip(*batch)
    T = max(x.size(0) for x in xs)
    X = torch.full((len(xs), T), 0, dtype=torch.long)
    for i, x in enumerate(xs):
        X[i, :x.size(0)] = x
    return X, torch.zeros(len(xs), dtype=torch.long)


class LSTMFuturePredictor(nn.Module):
    def __init__(self, vocab, emb_dim=256, hidden=256, layers=1, dropout=0.1, proj_dim=256, k_future=3):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=layers, dropout=dropout if layers>1 else 0, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(k_future)])
        self.k = k_future

    def encode_all(self, x):
        z, _ = self.lstm(self.drop(self.emb(x)))  # (B, T, H)
        return z

    def forward(self, x):
        z = self.encode_all(x)
        hT = z[:, -1, :]
        preds = [p(hT) for p in self.proj]  # list of (B, H)
        return z.detach(), preds  # return detached target latents + predictions


def latent_pred_loss(target_z, preds):
    # target_z: (B, T, H) latent sequence (detached), preds: list of (B, H)
    B, T, H = target_z.size()
    loss = 0.0
    for k, pred in enumerate(preds, start=1):
        if T - 1 - k < 0:
            continue
        tgt = target_z[:, -1 - k, :]  # future k steps from last context
        loss += nn.functional.mse_loss(pred, tgt)
    return loss / max(1, len(preds))


@dataclass
class TrainConfig:
    emb_dim: int = 256
    hidden: int = 256
    layers: int = 1
    dropout: float = 0.1
    k_future: int = 3
    lr: float = 3e-4
    batch_size: int = 128
    max_epochs: int = 20
    patience: int = 5


def evaluate(model, loader, device, cfg: TrainConfig):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            z, preds = model(x)
            loss = latent_pred_loss(z, preds)
            tot += loss.item() * x.size(0)
            n += x.size(0)
    return tot / max(1, n)


def train(model, train_loader, val_loader, cfg: TrainConfig, device):
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    best, bad, best_state = float('inf'), 0, None
    for ep in range(1, cfg.max_epochs + 1):
        model.train()
        for x, _ in train_loader:
            x = x.to(device)
            opt.zero_grad(set_to_none=True)
            z, preds = model(x)
            loss = latent_pred_loss(z, preds)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        v = evaluate(model, val_loader, device, cfg)
        if v + 1e-9 < best:
            best, bad = v, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        print(f"Epoch {ep}: val_latent_mse={v:.4f}, best={best:.4f}, bad={bad}")
        if bad >= cfg.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best
