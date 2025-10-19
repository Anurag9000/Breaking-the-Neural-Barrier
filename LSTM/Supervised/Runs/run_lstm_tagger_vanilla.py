import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from lstm_tagger_vanilla import LSTMTagger, LSTMTaggerConfig


class SyntheticTagging(Dataset):
    """Synthetic aligned tagging dataset: each token emits a tag = (token % num_tags).
    Replace with a real tagging dataset for practical use.
    """
    def __init__(self, n: int, vocab_size: int, min_len: int, max_len: int, num_tags: int, pad_idx: int = 0):
        self.samples = []
        rng = random.Random(7)
        for _ in range(n):
            L = rng.randint(min_len, max_len)
            x = [rng.randint(1, vocab_size - 1) for _ in range(L)]
            y = [t % num_tags for t in x]
            self.samples.append((x, y))
        self.pad_idx = pad_idx
        self.num_tags = num_tags

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_pad(batch: List[Tuple[List[int], List[int]]], pad_idx: int):
    lengths = torch.tensor([len(x) for x, _ in batch], dtype=torch.long)
    max_len = int(lengths.max())
    tokens = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
    tags = torch.full((len(batch), max_len), -100, dtype=torch.long)  # -100 ignored in CE
    for i, (x, y) in enumerate(batch):
        L = len(x)
        tokens[i, :L] = torch.tensor(x, dtype=torch.long)
        tags[i, :L] = torch.tensor(y, dtype=torch.long)
    return tokens, lengths, tags


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 64
    max_epochs: int = 30
    patience: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class EarlyStopper:
    def __init__(self, patience: int):
        self.patience = patience
        self.best = float("inf")
        self.bad_epochs = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module):
        improved = val_loss < self.best - 1e-7
        if improved:
            self.best = val_loss
            self.bad_epochs = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.bad_epochs += 1
        return improved

    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience

    def restore_best(self, model: nn.Module):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def token_acc(logits: torch.Tensor, gold: torch.Tensor) -> float:
    mask = gold != -100
    preds = logits.argmax(dim=-1)
    correct = (preds[mask] == gold[mask]).sum().item()
    total = int(mask.sum().item())
    return correct / max(1, total)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    count = 0
    for tokens, lengths, tags in loader:
        tokens, lengths, tags = tokens.to(device), lengths.to(device), tags.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens, lengths)
        loss = criterion(logits.view(-1, logits.size(-1)), tags.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.item()) * tokens.size(0)
        count += tokens.size(0)
    return total / max(1, count)


def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    count = 0
    acc_total = 0.0
    with torch.no_grad():
        for tokens, lengths, tags in loader:
            tokens, lengths, tags = tokens.to(device), lengths.to(device), tags.to(device)
            logits = model(tokens, lengths)
            loss = criterion(logits.view(-1, logits.size(-1)), tags.view(-1))
            total += float(loss.item()) * tokens.size(0)
            count += tokens.size(0)
            acc_total += token_acc(logits, tags)
    return total / max(1, count), acc_total / max(1, len(loader))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=20000)
    ap.add_argument("--num-tags", type=int, default=10)
    ap.add_argument("--min-len", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=50)
    ap.add_argument("--train-n", type=int, default=8000)
    ap.add_argument("--val-n", type=int, default=1000)
    ap.add_argument("--test-n", type=int, default=1000)

    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--num-layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--pad-idx", type=int, default=0)

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--outdir", type=str, default="results_lstm")

    args = ap.parse_args()

    random.seed(7)
    torch.manual_seed(7)

    os.makedirs(args.outdir, exist_ok=True)

    train_ds = SyntheticTagging(args.train_n, args.vocab_size, args.min_len, args.max_len, args.num_tags, pad_idx=args.pad_idx)
    val_ds = SyntheticTagging(args.val_n, args.vocab_size, args.min_len, args.max_len, args.num_tags, pad_idx=args.pad_idx)
    test_ds = SyntheticTagging(args.test_n, args.vocab_size, args.min_len, args.max_len, args.num_tags, pad_idx=args.pad_idx)

    collate = lambda batch: collate_pad(batch, pad_idx=args.pad_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    cfg = LSTMTaggerConfig(
        vocab_size=args.vocab_size,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_tags=args.num_tags,
        pad_idx=args.pad_idx,
        bidirectional=False,
    )
    model = LSTMTagger(cfg).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    stopper = EarlyStopper(patience=args.patience)

    for epoch in range(1, args.max_epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, args.device)
        val_loss, val_tok_acc = evaluate(model, val_loader, criterion, args.device)
        stopper.step(val_loss, model)
        print(f"Epoch {epoch:03d} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | val_tok_acc={val_tok_acc:.4f}")
        if stopper.should_stop():
            print("Early stopping triggered.")
            break

    stopper.restore_best(model)

    test_loss, test_tok_acc = evaluate(model, test_loader, criterion, args.device)
    print(f"TEST | loss={test_loss:.4f} | token_acc={test_tok_acc:.4f}")

    torch.save({
        "model_state": model.state_dict(),
        "config": cfg.__dict__,
        "test_loss": test_loss,
        "test_token_acc": test_tok_acc,
    }, os.path.join(args.outdir, "lstm_tagger_vanilla.pt"))


if __name__ == "__main__":
    main()
