from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim

from _common_librispeech_ctc import make_librispeech_ctc_loaders
from lstm_ctc import LSTMCTC, LSTMCTCConfig


class EarlyStopper:
    def __init__(self, patience: int):
        self.patience = patience
        self.best = float("inf")
        self.bad = 0
        self.state = None

    def step(self, value: float, model: nn.Module) -> bool:
        if value < self.best - 1e-7:
            self.best = value
            self.bad = 0
            self.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.bad += 1
        return self.bad >= self.patience

    def restore(self, model: nn.Module):
        if self.state is not None:
            model.load_state_dict(self.state)


def train_epoch(model, loader, opt, ctc_loss, device):
    model.train()
    total = 0.0
    n = 0
    for tokens, x_lens, flat_targets, y_lens in loader:
        tokens, x_lens = tokens.to(device), x_lens.to(device)
        flat_targets, y_lens = flat_targets.to(device), y_lens.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(tokens, x_lens)
        log_probs = logits.log_softmax(dim=-1)
        loss = ctc_loss(log_probs, flat_targets, x_lens, y_lens)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += float(loss.item()) * tokens.size(0)
        n += tokens.size(0)
    return total / max(1, n)


@torch.no_grad()
def eval_epoch(model, loader, ctc_loss, device):
    model.eval()
    total = 0.0
    n = 0
    for tokens, x_lens, flat_targets, y_lens in loader:
        tokens, x_lens = tokens.to(device), x_lens.to(device)
        flat_targets, y_lens = flat_targets.to(device), y_lens.to(device)
        logits = model(tokens, x_lens)
        log_probs = logits.log_softmax(dim=-1)
        loss = ctc_loss(log_probs, flat_targets, x_lens, y_lens)
        total += float(loss.item()) * tokens.size(0)
        n += tokens.size(0)
    return total / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=64)
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", type=str, default="results_lstm")
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    torch.manual_seed(args.seed)

    root = os.environ.get("BBNB_LIBRISPEECH_ROOT", "./data/LibriSpeech")
    train_loader, val_loader, test_loader, vocab_size, num_labels = make_librispeech_ctc_loaders(
        root=root,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        seed=args.seed,
    )

    cfg = LSTMCTCConfig(
        vocab_size=vocab_size,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        num_labels=num_labels,
        blank=0,
        pad_idx=0,
    )
    model = LSTMCTC(cfg).to(args.device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    es = EarlyStopper(args.patience)

    for epoch in range(1, args.max_epochs + 1):
        tr = train_epoch(model, train_loader, opt, ctc_loss, args.device)
        vl = eval_epoch(model, val_loader, ctc_loss, args.device)
        print(f"Epoch {epoch:03d} | train_ctc {tr:.4f} | val_ctc {vl:.4f}")
        if es.step(vl, model):
            print("Early stopping.")
            break

    es.restore(model)
    tl = eval_epoch(model, test_loader, ctc_loss, args.device)
    print(f"TEST | ctc_loss {tl:.4f}")
    torch.save({"model_state": model.state_dict(), "config": cfg.__dict__, "test_ctc": tl}, os.path.join(args.outdir, "lstm_ctc.pt"))


if __name__ == "__main__":
    main()
