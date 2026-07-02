import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from ..Models.dae_tokenmask_text_stl import TokenMaskTransformerDAE, token_dae_total_neurons


SPECIAL_TOKENS = ["<pad>", "<unk>", "<mask>"]


class TextDataset(Dataset):
    def __init__(self, lines: List[str], vocab: dict, max_len: int):
        self.lines = lines
        self.vocab = vocab
        self.max_len = max_len
        self.pad_id = vocab["<pad>"]
        self.unk_id = vocab["<unk>"]

    def __len__(self) -> int:
        return len(self.lines)

    def _encode(self, text: str) -> torch.Tensor:
        tokens = text.strip().split()
        ids = [self.vocab.get(tok, self.unk_id) for tok in tokens][: self.max_len]
        if len(ids) < self.max_len:
            ids.extend([self.pad_id] * (self.max_len - len(ids)))
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._encode(self.lines[idx])


def build_vocab(lines: List[str], vocab_size: int) -> dict:
    counter: Counter = Counter()
    for line in lines:
        counter.update(line.strip().split())
    most_common = [w for w, _ in counter.most_common(max(vocab_size - len(SPECIAL_TOKENS), 0))]
    vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for w in most_common:
        vocab[w] = len(vocab)
    return vocab


def random_mask_tokens(
    input_ids: torch.Tensor,
    mask_id: int,
    pad_id: int,
    mask_prob: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      masked_input (B, L),
      target (B, L) where non-masked positions are set to pad_id (ignored later).
    """
    if mask_prob <= 0.0:
        return input_ids, torch.full_like(input_ids, pad_id)
    B, L = input_ids.shape
    mask = torch.full((B, L), False, dtype=torch.bool, device=input_ids.device)
    rand = torch.rand((B, L), device=input_ids.device)
    mask = (rand < mask_prob) & input_ids.ne(pad_id)

    masked_input = input_ids.clone()
    masked_input[mask] = mask_id

    target = torch.full_like(input_ids, pad_id)
    target[mask] = input_ids[mask]
    return masked_input, target


def train_one_epoch(
    model: TokenMaskTransformerDAE,
    dl: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    mask_id: int,
    pad_id: int,
    mask_prob: float,
) -> float:
    model.train()
    ce = nn.CrossEntropyLoss(ignore_index=pad_id)
    total, n = 0.0, 0
    for ids in dl:
        ids = ids.to(device, non_blocking=True)
        masked_ids, targets = random_mask_tokens(ids, mask_id, pad_id, mask_prob)
        logits, _ = model(masked_ids)
        loss = ce(logits.view(-1, logits.size(-1)), targets.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += float(loss.item()) * ids.size(0)
        n += ids.size(0)
    return total / max(n, 1)


def eval_epoch(
    model: TokenMaskTransformerDAE,
    dl: DataLoader,
    device: torch.device,
    mask_id: int,
    pad_id: int,
    mask_prob: float,
) -> float:
    model.eval()
    ce = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    total, n = 0.0, 0
    with torch.no_grad():
        for ids in dl:
            ids = ids.to(device, non_blocking=True)
            masked_ids, targets = random_mask_tokens(ids, mask_id, pad_id, mask_prob)
            logits, _ = model(masked_ids)
            loss = ce(logits.view(-1, logits.size(-1)), targets.view(-1))
            total += float(loss.item())
            n += ids.size(0)
    return total / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--text-path", type=str, required=True, help="Plain text file, one example per line")
    p.add_argument("--results-dir", type=str, default="results_dae_tokenmask_text")
    p.add_argument("--max-len", type=int, default=64)
    p.add_argument("--vocab-size", type=int, default=20000)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--ffn-dim", type=int, default=1024)
    p.add_argument("--mask-prob", type=float, default=0.15)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10000000000000000000)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1337)

    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_path = Path(args.text_path)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    lines = [ln.strip() for ln in text_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"No non-empty lines found in {text_path}")

    vocab = build_vocab(lines, args.vocab_size)
    pad_id = vocab["<pad>"]
    mask_id = vocab["<mask>"]

    dataset = TextDataset(lines, vocab, args.max_len)
    n_total = len(dataset)
    n_val = int(n_total * args.val_frac)
    n_train = n_total - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    dl_train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=False)
    dl_val = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    model = TokenMaskTransformerDAE(
        vocab_size=len(vocab),
        d_model=args.d_model,
        depth=args.depth,
        num_heads=args.num_heads,
        dim_feedforward=args.ffn_dim,
        max_len=args.max_len,
        pad_id=pad_id,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log_path = results_dir / "training_log.txt"
    csv_path = results_dir / "training_stats.csv"

    log_f = log_path.open("w", encoding="utf-8")
    csv_f = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(csv_f)
    writer.writerow(
        ["epoch", "d_model", "depth", "neurons", "train_loss", "val_loss", "best_val", "best_epoch"]
    )

    neurons_metric = token_dae_total_neurons(args.d_model, args.depth)
    best_val = float("inf")
    best_epoch = -1
    es_counter = 0

    try:
        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model, dl_train, opt, device, mask_id, pad_id, args.mask_prob
            )
            val_loss = eval_epoch(
                model, dl_val, device, mask_id, pad_id, args.mask_prob
            )

            improved = val_loss < best_val - 1e-6
            if improved:
                best_val = val_loss
                best_epoch = epoch
                es_counter = 0
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "vocab": vocab,
                        "args": vars(args),
                    },
                    results_dir / "best.pt",
                )
            else:
                es_counter += 1

            msg = (
                f"Epoch {epoch:03d} | train={train_loss:.6f} | "
                f"val={val_loss:.6f} | best_val={best_val:.6f} @ {best_epoch}"
            )
            print(msg)
            log_f.write(msg + "\n")

            writer.writerow(
                [epoch, args.d_model, args.depth, neurons_metric, train_loss, val_loss, best_val, best_epoch]
            )
            csv_f.flush()

            if es_counter >= args.patience:
                stop_msg = f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)"
                print(stop_msg)
                log_f.write(stop_msg + "\n")
                break
    finally:
        log_f.flush()
        csv_f.flush()

    report = {
        "text_path": str(text_path),
        "vocab_size": len(vocab),
        "d_model": args.d_model,
        "depth": args.depth,
        "mask_prob": args.mask_prob,
        "neurons_metric": neurons_metric,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
    }
    with (results_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
