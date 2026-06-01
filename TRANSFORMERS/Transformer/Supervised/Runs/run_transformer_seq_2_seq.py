import argparse
import math
from pathlib import Path
from typing import List

import sys
import torch
import torch.nn as nn

sys.path.append(str(Path(__file__).resolve().parents[4]))
from utils.adp_logging import ContinuousLogger
from torch.utils.data import Dataset, DataLoader

from model_transformer_seq2seq import TransformerSeq2Seq

class TSVSeq2Seq(Dataset):
    def __init__(self, path: Path, src_vocab: dict, tgt_vocab: dict, max_len: int = 128):
        self.pairs = []
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                s = line.strip().split('\t')
                if len(s) != 2:
                    continue
                self.pairs.append((s[0], s[1]))

    def __len__(self):
        return len(self.pairs)

    def encode(self, text: str, vocab: dict) -> List[int]:
        ids = [vocab.get(tok, vocab['<unk>']) for tok in text.strip().split()]
        return ids[: self.max_len]

    def pad(self, ids: List[int], pad_id: int, length: int) -> List[int]:
        return ids + [pad_id] * (length - len(ids))

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        return src, tgt


def build_vocab(lines: List[str], min_freq: int = 1) -> dict:
    from collections import Counter
    cnt = Counter()
    for line in lines:
        cnt.update(line.strip().split())
    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    for tok, c in cnt.items():
        if c >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def collate(batch, src_vocab: dict, tgt_vocab: dict, max_len: int):
    src_ids, tgt_in_ids, tgt_out_ids = [], [], []
    for src, tgt in batch:
        s = [src_vocab.get(tok, src_vocab['<unk>']) for tok in src.split()]
        t = [tgt_vocab['<bos>']] + [tgt_vocab.get(tok, tgt_vocab['<unk>']) for tok in tgt.split()] + [tgt_vocab['<eos>']]
        s = s[:max_len]
        t = t[:max_len]
        src_ids.append(s)
        tgt_in_ids.append(t[:-1])
        tgt_out_ids.append(t[1:])
    S = max(len(s) for s in src_ids)
    T = max(len(t) for t in tgt_in_ids)
    src_pad = src_vocab['<pad>']
    tgt_pad = tgt_vocab['<pad>']

    def pad_to(x, L, pad):
        return x + [pad] * (L - len(x))

    src_ids = torch.tensor([pad_to(s, S, src_pad) for s in src_ids], dtype=torch.long)
    tgt_in = torch.tensor([pad_to(t, T, tgt_pad) for t in tgt_in_ids], dtype=torch.long)
    tgt_out = torch.tensor([pad_to(t, T, tgt_pad) for t in tgt_out_ids], dtype=torch.long)
    src_pad_mask = (src_ids == src_pad)
    tgt_pad_mask = (tgt_in == tgt_pad)
    return src_ids, src_pad_mask, tgt_in, tgt_pad_mask, tgt_out


def evaluate(model, loader, criterion, device, tgt_pad_id: int):
    model.eval()
    total, denom = 0.0, 0
    with torch.no_grad():
        for src, src_pad, tgt_in, tgt_pad, tgt_out in loader:
            src, src_pad = src.to(device), src_pad.to(device)
            tgt_in, tgt_pad = tgt_in.to(device), tgt_pad.to(device)
            tgt_out = tgt_out.to(device)
            logits = model(src, src_pad, tgt_in, tgt_pad)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.view(-1))
            mask = (tgt_out.view(-1) != tgt_pad_id)
            total += (loss.detach().cpu().item() * mask.sum().item())
            denom += mask.sum().item()
    return total / max(denom, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train_tsv', type=str, required=True, help='TSV with "src\tgt" per line')
    p.add_argument('--val_tsv', type=str, required=True)
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--nhead', type=int, default=8)
    p.add_argument('--enc_layers', type=int, default=4)
    p.add_argument('--dec_layers', type=int, default=4)
    p.add_argument('--ff', type=int, default=1024)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--max_len', type=int, default=128)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    train_path = Path(args.train_tsv)
    val_path = Path(args.val_tsv)
    if not train_path.exists():
        raise FileNotFoundError(f"train_tsv not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"val_tsv not found: {val_path}")

    train_lines, val_lines = [], []
    with open(train_path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip().split('\t')
            if len(s) == 2:
                train_lines.extend([s[0], s[1]])
    with open(val_path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip().split('\t')
            if len(s) == 2:
                val_lines.extend([s[0], s[1]])

    src_vocab = build_vocab([l for i, l in enumerate(train_lines) if i % 2 == 0])
    tgt_vocab = build_vocab([l for i, l in enumerate(train_lines) if i % 2 == 1])

    train_ds = TSVSeq2Seq(train_path, src_vocab, tgt_vocab, args.max_len)
    val_ds = TSVSeq2Seq(val_path, src_vocab, tgt_vocab, args.max_len)

    coll = lambda batch: collate(batch, src_vocab, tgt_vocab, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=coll)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=coll)

    model = TransformerSeq2Seq(
        src_vocab=len(src_vocab), tgt_vocab=len(tgt_vocab), d_model=args.d_model,
        nhead=args.nhead, num_encoder_layers=args.enc_layers, num_decoder_layers=args.dec_layers,
        dim_feedforward=args.ff, dropout=args.dropout, max_len=args.max_len,
    ).to(args.device)

    criterion = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val, bad = float('inf'), 0

    # Init Logger

    logger = ContinuousLogger(Path('results_run_transformer_seq_2_seq'), 'run_transformer_seq_2_seq', 'train')

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_tokens = 0
        total_loss = 0.0
        for src, src_pad, tgt_in, tgt_pad, tgt_out in train_loader:
            src, src_pad = src.to(args.device), src_pad.to(args.device)
            tgt_in, tgt_pad = tgt_in.to(args.device), tgt_pad.to(args.device)
            tgt_out = tgt_out.to(args.device)
            optim.zero_grad()
            logits = model(src, src_pad, tgt_in, tgt_pad)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_out.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optim.step()
            ntok = (tgt_out != tgt_vocab['<pad>']).sum().item()
            total_tokens += ntok
            total_loss += loss.detach().cpu().item() * ntok
        train_ppl = math.exp(total_loss / max(total_tokens, 1))
        val_loss = evaluate(model, val_loader, criterion, args.device, tgt_vocab['<pad>'])
        val_ppl = math.exp(val_loss)
        # Log

        msg = f"Epoch {epoch}: train_ppl={train_ppl:.3f} val_ppl={val_ppl:.3f}"

        logger.log_console(msg)

        logger.log_epoch_stats({

            "epoch": epoch,

            "val_loss": val_loss if 'val_loss' in locals() else (loss.item() if 'loss' in locals() else 0),

            "train_loss": loss.item() if 'loss' in locals() else 0

        })
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            bad = 0
            torch.save({'model': model.state_dict(), 'src_vocab': src_vocab, 'tgt_vocab': tgt_vocab}, 'TransformerSeq2Seq_best.pth')
        else:
            bad += 1
            if bad >= args.patience:
                print('Early stopping.')
                break

    print('Done. Best validation loss:', best_val)

if __name__ == '__main__':
    main()
