from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.datasets import fetch_20newsgroups


_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def _tokenize(text: str) -> List[str]:
    return [tok.lower() for tok in _TOKEN_RE.findall(text)]


class NewsGroupSequenceDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: dict[str, int], pad_idx: int = 0):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.pad_idx = pad_idx

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        ids = [self.vocab.get(tok, 1) for tok in _tokenize(self.texts[index])]
        if not ids:
            ids = [1]
        return ids, self.labels[index]


def collate_pad(batch: List[Tuple[List[int], int]], pad_idx: int):
    lengths = torch.tensor([len(x) for x, _ in batch], dtype=torch.long)
    max_len = int(lengths.max())
    tokens = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
    labels = torch.tensor([y for _, y in batch], dtype=torch.long)
    for i, (x, _) in enumerate(batch):
        tokens[i, : len(x)] = torch.tensor(x, dtype=torch.long)
    return tokens, lengths, labels


def make_20newsgroups_loaders(
    batch_size: int = 64,
    vocab_size: int = 20000,
    val_frac: float = 0.1,
    seed: int = 42,
):
    train_raw = fetch_20newsgroups(subset="train", remove=())
    test_raw = fetch_20newsgroups(subset="test", remove=())
    counts = Counter()
    for text in train_raw.data:
        counts.update(_tokenize(text))
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, _freq in counts.most_common(max(0, vocab_size - 2)):
        vocab[tok] = len(vocab)

    train_ds = NewsGroupSequenceDataset(train_raw.data, train_raw.target.tolist(), vocab)
    test_ds = NewsGroupSequenceDataset(test_raw.data, test_raw.target.tolist(), vocab)
    n_val = max(1, int(len(train_ds) * val_frac))
    n_train = len(train_ds) - n_val
    train_ds, val_ds = random_split(train_ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    def collate(batch):
        return collate_pad(batch, pad_idx=0)

    kwargs = dict(batch_size=batch_size, num_workers=2, collate_fn=collate, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader, len(vocab), 20

