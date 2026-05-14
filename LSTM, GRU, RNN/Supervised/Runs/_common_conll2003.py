from __future__ import annotations

from collections import Counter
from typing import List, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split


def _build_vocab(train_tokens: List[List[str]], vocab_size: int) -> dict[str, int]:
    counts = Counter()
    for tokens in train_tokens:
        counts.update(tok.lower() for tok in tokens)
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, _ in counts.most_common(max(0, vocab_size - 2)):
        vocab[tok] = len(vocab)
    return vocab


class Conll2003PosDataset(Dataset):
    def __init__(self, tokens: List[List[str]], tags: List[List[int]], vocab: dict[str, int]):
        self.tokens = tokens
        self.tags = tags
        self.vocab = vocab

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index: int):
        ids = [self.vocab.get(tok.lower(), 1) for tok in self.tokens[index]]
        return ids, self.tags[index]


def collate_pad(batch: List[Tuple[List[int], List[int]]], pad_idx: int = 0):
    lengths = torch.tensor([len(x) for x, _ in batch], dtype=torch.long)
    max_len = int(lengths.max())
    tokens = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
    tags = torch.full((len(batch), max_len), 0, dtype=torch.long)
    for i, (x, y) in enumerate(batch):
        L = len(x)
        tokens[i, :L] = torch.tensor(x, dtype=torch.long)
        tags[i, :L] = torch.tensor(y, dtype=torch.long)
    return tokens, lengths, tags


def make_conll2003_pos_loaders(
    batch_size: int = 64,
    vocab_size: int = 20000,
    seed: int = 42,
):
    dataset = load_dataset("conll2003")
    train_raw = dataset["train"]
    val_raw = dataset["validation"]
    test_raw = dataset["test"]

    vocab = _build_vocab(train_raw["tokens"], vocab_size=vocab_size)
    num_tags = int(train_raw.features["pos_tags"].feature.num_classes)

    train_ds = Conll2003PosDataset(train_raw["tokens"], train_raw["pos_tags"], vocab)
    val_ds = Conll2003PosDataset(val_raw["tokens"], val_raw["pos_tags"], vocab)
    test_ds = Conll2003PosDataset(test_raw["tokens"], test_raw["pos_tags"], vocab)

    def collate(batch):
        return collate_pad(batch, pad_idx=0)

    kwargs = dict(batch_size=batch_size, num_workers=2, collate_fn=collate, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader, len(vocab), num_tags
