from __future__ import annotations

import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset


def simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return text.strip().split()


class Vocab:
    def __init__(self, min_freq: int = 2, max_size: int = 50000, specials: Sequence[str] | None = None):
        self.min_freq = min_freq
        self.max_size = max_size
        self.specials = list(specials) if specials is not None else ["<pad>", "<unk>"]
        self.itos: List[str] = []
        self.stoi: dict[str, int] = {}

    def build(self, texts: Iterable[List[str]]) -> None:
        counter = Counter()
        for toks in texts:
            counter.update(toks)
        kept = [w for w, c in counter.items() if c >= self.min_freq]
        kept.sort(key=lambda w: (-counter[w], w))
        if self.max_size is not None:
            kept = kept[: max(0, self.max_size - len(self.specials))]
        self.itos = list(self.specials) + kept
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode_ids(self, toks: List[str], max_len: int) -> torch.Tensor:
        unk = self.stoi.get("<unk>", 1)
        ids = [self.stoi.get(t, unk) for t in toks[:max_len]]
        return torch.tensor(ids, dtype=torch.long)

    def pad_idx(self) -> int:
        return self.stoi.get("<pad>", 0)

    def __len__(self) -> int:
        return len(self.itos)


def add_special_tokens(vocab: Vocab, tokens: Sequence[str]) -> Vocab:
    for token in tokens:
        if token not in vocab.stoi:
            vocab.stoi[token] = len(vocab.itos)
            vocab.itos.append(token)
    return vocab


class TextClassificationDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[int, str]]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[int, str]:
        return self.samples[idx]


class TextCorpusDataset(Dataset):
    def __init__(self, texts: Sequence[str]):
        self.texts = list(texts)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


def _slice_samples(samples: List[Tuple[int, str]], max_samples: int | None) -> List[Tuple[int, str]]:
    if max_samples is None:
        return samples
    return samples[: max(0, int(max_samples))]


def load_ag_news_samples(
    *,
    seed: int = 0,
    val_fraction: float = 0.1,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]], List[Tuple[int, str]]]:
    ds = load_dataset("ag_news")
    train_samples = [(int(row["label"]), str(row["text"])) for row in ds["train"]]
    test_samples = [(int(row["label"]), str(row["text"])) for row in ds["test"]]

    rng = random.Random(seed)
    rng.shuffle(train_samples)
    n_val = max(1, int(len(train_samples) * float(val_fraction)))
    val_samples = train_samples[:n_val]
    train_samples = train_samples[n_val:]

    train_samples = _slice_samples(train_samples, max_train_samples)
    val_samples = _slice_samples(val_samples, max_val_samples)
    test_samples = _slice_samples(test_samples, max_test_samples)
    return train_samples, val_samples, test_samples


def build_vocab_from_texts(texts: Iterable[str], min_freq: int = 2, max_size: int = 50000) -> Vocab:
    vocab = Vocab(min_freq=min_freq, max_size=max_size)
    vocab.build(simple_tokenize(text) for text in texts)
    return vocab


def collate_classification(batch, vocab: Vocab, max_len: int):
    labels, ids = [], []
    for label, text in batch:
        labels.append(int(label))
        toks = simple_tokenize(text)
        ids.append(vocab.encode_ids(toks, max_len))
    seq_len = max(1, max((x.size(0) for x in ids), default=1))
    pad = vocab.pad_idx()
    padded = torch.full((len(ids), seq_len), pad, dtype=torch.long)
    for i, seq in enumerate(ids):
        padded[i, : seq.size(0)] = seq
    return padded, torch.tensor(labels, dtype=torch.long)


def collate_ssl_views(batch, vocab: Vocab, max_len: int, word_dropout: float = 0.1):
    texts = [simple_tokenize(text) for text in batch]

    def drop_words(toks: List[str]) -> List[str]:
        if word_dropout <= 0:
            return toks
        keep = [t for t in toks if random.random() > word_dropout]
        return keep if keep else toks[:1]

    def pack(views: List[List[str]]):
        seqs = [vocab.encode_ids(t, max_len) for t in views]
        lens = torch.tensor([max(1, seq.size(0)) for seq in seqs], dtype=torch.long)
        max_seq = max(1, max((seq.size(0) for seq in seqs), default=1))
        pad = vocab.pad_idx()
        ids = torch.full((len(seqs), max_seq), pad, dtype=torch.long)
        for i, seq in enumerate(seqs):
            ids[i, : seq.size(0)] = seq
        return ids, lens

    view1 = [drop_words(toks) for toks in texts]
    view2 = [drop_words(toks) for toks in texts]
    return pack(view1), pack(view2)


def make_ag_news_classification_loaders(
    *,
    batch_size: int = 64,
    max_len: int = 256,
    seed: int = 0,
    val_fraction: float = 0.1,
    min_freq: int = 2,
    max_vocab: int = 50000,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    max_test_samples: int | None = None,
    num_workers: int = 0,
):
    train_samples, val_samples, test_samples = load_ag_news_samples(
        seed=seed,
        val_fraction=val_fraction,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples,
    )
    vocab = build_vocab_from_texts((text for _, text in train_samples), min_freq=min_freq, max_size=max_vocab)

    collate = lambda batch: collate_classification(batch, vocab, max_len)
    train_loader = DataLoader(TextClassificationDataset(train_samples), batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate)
    val_loader = DataLoader(TextClassificationDataset(val_samples), batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    test_loader = DataLoader(TextClassificationDataset(test_samples), batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate)
    return train_loader, val_loader, test_loader, vocab, 4


def make_ag_news_ssl_loaders(
    *,
    batch_size: int = 256,
    max_len: int = 128,
    seed: int = 0,
    val_fraction: float = 0.1,
    word_dropout: float = 0.1,
    min_freq: int = 2,
    max_vocab: int = 50000,
    num_workers: int = 0,
):
    train_samples, val_samples, test_samples = load_ag_news_samples(seed=seed, val_fraction=val_fraction)
    vocab = build_vocab_from_texts((text for _, text in train_samples), min_freq=min_freq, max_size=max_vocab)

    train_texts = [text for _, text in train_samples]
    val_texts = [text for _, text in val_samples]
    test_texts = [text for _, text in test_samples]

    collate = lambda batch: collate_ssl_views(batch, vocab, max_len, word_dropout=word_dropout)

    train_loader = DataLoader(TextCorpusDataset(train_texts), batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate, drop_last=False)
    val_loader = DataLoader(TextCorpusDataset(val_texts), batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate, drop_last=False)
    test_loader = DataLoader(TextCorpusDataset(test_texts), batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate, drop_last=False)
    return train_loader, val_loader, test_loader, vocab


def make_ag_news_bow_autoencoder_loaders(
    *,
    batch_size: int = 64,
    max_len: int = 256,
    seed: int = 0,
    val_fraction: float = 0.1,
    min_freq: int = 2,
    max_vocab: int = 50000,
    num_workers: int = 0,
):
    train_samples, val_samples, test_samples = load_ag_news_samples(
        seed=seed,
        val_fraction=val_fraction,
    )
    vocab = build_vocab_from_texts((text for _, text in train_samples), min_freq=min_freq, max_size=max_vocab)

    train_texts = [text for _, text in train_samples]
    val_texts = [text for _, text in val_samples]
    test_texts = [text for _, text in test_samples]

    collate = lambda batch: collate_unsup(batch, vocab, max_len)
    train_loader = DataLoader(TextCorpusDataset(train_texts), batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate, drop_last=False)
    val_loader = DataLoader(TextCorpusDataset(val_texts), batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate, drop_last=False)
    test_loader = DataLoader(TextCorpusDataset(test_texts), batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate, drop_last=False)
    return train_loader, val_loader, test_loader, vocab, len(vocab)


def make_ag_news_semisup_loaders(
    *,
    batch_size: int = 128,
    max_len: int = 256,
    seed: int = 0,
    val_fraction: float = 0.1,
    label_fraction: float = 0.1,
    min_freq: int = 2,
    max_vocab: int = 50000,
    num_workers: int = 0,
):
    train_samples, val_samples, test_samples = load_ag_news_samples(seed=seed, val_fraction=val_fraction)
    vocab = build_vocab_from_texts((text for _, text in train_samples), min_freq=min_freq, max_size=max_vocab)
    add_special_tokens(vocab, ["<mask>"])

    rng = random.Random(seed)
    train_samples = list(train_samples)
    rng.shuffle(train_samples)
    n_label = max(1, int(len(train_samples) * float(label_fraction)))
    labeled_samples = train_samples[:n_label]
    unlabeled_samples = train_samples[n_label:]

    collate = lambda batch: collate_classification(batch, vocab, max_len)
    labeled_loader = DataLoader(TextClassificationDataset(labeled_samples), batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate, drop_last=False)
    unlabeled_loader = DataLoader(TextClassificationDataset(unlabeled_samples), batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate, drop_last=False)
    val_loader = DataLoader(TextClassificationDataset(val_samples), batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate, drop_last=False)
    test_loader = DataLoader(TextClassificationDataset(test_samples), batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate, drop_last=False)
    return labeled_loader, unlabeled_loader, val_loader, test_loader, vocab, 4
