
import re, csv, random
from collections import Counter
from typing import List
import torch
from torch.utils.data import Dataset

def simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return text.strip().split()

class Vocab:
    def __init__(self, min_freq: int=2, max_size: int=50000, specials=None):
        self.min_freq = min_freq
        self.max_size = max_size
        self.specials = specials or ["<pad>", "<unk>"]
        self.itos = []; self.stoi = {}

    def build(self, texts: List[List[str]]):
        counter = Counter()
        for toks in texts: counter.update(toks)
        kept = [w for w, c in counter.items() if c >= self.min_freq]
        kept.sort(key=lambda w: (-counter[w], w))
        if self.max_size is not None:
            kept = kept[: self.max_size - len(self.specials)]
        self.itos = list(self.specials) + kept
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, toks: List[str]) -> List[int]:
        unk = self.stoi.get("<unk>", 1)
        return [self.stoi.get(t, unk) for t in toks]

    def pad_idx(self): return self.stoi.get("<pad>", 0)
    def __len__(self): return len(self.itos)

class TextOnlyCSV(Dataset):
    """CSV with header column 'text'. Extra columns ignored."""
    def __init__(self, path: str):
        self.texts = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.texts.append(r["text"])

    def __len__(self): return len(self.texts)
    def __getitem__(self, i): return self.texts[i]

def build_vocab_from_csv(train_csv: str, min_freq=2, max_size=50000):
    ds = TextOnlyCSV(train_csv)
    texts = [simple_tokenize(t) for t in ds.texts]
    vocab = Vocab(min_freq=min_freq, max_size=max_size)
    vocab.build(texts)
    return vocab

def collate_unsup(batch, vocab, max_len: int, view_word_dropout: float=0.0):
    """
    Returns:
      - view (token_ids, lengths) : for encoder (avg embeddings)
      - bow_tf: (B, |V|) soft term-frequency distribution (target for decoder)
    """
    texts = batch
    # apply word dropout for denoising (only to encoder input)
    token_lists = [simple_tokenize(t) for t in texts]
    if view_word_dropout > 0.0:
        dropped = []
        for toks in token_lists:
            keep = [t for t in toks if random.random() > view_word_dropout]
            dropped.append(keep if keep else toks[:1])
        token_lists = dropped

    # encoder inputs (avg embedding)
    ids = []
    lengths = []
    for toks in token_lists:
        ids.append(torch.tensor([vocab.stoi.get(t, vocab.stoi.get("<unk>", 1)) for t in toks[:max_len]], dtype=torch.long))
        lengths.append(len(ids[-1]))
    maxL = max(lengths) if lengths else 1
    token_ids = torch.full((len(batch), maxL), vocab.pad_idx(), dtype=torch.long)
    for i, seq in enumerate(ids):
        token_ids[i, :seq.size(0)] = seq
    lens = torch.tensor(lengths, dtype=torch.long)

    # BOW soft targets (term-frequency)
    V = len(vocab)
    bow = torch.zeros(len(batch), V, dtype=torch.float)
    for i, text in enumerate(texts):
        toks = simple_tokenize(text)[:max_len]
        if len(toks) == 0:
            continue
        for t in toks:
            idx = vocab.stoi.get(t, vocab.stoi.get("<unk>", 1))
            bow[i, idx] += 1.0
    # normalize to TF distribution (avoid div0)
    row_sums = bow.sum(dim=1, keepdim=True).clamp(min=1.0)
    bow_tf = bow / row_sums
    return (token_ids, lens), bow_tf
