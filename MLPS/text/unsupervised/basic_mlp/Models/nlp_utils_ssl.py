
import re, csv, random
from collections import Counter
from typing import List, Tuple
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

    def encode_ids(self, toks: List[str], max_len: int) -> torch.Tensor:
        unk = self.stoi.get("<unk>", 1)
        ids = [self.stoi.get(t, unk) for t in toks[:max_len]]
        return torch.tensor(ids, dtype=torch.long)

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

def collate_ssl(batch, vocab, max_len: int, word_dropout: float=0.1):
    """
    For SSL contrastive: create two augmented views per text via word dropout.
    Returns ((ids1, len1), (ids2, len2))
    """
    token_lists = [simple_tokenize(t) for t in batch]

    def drop_words(toks):
        if word_dropout <= 0: return toks
        keep = [t for t in toks if random.random() > word_dropout]
        return keep if keep else toks[:1]

    views1 = [drop_words(toks) for toks in token_lists]
    views2 = [drop_words(toks) for toks in token_lists]

    def pack(views):
        seqs = [vocab.encode_ids(t, max_len) for t in views]
        lens = [len(s) for s in seqs]
        maxL = max(lens) if lens else 1
        pad = vocab.pad_idx()
        ids = torch.full((len(seqs), maxL), pad, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :s.size(0)] = s
        return ids, torch.tensor(lens, dtype=torch.long)

    return pack(views1), pack(views2)
