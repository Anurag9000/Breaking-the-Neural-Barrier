from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import List, Tuple

import nltk
import torch
from nltk.tree import Tree
from torch.utils.data import DataLoader, Dataset


def _ensure_treebank() -> None:
    try:
        nltk.data.find("corpora/treebank")
    except LookupError:
        nltk.download("treebank", quiet=True)


def _build_vocab(sent_trees: List[Tree], vocab_size: int) -> dict[str, int]:
    counts = Counter()
    for tree in sent_trees:
        counts.update(tok.lower() for tok in tree.leaves())
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok, _ in counts.most_common(max(0, vocab_size - 2)):
        vocab[tok] = len(vocab)
    return vocab


def _tree_to_sample(tree: Tree, word_vocab: dict[str, int], label_vocab: dict[str, int]):
    tokens: List[int] = []
    children: List[List[int]] = []

    def build(node) -> int:
        idx = len(tokens)
        if isinstance(node, str):
            tokens.append(word_vocab.get(node.lower(), 1))
            children.append([])
            return idx
        tokens.append(0)
        children.append([])
        for child in node:
            child_idx = build(child)
            children[idx].append(child_idx)
        return idx

    root = build(tree)
    label = label_vocab[tree.label()]
    return tokens, children, root, label


class TreebankConstituentDataset(Dataset):
    def __init__(self, trees: List[Tree], word_vocab: dict[str, int], label_vocab: dict[str, int], max_trees: int | None = None):
        samples = []
        for sent_tree in trees:
            for subtree in sent_tree.subtrees(lambda t: isinstance(t, Tree) and t.height() >= 3 and len(t.leaves()) >= 2):
                if subtree.label() not in label_vocab:
                    continue
                samples.append(_tree_to_sample(subtree, word_vocab, label_vocab))
                if max_trees is not None and len(samples) >= max_trees:
                    break
            if max_trees is not None and len(samples) >= max_trees:
                break
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


def collate_tree_batch(batch):
    tokens, children, roots, labels = zip(*batch)
    max_nodes = max(len(t) for t in tokens)
    padded_tokens = torch.zeros(len(batch), max_nodes, dtype=torch.long)
    padded_children: List[List[List[int]]] = []
    for i, (tok, child_list) in enumerate(zip(tokens, children)):
        padded_tokens[i, : len(tok)] = torch.tensor(tok, dtype=torch.long)
        cur = [list(ch) for ch in child_list]
        cur.extend([[] for _ in range(max_nodes - len(cur))])
        padded_children.append(cur)
    return padded_tokens, padded_children, list(roots), torch.tensor(labels, dtype=torch.long)


def make_treebank_loaders(
    batch_size: int = 16,
    vocab_size: int = 20000,
    seed: int = 42,
    max_trees: int | None = None,
):
    _ensure_treebank()
    from nltk.corpus import treebank

    parsed = list(treebank.parsed_sents())
    rng = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(parsed), generator=rng).tolist()
    parsed = [parsed[i] for i in perm]

    split1 = int(len(parsed) * 0.8)
    split2 = int(len(parsed) * 0.9)
    train_trees = parsed[:split1]
    val_trees = parsed[split1:split2]
    test_trees = parsed[split2:]

    word_vocab = _build_vocab(train_trees, vocab_size=vocab_size)
    label_counts = Counter()
    for sent_tree in train_trees:
        for subtree in sent_tree.subtrees(lambda t: isinstance(t, Tree) and t.height() >= 3 and len(t.leaves()) >= 2):
            label_counts[subtree.label()] += 1
    label_vocab = {label: idx for idx, (label, _count) in enumerate(label_counts.most_common())}

    train_ds = TreebankConstituentDataset(train_trees, word_vocab, label_vocab, max_trees=max_trees)
    val_ds = TreebankConstituentDataset(val_trees, word_vocab, label_vocab, max_trees=max_trees)
    test_ds = TreebankConstituentDataset(test_trees, word_vocab, label_vocab, max_trees=max_trees)

    kwargs = dict(batch_size=batch_size, num_workers=2, collate_fn=collate_tree_batch, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader, len(word_vocab), len(label_vocab)
