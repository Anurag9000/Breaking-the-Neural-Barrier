from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
import torchaudio
from torchaudio.datasets import LIBRISPEECH
from torch.utils.data import DataLoader, Dataset, random_split


ALPHABET = " abcdefghijklmnopqrstuvwxyz'"
CHAR_TO_ID = {ch: idx + 1 for idx, ch in enumerate(ALPHABET)}


def _encode_transcript(text: str) -> List[int]:
    encoded = [CHAR_TO_ID[ch] for ch in text.lower() if ch in CHAR_TO_ID]
    return encoded if encoded else [CHAR_TO_ID[" "]]


def _quantize_energy(frame_energy: torch.Tensor, vocab_size: int) -> torch.Tensor:
    frame_energy = frame_energy.float()
    lo = frame_energy.min()
    hi = frame_energy.max()
    scaled = (frame_energy - lo) / (hi - lo + 1e-6)
    return torch.clamp((scaled * (vocab_size - 2)).long() + 1, 1, vocab_size - 1)


class LibriSpeechCTCDataset(Dataset):
    def __init__(self, root: str | Path, url: str, vocab_size: int = 64):
        self.dataset = LIBRISPEECH(root=str(root), url=url, download=True)
        self.vocab_size = vocab_size
        self.mfcc = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=13)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int):
        waveform, sample_rate, transcript, *_ = self.dataset[index]
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        mfcc = self.mfcc(waveform).squeeze(0)  # (13, T)
        energy = mfcc.abs().mean(dim=0)
        tokens = _quantize_energy(energy, self.vocab_size)
        targets = torch.tensor(_encode_transcript(transcript), dtype=torch.long)
        return tokens.tolist(), targets.tolist()


def collate_ctc(batch: List[Tuple[List[int], List[int]]], pad_idx: int = 0):
    x_lens = torch.tensor([len(x) for x, _ in batch], dtype=torch.long)
    max_t = int(x_lens.max())
    tokens = torch.full((len(batch), max_t), pad_idx, dtype=torch.long)
    for i, (x, _) in enumerate(batch):
        tokens[i, : len(x)] = torch.tensor(x, dtype=torch.long)
    targets = [torch.tensor(y, dtype=torch.long) for _, y in batch]
    flat_targets = torch.cat(targets, dim=0)
    y_lens = torch.tensor([len(y) for _, y in batch], dtype=torch.long)
    return tokens, x_lens, flat_targets, y_lens


def make_librispeech_ctc_loaders(
    root: str | Path,
    batch_size: int = 16,
    vocab_size: int = 64,
    seed: int = 42,
):
    train_ds = LibriSpeechCTCDataset(root=root, url="train-clean-100", vocab_size=vocab_size)
    val_ds = LibriSpeechCTCDataset(root=root, url="dev-clean", vocab_size=vocab_size)
    test_ds = LibriSpeechCTCDataset(root=root, url="test-clean", vocab_size=vocab_size)

    kwargs = dict(batch_size=batch_size, num_workers=2, collate_fn=collate_ctc, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader, vocab_size, len(ALPHABET)
