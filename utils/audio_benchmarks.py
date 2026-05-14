from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, List, Tuple

import torch
import torchaudio
from torchaudio.datasets import LIBRISPEECH, SPEECHCOMMANDS
from torch.utils.data import DataLoader, Dataset


def _speechcommands_labels(root: Path, download: bool) -> List[str]:
    probe = SPEECHCOMMANDS(root=str(root), download=download)
    base_path = Path(getattr(probe, "_path", root))
    labels = sorted(
        {
            wav.parent.name
            for wav in base_path.rglob("*.wav")
            if wav.parent.name and wav.parent.name != "_background_noise_"
        }
    )
    if not labels:
        raise RuntimeError(f"Could not infer SpeechCommands labels under {base_path}")
    return labels


class SpeechCommandsMFCCDataset(Dataset):
    def __init__(self, root: Path, subset: str, label_to_index: Dict[str, int], download: bool = True, n_mfcc: int = 80):
        self.dataset = SPEECHCOMMANDS(root=str(root), download=download, subset=subset)
        self.label_to_index = label_to_index
        self.mfcc = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=n_mfcc)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        waveform, sample_rate, label, *_ = self.dataset[idx]
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        feats = self.mfcc(waveform).squeeze(0).transpose(0, 1).contiguous()
        return feats, self.label_to_index[label]


def make_speechcommands_loaders(
    *,
    root: str | Path = "./data/SpeechCommands",
    batch_size: int = 64,
    download: bool = True,
    n_mfcc: int = 80,
    num_workers: int = 0,
):
    root = Path(root)
    labels = _speechcommands_labels(root, download=download)
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    train_ds = SpeechCommandsMFCCDataset(root, "training", label_to_index, download=download, n_mfcc=n_mfcc)
    val_ds = SpeechCommandsMFCCDataset(root, "validation", label_to_index, download=download, n_mfcc=n_mfcc)
    test_ds = SpeechCommandsMFCCDataset(root, "testing", label_to_index, download=download, n_mfcc=n_mfcc)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, len(labels)


_LIBRISPEECH_CHARS = ["<blank>", "<unk>", " ", "'"] + list("abcdefghijklmnopqrstuvwxyz")
_LIBRISPEECH_CHAR_TO_IDX = {ch: idx for idx, ch in enumerate(_LIBRISPEECH_CHARS)}


def _normalize_librispeech_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class LibriSpeechCTCDataset(Dataset):
    def __init__(self, root: Path, subset: str, download: bool = True, n_mfcc: int = 64):
        self.dataset = LIBRISPEECH(root=str(root), url=subset, download=download)
        self.mfcc = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=n_mfcc)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        waveform, sample_rate, transcript, *_ = self.dataset[idx]
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        feats = self.mfcc(waveform).squeeze(0).contiguous()
        spec = feats.unsqueeze(0)  # (1, n_mfcc, time)
        text = _normalize_librispeech_text(transcript)
        token_ids = [
            _LIBRISPEECH_CHAR_TO_IDX.get(ch, _LIBRISPEECH_CHAR_TO_IDX["<unk>"])
            for ch in text
            if ch in _LIBRISPEECH_CHAR_TO_IDX or ch == " "
        ]
        if not token_ids:
            token_ids = [_LIBRISPEECH_CHAR_TO_IDX["<unk>"]]
        return spec, torch.tensor(token_ids, dtype=torch.long)


def _collate_librispeech_ctc(batch):
    specs, labels = zip(*batch)
    specs = torch.stack(specs, dim=0)
    lengths = torch.tensor([len(label) for label in labels], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = torch.zeros(len(labels), max_len, dtype=torch.long)
    for i, label in enumerate(labels):
        padded[i, : len(label)] = label
    return specs, padded, lengths


def make_librispeech_ctc_loaders(
    *,
    root: str | Path = "./data/LibriSpeech",
    batch_size: int = 64,
    download: bool = True,
    n_mfcc: int = 64,
    num_workers: int = 0,
):
    root = Path(root)
    train_ds = LibriSpeechCTCDataset(root, "train-clean-100", download=download, n_mfcc=n_mfcc)
    val_ds = LibriSpeechCTCDataset(root, "dev-clean", download=download, n_mfcc=n_mfcc)
    test_ds = LibriSpeechCTCDataset(root, "test-clean", download=download, n_mfcc=n_mfcc)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=_collate_librispeech_ctc)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate_librispeech_ctc)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=_collate_librispeech_ctc)
    return train_loader, val_loader, test_loader, len(_LIBRISPEECH_CHARS)
