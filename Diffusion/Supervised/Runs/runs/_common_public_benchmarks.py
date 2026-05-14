from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as nnF
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets
from torchvision.transforms import functional as TVF
from sklearn.feature_extraction.text import HashingVectorizer
import torchaudio


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _pil_to_normalized_tensor(image) -> torch.Tensor:
    x = TVF.pil_to_tensor(image).float() / 255.0
    if x.size(0) == 1:
        return x
    mean = IMAGENET_MEAN.to(x.device)
    std = IMAGENET_STD.to(x.device)
    return (x - mean) / std


def _resize_tensor(x: torch.Tensor, size: int | Tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    if isinstance(size, int):
        size = (size, size)
    if x.dim() == 3:
        if mode == "nearest":
            return nnF.interpolate(x.unsqueeze(0), size=size, mode=mode).squeeze(0)
        return nnF.interpolate(x.unsqueeze(0), size=size, mode=mode, align_corners=False).squeeze(0)
    if x.dim() == 2:
        if mode == "nearest":
            return nnF.interpolate(x.unsqueeze(0).unsqueeze(0).float(), size=size, mode=mode).squeeze(0).squeeze(0)
        return nnF.interpolate(x.unsqueeze(0).unsqueeze(0).float(), size=size, mode=mode, align_corners=False).squeeze(0).squeeze(0)
    raise ValueError(f"Unexpected tensor rank {x.dim()}")


def _resize_flow(flow: torch.Tensor, size: int | Tuple[int, int]) -> torch.Tensor:
    if isinstance(size, int):
        size = (size, size)
    old_h, old_w = flow.shape[-2:]
    new_h, new_w = size
    scale_x = float(new_w) / float(old_w)
    scale_y = float(new_h) / float(old_h)
    resized = nnF.interpolate(flow.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)
    resized = resized.clone()
    resized[0] *= scale_x
    resized[1] *= scale_y
    return resized


def _split_dataset(dataset: Dataset, train_frac: float = 0.8, val_frac: float = 0.1):
    n = len(dataset)
    n_train = max(1, int(n * train_frac))
    n_val = max(1, int(n * val_frac))
    n_test = max(1, n - n_train - n_val)
    if n_train + n_val + n_test > n:
        n_train = n - n_val - n_test
    return random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(0))


class FlyingChairsTripletDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train", image_size: int = 128):
        self.dataset = datasets.FlyingChairs(root=str(root), split=split, transforms=None)
        self.image_size = image_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img1, img2, flow = self.dataset[index]
        img1 = _resize_tensor(_pil_to_normalized_tensor(img1), self.image_size)
        img2 = _resize_tensor(_pil_to_normalized_tensor(img2), self.image_size)
        pair = torch.cat([img1, img2], dim=0)
        flow_t = torch.from_numpy(np.asarray(flow)).float()
        flow_t = _resize_flow(flow_t, self.image_size)
        return pair, flow_t


class VocSegmentationDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        year: str = "2012",
        image_set: str = "train",
        image_size: int = 256,
        target_mode: str = "onehot",
        num_classes: int = 21,
    ):
        self.dataset = datasets.VOCSegmentation(
            root=str(root),
            year=year,
            image_set=image_set,
            download=False,
        )
        self.image_size = image_size
        self.target_mode = target_mode
        self.num_classes = num_classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, target = self.dataset[index]
        image = _resize_tensor(_pil_to_normalized_tensor(image), self.image_size)
        mask = torch.from_numpy(np.array(target, dtype=np.int64))
        mask = mask.clone()
        mask[mask == 255] = 0
        mask = _resize_tensor(mask, self.image_size, mode="nearest").long()
        if self.target_mode == "onehot":
            target_tensor = torch.nn.functional.one_hot(mask.clamp(0, self.num_classes - 1), num_classes=self.num_classes).permute(2, 0, 1).float()
        elif self.target_mode == "mask":
            target_tensor = mask.float().unsqueeze(0) / float(max(1, self.num_classes - 1))
        else:
            raise ValueError(f"Unknown target_mode={self.target_mode}")
        return image, target_tensor


class CocoCaptionDataset(Dataset):
    def __init__(self, root: str | Path, ann_file: str | Path, image_size: int = 224, text_dim: int = 512):
        self.dataset = datasets.CocoCaptions(root=str(root), annFile=str(ann_file), transform=None)
        self.image_size = image_size
        self.vectorizer = HashingVectorizer(
            n_features=text_dim,
            alternate_sign=False,
            lowercase=True,
            norm=None,
            ngram_range=(1, 2),
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, captions = self.dataset[index]
        image = _resize_tensor(_pil_to_normalized_tensor(image), self.image_size)
        caption = captions[0] if isinstance(captions, (list, tuple)) and captions else str(captions)
        text_vec = self.vectorizer.transform([caption]).toarray()[0].astype(np.float32)
        return image, torch.from_numpy(text_vec)


class CocoKeypointDataset(Dataset):
    def __init__(self, root: str | Path, ann_file: str | Path, image_size: int = 256):
        self.dataset = datasets.CocoDetection(root=str(root), annFile=str(ann_file))
        self.image_size = image_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, targets = self.dataset[index]
        orig_w, orig_h = image.size
        image = _resize_tensor(_pil_to_normalized_tensor(image), self.image_size)

        kp = None
        for ann in targets:
            keypoints = ann.get("keypoints")
            if keypoints and ann.get("num_keypoints", 0) > 0:
                kp = torch.tensor(keypoints, dtype=torch.float32).view(-1, 3)
                break

        if kp is None or kp.numel() == 0:
            return self[(index + 1) % len(self)]

        heatmaps = torch.zeros(kp.size(0), self.image_size, self.image_size, dtype=torch.float32)
        sigma = 2.0
        yy = torch.arange(self.image_size, dtype=torch.float32).view(-1, 1)
        xx = torch.arange(self.image_size, dtype=torch.float32).view(1, -1)
        # Scale keypoints from the original annotation space to the resized image space.
        # CocoDetection returns images before resizing, so the annotation coordinates are still valid.
        scale_x = self.image_size / max(orig_w, 1.0)
        scale_y = self.image_size / max(orig_h, 1.0)
        for j, (x, y, v) in enumerate(kp):
            if v <= 0:
                continue
            cx = x * scale_x
            cy = y * scale_y
            heatmaps[j] = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        return image, heatmaps


class LibriSpeechWaveformDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        url: str = "train-clean-100",
        segment_seconds: float = 3.0,
        sample_rate: int = 16000,
        train: bool = True,
    ):
        self.dataset = torchaudio.datasets.LIBRISPEECH(root=str(root), url=url, download=False)
        self.segment_length = int(segment_seconds * sample_rate)
        self.sample_rate = sample_rate
        self.train = train

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        waveform, sr, transcript, speaker_id, chapter_id, utterance_id = self.dataset[index]
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.size(-1) < self.segment_length:
            waveform = nnF.pad(waveform, (0, self.segment_length - waveform.size(-1)))
        elif waveform.size(-1) > self.segment_length:
            if self.train:
                max_start = waveform.size(-1) - self.segment_length
                start = torch.randint(0, max_start + 1, (1,)).item()
            else:
                start = (waveform.size(-1) - self.segment_length) // 2
            waveform = waveform[..., start : start + self.segment_length]
        return waveform.float(), transcript


class LibriSpeechSpecDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        url: str = "train-clean-100",
        segment_seconds: float = 3.0,
        sample_rate: int = 16000,
        n_mels: int = 128,
        train: bool = True,
    ):
        self.waveforms = LibriSpeechWaveformDataset(
            root=root,
            url=url,
            segment_seconds=segment_seconds,
            sample_rate=sample_rate,
            train=train,
        )
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
        self.db = torchaudio.transforms.AmplitudeToDB(stype="power")

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, index):
        waveform, transcript = self.waveforms[index]
        spec = self.db(self.mel(waveform))
        return spec, transcript


class UCF101TripletDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        annotation_path: str | Path,
        frames_per_clip: int = 7,
        train: bool = True,
        image_size: int = 128,
        fold: int = 1,
    ):
        self.dataset = datasets.UCF101(
            root=str(root),
            annotation_path=str(annotation_path),
            frames_per_clip=frames_per_clip,
            step_between_clips=max(1, frames_per_clip),
            fold=fold,
            train=train,
            output_format="TCHW",
        )
        self.image_size = image_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        video, _audio, _label = self.dataset[index]
        video = video.float()
        if video.max() > 1:
            video = video / 255.0
        frames = []
        for frame in video:
            frames.append(_resize_tensor(frame, self.image_size))
        video = torch.stack(frames, dim=0)
        prev = video[0]
        mid = video[len(video) // 2]
        nxt = video[-1]
        return prev, mid, nxt


class UCF101ClipDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        annotation_path: str | Path,
        frames_per_clip: int = 16,
        train: bool = True,
        image_size: int = 64,
        fold: int = 1,
    ):
        self.dataset = datasets.UCF101(
            root=str(root),
            annotation_path=str(annotation_path),
            frames_per_clip=frames_per_clip,
            step_between_clips=max(1, frames_per_clip),
            fold=fold,
            train=train,
            output_format="TCHW",
        )
        self.image_size = image_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        video, _audio, _label = self.dataset[index]
        video = video.float()
        if video.max() > 1:
            video = video / 255.0
        frames = []
        for frame in video:
            frames.append(_resize_tensor(frame, self.image_size))
        video = torch.stack(frames, dim=0)
        return video.permute(1, 0, 2, 3)


class PairedImageFolderDataset(Dataset):
    def __init__(
        self,
        source_root: str | Path,
        target_root: str | Path,
        image_size: int = 128,
    ):
        self.source = datasets.ImageFolder(str(source_root))
        self.target = datasets.ImageFolder(str(target_root))
        self.image_size = image_size

    def __len__(self):
        return max(len(self.source), len(self.target))

    def __getitem__(self, index):
        src_img, _ = self.source[index % len(self.source)]
        tgt_img, _ = self.target[index % len(self.target)]
        src = _resize_tensor(_pil_to_normalized_tensor(src_img), self.image_size)
        tgt = _resize_tensor(_pil_to_normalized_tensor(tgt_img), self.image_size)
        return src, tgt
