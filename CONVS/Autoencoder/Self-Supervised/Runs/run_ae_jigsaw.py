import os
import json
import random
from dataclasses import dataclass
from itertools import permutations

import torch
from torch.utils.data import DataLoader, random_split
import torchvision as tv
import torchvision.transforms as T

from ae_jigsaw import JigsawModel, JigsawTrainer
from _common_real_image import make_real_image_loaders

# -----------------------------
# Jigsaw dataset wrapper
# -----------------------------

class JigsawDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, grid_size: int = 3, perm_set_size: int = 30, seed: int = 123, patch_jitter: int = 2):
        self.base = base_ds
        self.G = grid_size
        self.K = self.G * self.G
        self.rng = random.Random(seed)
        self.patch_jitter = int(patch_jitter)
        # Build a fixed permutation set (subset of all K! to keep it tractable)
        all_idx = list(range(self.K))
        # Use a deterministic selection of permutations seeded for reproducibility
        # Strategy: shuffle a long list of random perms then take first perm_set_size unique
        seen = set()
        perms = []
        while len(perms) < perm_set_size:
            p = tuple(self.rng.sample(all_idx, self.K))
            if p not in seen:
                seen.add(p)
                perms.append(p)
        self.permutations = perms  # list of tuples of length K

    def __len__(self):
        return len(self.base)

    def _to_patches(self, x):
        # x: (C,H,W) -> list of K patches (C,h,w)
        C, H, W = x.shape
        g = self.G
        h = H // g
        w = W // g
        patches = []
        for gy in range(g):
            for gx in range(g):
                y0 = gy*h
                x0 = gx*w
                patch = x[:, y0:y0+h, x0:x0+w]
                # optional jitter crop for augmentation
                if self.patch_jitter > 0 and h > 2*self.patch_jitter and w > 2*self.patch_jitter:
                    dy = random.randint(-self.patch_jitter, self.patch_jitter)
                    dx = random.randint(-self.patch_jitter, self.patch_jitter)
                    patch = patch[:, self.patch_jitter+dy:h-self.patch_jitter+dy, self.patch_jitter+dx:w-self.patch_jitter+dx]
                    patch = torch.nn.functional.interpolate(patch.unsqueeze(0), size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
                patches.append(patch)
        return patches  # length K

    def __getitem__(self, idx):
        x, _ = self.base[idx]
        patches = self._to_patches(x)
        # choose a permutation id
        pid = self.rng.randrange(len(self.permutations))
        perm = self.permutations[pid]
        shuffled = [patches[i] for i in perm]
        # stack into (K,C,h,w)
        stacked = torch.stack(shuffled, dim=0)
        return stacked, pid

# -----------------------------
# Configs ( runner style)
# -----------------------------

@dataclass
class DataConfig:
    root: str = "./data"
    batch_size: int = 256
    num_workers: int = 4
    val_split: float = 0.1
    seed: int = 123

@dataclass
class ModelConfig:
    in_ch: int = 3
    grid_size: int = 3
    width: int = 64
    num_permutations: int = 30

@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    es_patience: int = 30
    max_epochs: int = 200
    results_dir: str = "results_ae_jigsaw"

# -----------------------------
# Utils
# -----------------------------

def set_all_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_dataloaders(dc: DataConfig, mc: ModelConfig):
    train_loader, val_loader, test_loader = make_real_image_loaders(
        dc.root, dc.batch_size, val_ratio=dc.val_split, num_workers=dc.num_workers, image_size=32
    )
    train_set = JigsawDataset(train_loader.dataset, grid_size=mc.grid_size, perm_set_size=mc.num_permutations, seed=dc.seed)
    val_set = JigsawDataset(val_loader.dataset, grid_size=mc.grid_size, perm_set_size=mc.num_permutations, seed=dc.seed + 1)
    test_set = JigsawDataset(test_loader.dataset, grid_size=mc.grid_size, perm_set_size=mc.num_permutations, seed=dc.seed + 2)
    train_loader = DataLoader(train_set, batch_size=dc.batch_size, shuffle=True, num_workers=dc.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=dc.batch_size, shuffle=False, num_workers=dc.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=dc.batch_size, shuffle=False, num_workers=dc.num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader

# -----------------------------
# Main
# -----------------------------

def main():
    dc = DataConfig()
    set_all_seeds(dc.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mc = ModelConfig(
        in_ch=3,
        grid_size=3,
        width=64,
        num_permutations=30,
    )

    tc = TrainConfig(
        lr=1e-3,
        weight_decay=1e-4,
        grad_clip=1.0,
        es_patience=30,
        max_epochs=200,
        results_dir="results_ae_jigsaw",
    )

    os.makedirs(tc.results_dir, exist_ok=True)

    train_loader, val_loader, test_loader = make_dataloaders(dc, mc)

    model = JigsawModel(
        in_ch=mc.in_ch,
        grid_size=mc.grid_size,
        width=mc.width,
        num_permutations=mc.num_permutations,
    )

    trainer = JigsawTrainer(
        model=model,
        device=device,
        lr=tc.lr,
        weight_decay=tc.weight_decay,
        grad_clip=tc.grad_clip,
        es_patience=tc.es_patience,
        max_epochs=tc.max_epochs,
        results_dir=tc.results_dir,
    )

    best_val = trainer.fit(train_loader, val_loader)
    test_loss, test_acc = trainer.evaluate(test_loader)

    ckpt = os.path.join(tc.results_dir, "Jigsaw_best.pth")
    trainer.save(ckpt)

    report = {
        "device": str(device),
        "best_val_ce": float(best_val),
        "test_ce": float(test_loss),
        "test_acc": float(test_acc),
        "grid_size": mc.grid_size,
        "num_permutations": mc.num_permutations,
        "total_neurons": model.total_neurons(),
        "depth": model.depth(),
        "widths": model.widths_list(),
    }
    with open(os.path.join(tc.results_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("==== Final Report ====")
    for k, v in report.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
