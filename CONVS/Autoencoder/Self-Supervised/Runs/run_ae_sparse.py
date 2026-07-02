import os
import json
import random
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, random_split
import torchvision as tv
import torchvision.transforms as T

from ae_sparse import SparseConvAE, SparseAETrainer
from _common_real_image import make_real_image_loaders

# -----------------------------
# Configs ( runner style)
# -----------------------------

@dataclass
class DataConfig:
    root: str = "./data"
    batch_size: int = 256
    num_workers: int = 0
    val_split: float = 0.1
    seed: int = 2026

@dataclass
class ModelConfig:
    in_ch: int = 3
    widths: list = None
    pooling_indices: list = None
    mode: str = "l1"   # "l1" or "k"
    k: int = 0          # if 0, ignored and k_frac used
    k_frac: float = 0.05

@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    es_patience: int = 30
    max_epochs: int = 300
    results_dir: str = "results_ae_sparse"
    lam_l1: float = 1e-5


# -----------------------------
# Utils
# -----------------------------

def set_all_seeds(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_dataloaders(dc: DataConfig):
    return make_real_image_loaders(dc.root, dc.batch_size, val_ratio=dc.val_split, num_workers=dc.num_workers, image_size=32)


# -----------------------------
# Main
# -----------------------------

def main():
    dc = DataConfig()
    set_all_seeds(dc.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mc = ModelConfig(
        in_ch=3,
        widths=[32, 64, 128],
        pooling_indices=[0, 2],
        mode="l1",     # change to "k" for k-sparse
        k=0,            # set >0 for absolute k; if 0, k_frac is used
        k_frac=0.05,
    )

    tc = TrainConfig(
        lr=1e-3,
        weight_decay=1e-4,
        grad_clip=1.0,
        es_patience=30,
        max_epochs=300,
        results_dir="results_ae_sparse",
        lam_l1=1e-5,
    )

    os.makedirs(tc.results_dir, exist_ok=True)

    train_loader, val_loader, test_loader = make_dataloaders(dc)

    model = SparseConvAE(
        in_ch=mc.in_ch,
        widths=mc.widths,
        pooling_indices=mc.pooling_indices,
        mode=mc.mode,
        k=(None if mc.k == 0 else mc.k),
        k_frac=mc.k_frac,
    )

    trainer = SparseAETrainer(
        model=model,
        device=device,
        lr=tc.lr,
        weight_decay=tc.weight_decay,
        grad_clip=tc.grad_clip,
        es_patience=tc.es_patience,
        max_epochs=tc.max_epochs,
        results_dir=tc.results_dir,
        lam_l1=tc.lam_l1,
    )

    best_val = trainer.fit(train_loader, val_loader)
    test_loss = trainer.evaluate(test_loader)

    ckpt = os.path.join(tc.results_dir, "SparseAE_best.pth")
    trainer.save(ckpt)

    report = {
        "device": str(device),
        "best_val_mse": float(best_val),
        "test_mse": float(test_loss),
        "widths": model.widths_list(),
        "pooling_indices": list(model.pooling_indices),
        "total_neurons": model.total_neurons(),
        "depth": model.depth(),
        "mode": model.mode,
        "k": (None if model.k is None else int(model.k)),
        "k_frac": (None if model.k_frac is None else float(model.k_frac)),
        "lam_l1": tc.lam_l1,
    }
    with open(os.path.join(tc.results_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("==== Final Report ====")
    for k, v in report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
