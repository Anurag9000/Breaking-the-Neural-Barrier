import os
import json
import random
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, random_split
import torchvision as tv
import torchvision.transforms as T

from ae_masked import MaskedConvAE, MaskedAETrainer
from _common_real_image import make_real_image_loaders

# -----------------------------
# Configs ( runner style)
# -----------------------------

@dataclass
class DataConfig:
    root: str = "./data"
    batch_size: int = 256
    num_workers: int = 4
    val_split: float = 0.1
    seed: int = 2025

@dataclass
class ModelConfig:
    in_ch: int = 3
    widths: list = None
    pooling_indices: list = None

@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    es_patience: int = 30
    max_epochs: int = 300
    results_dir: str = "results_ae_masked"
    mask_ratio: float = 0.6
    patch_size: int = 4


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
    )

    tc = TrainConfig(
        lr=1e-3,
        weight_decay=1e-4,
        grad_clip=1.0,
        es_patience=30,
        max_epochs=300,
        results_dir="results_ae_masked",
        mask_ratio=0.6,
        patch_size=4,
    )

    os.makedirs(tc.results_dir, exist_ok=True)

    train_loader, val_loader, test_loader = make_dataloaders(dc)

    model = MaskedConvAE(
        in_ch=mc.in_ch,
        widths=mc.widths,
        pooling_indices=mc.pooling_indices,
    )

    trainer = MaskedAETrainer(
        model=model,
        device=device,
        lr=tc.lr,
        weight_decay=tc.weight_decay,
        grad_clip=tc.grad_clip,
        es_patience=tc.es_patience,
        max_epochs=tc.max_epochs,
        results_dir=tc.results_dir,
        mask_ratio=tc.mask_ratio,
        patch_size=tc.patch_size,
    )

    best_val = trainer.fit(train_loader, val_loader)
    test_loss = trainer.evaluate(test_loader)

    ckpt = os.path.join(tc.results_dir, "MAE_best.pth")
    trainer.save(ckpt)

    report = {
        "device": str(device),
        "best_val_masked_mse": float(best_val),
        "test_masked_mse": float(test_loss),
        "widths": model.widths_list(),
        "pooling_indices": list(model.pooling_indices),
        "total_neurons": model.total_neurons(),
        "depth": model.depth(),
        "mask_ratio": tc.mask_ratio,
        "patch_size": tc.patch_size,
    }
    with open(os.path.join(tc.results_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("==== Final Report ====")
    for k, v in report.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
