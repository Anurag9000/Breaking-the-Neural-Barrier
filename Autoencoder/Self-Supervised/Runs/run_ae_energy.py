import os
import json
import random
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, random_split
import torchvision as tv
import torchvision.transforms as T

from ae_energy import EnergyConvAE, EnergyAETrainer

# -----------------------------
# Configs ( runner style)
# -----------------------------

@dataclass
class DataConfig:
    root: str = "./data"
    batch_size: int = 256
    num_workers: int = 4
    val_split: float = 0.1
    seed: int = 9090

@dataclass
class ModelConfig:
    in_ch: int = 3
    widths: list = None
    pooling_indices: list = None
    huber_beta: float = 0.1

@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    es_patience: int = 30
    max_epochs: int = 300
    results_dir: str = "results_ae_energy"
    # Energy hinge params
    margin: float = 0.5
    lambda_neg: float = 1.0
    neg_mode: str = "batch_permute"  # "batch_permute" | "gaussian" | "cutout"
    neg_strength: float = 0.5         # std for gaussian
    cutout_frac: float = 0.4          # area fraction for cutout

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
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    eval_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    full_train = tv.datasets.CIFAR10(root=dc.root, train=True, download=True, transform=train_tf)
    N = len(full_train)
    val_n = int(N * dc.val_split)
    train_n = N - val_n

    g = torch.Generator().manual_seed(dc.seed)
    train_set, val_set = random_split(full_train, [train_n, val_n], generator=g)

    test_set = tv.datasets.CIFAR10(root=dc.root, train=False, download=True, transform=eval_tf)

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
        widths=[32, 64, 128],
        pooling_indices=[0, 2],
        huber_beta=0.1,
    )

    tc = TrainConfig(
        lr=1e-3,
        weight_decay=1e-4,
        grad_clip=1.0,
        es_patience=30,
        max_epochs=300,
        results_dir="results_ae_energy",
        margin=0.5,
        lambda_neg=1.0,
        neg_mode="batch_permute",  # set to "gaussian" or "cutout" to change negatives
        neg_strength=0.5,
        cutout_frac=0.4,
    )

    os.makedirs(tc.results_dir, exist_ok=True)

    train_loader, val_loader, test_loader = make_dataloaders(dc)

    model = EnergyConvAE(
        in_ch=mc.in_ch,
        widths=mc.widths,
        pooling_indices=mc.pooling_indices,
        huber_beta=mc.huber_beta,
    )

    trainer = EnergyAETrainer(
        model=model,
        device=device,
        lr=tc.lr,
        weight_decay=tc.weight_decay,
        grad_clip=tc.grad_clip,
        es_patience=tc.es_patience,
        max_epochs=tc.max_epochs,
        results_dir=tc.results_dir,
        margin=tc.margin,
        lambda_neg=tc.lambda_neg,
        neg_mode=tc.neg_mode,
        neg_strength=tc.neg_strength,
        cutout_frac=tc.cutout_frac,
    )

    best_val = trainer.fit(train_loader, val_loader)
    test_pos = trainer.evaluate(test_loader)

    ckpt = os.path.join(tc.results_dir, "EnergyAE_best.pth")
    trainer.save(ckpt)

    report = {
        "device": str(device),
        "best_val_pos_energy": float(best_val),
        "test_pos_energy": float(test_pos),
        "widths": model.widths_list(),
        "pooling_indices": list(model.pooling_indices),
        "total_neurons": model.total_neurons(),
        "depth": model.depth(),
        "huber_beta": mc.huber_beta,
        "margin": tc.margin,
        "lambda_neg": tc.lambda_neg,
        "neg_mode": tc.neg_mode,
        "neg_strength": tc.neg_strength,
        "cutout_frac": tc.cutout_frac,
    }
    with open(os.path.join(tc.results_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("==== Final Report ====")
    for k, v in report.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
