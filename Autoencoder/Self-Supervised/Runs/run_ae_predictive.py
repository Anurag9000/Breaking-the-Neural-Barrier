import os
import json
import random
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, random_split
import torchvision as tv
import torchvision.transforms as T

from ae_predictive import PredictiveSeqAE, PredictiveSeqAETrainer
from _common_real_image import make_real_image_loaders

# -----------------------------
# CIFAR-10 as sequences: (B, C, H, W) -> (B, T=H, F=C*W)
# -----------------------------

class ImageRowsAsSequence(torch.utils.data.Dataset):
    def __init__(self, base_ds):
        self.base = base_ds
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        x, y = self.base[idx]
        C, H, W = x.shape
        seq = x.permute(1, 0, 2).contiguous().view(H, C * W)  # (T=H, F=C*W)
        return seq, y

# -----------------------------
# Configs ( runner style)
# -----------------------------

@dataclass
class DataConfig:
    root: str = "./data"
    batch_size: int = 128
    num_workers: int = 4
    val_split: float = 0.1
    seed: int = 2718

@dataclass
class ModelConfig:
    hidden_size: int = 256
    num_layers: int = 2

@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    es_patience: int = 30
    max_epochs: int = 200
    results_dir: str = "results_ae_predictive"
    loss_type: str = "mse"

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
    base_train, base_val, base_test = make_real_image_loaders(
        dc.root, dc.batch_size, val_ratio=dc.val_split, num_workers=dc.num_workers, image_size=32
    )
    train_set = ImageRowsAsSequence(base_train.dataset)
    val_set = ImageRowsAsSequence(base_val.dataset)
    test_set = ImageRowsAsSequence(base_test.dataset)
    train_loader = DataLoader(train_set, batch_size=dc.batch_size, shuffle=True, num_workers=dc.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=dc.batch_size, shuffle=False, num_workers=dc.num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=dc.batch_size, shuffle=False, num_workers=dc.num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader, 32, 3*32

# -----------------------------
# Main
# -----------------------------

def main():
    dc = DataConfig()
    set_all_seeds(dc.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, T_len, F_dim = make_dataloaders(dc)

    mc = ModelConfig(
        hidden_size=256,
        num_layers=2,
    )

    model = PredictiveSeqAE(
        feature_dim=F_dim,
        hidden_size=mc.hidden_size,
        num_layers=mc.num_layers,
        dropout=0.1,
    )

    tc = TrainConfig(
        lr=1e-3,
        weight_decay=1e-4,
        grad_clip=1.0,
        es_patience=30,
        max_epochs=200,
        results_dir="results_ae_predictive",
        loss_type="mse",
    )

    os.makedirs(tc.results_dir, exist_ok=True)

    trainer = PredictiveSeqAETrainer(
        model=model,
        device=device,
        lr=tc.lr,
        weight_decay=tc.weight_decay,
        grad_clip=tc.grad_clip,
        es_patience=tc.es_patience,
        max_epochs=tc.max_epochs,
        results_dir=tc.results_dir,
        loss_type=tc.loss_type,
    )

    best_val = trainer.fit(train_loader, val_loader)
    test_loss = trainer.evaluate(test_loader)

    ckpt = os.path.join(tc.results_dir, "PredictiveSeqAE_best.pth")
    trainer.save(ckpt)

    report = {
        "device": str(device),
        "best_val_nextstep": float(best_val),
        "test_nextstep": float(test_loss),
        "feature_dim": int(F_dim),
        "T_len": int(T_len),
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "total_neurons": model.total_neurons(),
        "depth": model.depth(),
    }
    with open(os.path.join(tc.results_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("==== Final Report ====")
    for k, v in report.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
