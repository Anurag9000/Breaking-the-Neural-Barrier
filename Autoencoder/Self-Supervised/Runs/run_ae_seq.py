import os
import json
import random
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, random_split
import torchvision as tv
import torchvision.transforms as T

from ae_seq import SeqAE, SeqAETrainer

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
        # x: (C,H,W) tensor
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
    seed: int = 31415

@dataclass
class ModelConfig:
    hidden_size: int = 256
    num_layers: int = 2
    bidirectional: bool = False

@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    es_patience: int = 30
    max_epochs: int = 200
    results_dir: str = "results_ae_seq"
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

    test_base = tv.datasets.CIFAR10(root=dc.root, train=False, download=True, transform=eval_tf)

    train_set = ImageRowsAsSequence(train_set)
    val_set = ImageRowsAsSequence(val_set)
    test_set = ImageRowsAsSequence(test_base)

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
        bidirectional=False,
    )

    model = SeqAE(
        feature_dim=F_dim,
        hidden_size=mc.hidden_size,
        num_layers=mc.num_layers,
        dropout=0.1,
        bidirectional=mc.bidirectional,
    )

    tc = TrainConfig(
        lr=1e-3,
        weight_decay=1e-4,
        grad_clip=1.0,
        es_patience=30,
        max_epochs=200,
        results_dir="results_ae_seq",
        loss_type="mse",
    )

    os.makedirs(tc.results_dir, exist_ok=True)

    trainer = SeqAETrainer(
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

    ckpt = os.path.join(tc.results_dir, "SeqAE_best.pth")
    trainer.save(ckpt)

    report = {
        "device": str(device),
        "best_val_recon": float(best_val),
        "test_recon": float(test_loss),
        "feature_dim": int(F_dim),
        "T_len": int(T_len),
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "bidirectional": model.bidirectional,
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
