# Runner for DGI on Planetoid datasets (Cora by default)
# Mirrors ADP runners: config block, data pipeline, fit, evaluate, save

from dataclasses import dataclass
import torch
import torch.nn as nn
from pathlib import Path

try:
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T
except Exception:
    raise ImportError("Requires torch_geometric.")

from adp_gnn_dgi import DGI, TrainConfig, train_with_early_stop, evaluate_ssl, snapshot


@dataclass
class Config:
    dataset: str = 'Cora'
    data_root: str = './data'
    hidden: int = 256
    out_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    seed: int = 42
    save_path: str = './checkpoints/dgi_cora.pt'


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_data(cfg: Config):
    transform = T.NormalizeFeatures()
    data = Planetoid(cfg.data_root, cfg.dataset, transform=transform)[0]
    return data


def main():
    cfg = Config()
    set_seed(cfg.seed)

    data = get_data(cfg)
    in_dim = data.num_features

    model = DGI(
        in_dim=in_dim,
        hidden=cfg.hidden,
        out_dim=cfg.out_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
    )

    tcfg = TrainConfig()
    ran, best_loss, _ = train_with_early_stop(model, data, tcfg)
    val_loss, metrics = evaluate_ssl(model, data)

    Path(cfg.save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(snapshot(model), cfg.save_path)

    print({
        'epochs': ran,
        'best_ssl_loss': best_loss,
        'final_ssl_loss': val_loss,
        **metrics
    })


if __name__ == '__main__':
    main()
