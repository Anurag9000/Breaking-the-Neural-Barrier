# Runner for SimSiam-G on PROTEINS

from dataclasses import dataclass
from pathlib import Path
import torch

try:
    from torch_geometric.datasets import TUDataset
    import torch_geometric.transforms as T
    from torch_geometric.loader import DataLoader
except Exception:
    raise ImportError("Requires torch_geometric.")

from adp_gnn_simsiam import SimSiamG, TrainConfig, train_with_early_stop, evaluate_ssl, snapshot


@dataclass
class Config:
    dataset: str = 'PROTEINS'
    data_root: str = './data'
    hidden: int = 256
    out_dim: int = 256
    proj_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.2
    batch_size: int = 64
    seed: int = 42
    save_path: str = './checkpoints/simsiam_proteins.pt'


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def get_loader(cfg: Config):
    transform = T.NormalizeFeatures()
    ds = TUDataset(cfg.data_root, name=cfg.dataset, use_node_attr=True, transform=transform)
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)


def main():
    cfg = Config(); set_seed(cfg.seed)
    loader = get_loader(cfg)
    in_dim = loader.dataset.num_features

    model = SimSiamG(in_dim, cfg.hidden, cfg.out_dim, cfg.proj_dim, cfg.num_layers, cfg.dropout)
    tcfg = TrainConfig()

    best_loss=float('inf'); best_snap=None; ran_total=0
    for batch in loader:
        ran, loss, _ = train_with_early_stop(model, batch, tcfg)
        ran_total += ran
        val,_ = evaluate_ssl(model, batch)
        if val < best_loss:
            best_loss=val; best_snap=snapshot(model)
    if best_snap is not None:
        Path(cfg.save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_snap, cfg.save_path)
    print({'epochs': ran_total, 'best_ssl_loss': best_loss})

if __name__=='__main__':
    main()
