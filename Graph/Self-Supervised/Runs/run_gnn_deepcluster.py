# Runner for DeepCluster-G on PROTEINS (graph-level)

from dataclasses import dataclass
from pathlib import Path
import torch

try:
    from torch_geometric.datasets import TUDataset
    import torch_geometric.transforms as T
    from torch_geometric.loader import DataLoader
except Exception:
    raise ImportError("Requires torch_geometric.")

from adp_gnn_deepcluster import DeepClusterG, TrainConfig, train_with_early_stop, evaluate_ssl, snapshot


@dataclass
class Config:
    dataset: str = 'PROTEINS'
    data_root: str = './data'
    hidden: int = 256
    rep_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.2
    num_clusters: int = 50
    batch_size: int = 128
    seed: int = 42
    save_path: str = './checkpoints/deepcluster_proteins.pt'


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

    model = DeepClusterG(in_dim, cfg.hidden, cfg.rep_dim, cfg.num_clusters, cfg.num_layers, cfg.dropout)
    tcfg = TrainConfig()

    # Fit epoch-wise over the loader (labels reassigned inside train function using current batch)
    best_metric=float('inf'); best_snap=None; ran_total=0
    for batch in loader:
        ran, metric, _ = train_with_early_stop(model, batch, tcfg)
        ran_total += ran
        val,_ = evaluate_ssl(model, batch)
        if val < best_metric:
            best_metric=val; best_snap=snapshot(model)

    if best_snap is not None:
        Path(cfg.save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_snap, cfg.save_path)
    print({'epochs': ran_total, 'best_metric': best_metric})

if __name__=='__main__':
    main()
