# Runner for GCA on Planetoid (Cora) with degree-aware feature masking

from dataclasses import dataclass
import torch

try:
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T
    from torch_geometric.utils import degree
except Exception:
    raise ImportError("Requires torch_geometric.")

from adp_gnn_gca import GCA, TrainConfig, train_with_early_stop, evaluate_ssl, snapshot


@dataclass
class Config:
    dataset: str = 'Cora'
    data_root: str = './data'
    hidden: int = 256
    out_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.2
    seed: int = 42
    save_path: str = './checkpoints/gca_cora.pt'


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
    # Precompute degree for adaptive masking
    N = data.num_nodes
    deg = degree(data.edge_index[0], num_nodes=N)
    data.deg = deg
    return data


def main():
    cfg = Config()
    set_seed(cfg.seed)

    data = get_data(cfg)
    in_dim = data.num_features

    model = GCA(in_dim, cfg.hidden, cfg.out_dim, cfg.num_layers, cfg.dropout)
    tcfg = TrainConfig()

    ran, best_loss, _ = train_with_early_stop(model, data, tcfg)
    val, _ = evaluate_ssl(model, data)
    torch.save(snapshot(model), cfg.save_path)
    print({'epochs': ran, 'best_ssl_loss': best_loss, 'final_ssl_loss': val})


if __name__ == '__main__':
    main()
