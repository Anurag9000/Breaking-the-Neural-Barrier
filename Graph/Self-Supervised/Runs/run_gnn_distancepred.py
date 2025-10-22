# Runner for Distance/Position Prediction on Cora

from dataclasses import dataclass
import torch

try:
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T
except Exception:
    raise ImportError("Requires torch_geometric.")

from adp_gnn_distancepred import DistancePred, TrainConfig, train_with_early_stop, evaluate_ssl, snapshot


@dataclass
class Config:
    dataset: str = 'Cora'
    data_root: str = './data'
    hidden: int = 128
    out_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    num_anchors: int = 8
    max_bin: int = 4
    seed: int = 42
    save_path: str = './checkpoints/distancepred_cora.pt'


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def get_data(cfg: Config):
    transform = T.NormalizeFeatures()
    data = Planetoid(cfg.data_root, cfg.dataset, transform=transform)[0]
    return data


def main():
    cfg = Config(); set_seed(cfg.seed)
    data = get_data(cfg)

    model = DistancePred(data.num_features, cfg.hidden, cfg.out_dim, cfg.num_layers, cfg.dropout, cfg.num_anchors, cfg.max_bin)
    tcfg = TrainConfig()

    ran, best_loss, _ = train_with_early_stop(model, data, tcfg)
    val,_ = evaluate_ssl(model, data)
    torch.save(snapshot(model), cfg.save_path)
    print({'epochs': ran, 'best_ssl_loss': best_loss, 'final_ssl_loss': val})

if __name__=='__main__':
    main()
