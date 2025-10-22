# Runner for LINE on Cora (Planetoid)

from dataclasses import dataclass
import torch

try:
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T
except Exception:
    raise ImportError("Requires torch_geometric.")

from adp_gnn_line import LINE, TrainConfig, edges_to_pairs, train_with_early_stop, evaluate_ssl, snapshot


@dataclass
class Config:
    dataset: str = 'Cora'
    data_root: str = './data'
    embed_dim: int = 128
    order: str = 'both'
    seed: int = 42
    save_path: str = './checkpoints/line_cora.pt'


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

    pairs = edges_to_pairs(data.edge_index)
    model = LINE(num_nodes=data.num_nodes, embed_dim=cfg.embed_dim, order=cfg.order)

    tcfg = TrainConfig()
    ran, best_loss, _ = train_with_early_stop(model, pairs, tcfg, data.num_nodes)
    val,_ = evaluate_ssl(model, pairs, tcfg, data.num_nodes)
    torch.save(snapshot(model), cfg.save_path)
    print({'epochs': ran, 'best_ssl_loss': best_loss, 'final_ssl_loss': val, 'pairs': pairs.size(0)})

if __name__=='__main__':
    main()
