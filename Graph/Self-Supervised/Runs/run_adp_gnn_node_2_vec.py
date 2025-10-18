# Runner for Node2Vec on Cora (Planetoid)

from dataclasses import dataclass
import torch

try:
    from torch_geometric.datasets import Planetoid
    import torch_geometric.transforms as T
except Exception:
    raise ImportError("Requires torch_geometric.")

from adp_gnn_node2vec import SkipGram, TrainConfig, build_adj, biased_walks, skipgram_pairs, train_with_early_stop, evaluate_ssl, snapshot


@dataclass
class Config:
    dataset: str = 'Cora'
    data_root: str = './data'
    embed_dim: int = 128
    p: float = 1.0
    q: float = 1.0
    seed: int = 42
    save_path: str = './checkpoints/node2vec_cora.pt'


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

    tcfg = TrainConfig(p=cfg.p, q=cfg.q)
    adj = build_adj(data.edge_index, data.num_nodes)
    walks = biased_walks(adj, tcfg.p, tcfg.q, tcfg.walk_length, tcfg.walks_per_node)
    pairs = skipgram_pairs(walks, tcfg.window_size)

    model = SkipGram(num_nodes=data.num_nodes, embed_dim=cfg.embed_dim)
    ran, best_loss, _ = train_with_early_stop(model, pairs, tcfg, data.num_nodes)
    val,_ = evaluate_ssl(model, pairs, tcfg, data.num_nodes)
    torch.save(snapshot(model), cfg.save_path)
    print({'epochs': ran, 'best_ssl_loss': best_loss, 'final_ssl_loss': val, 'pairs': len(pairs)})

if __name__=='__main__':
    main()
