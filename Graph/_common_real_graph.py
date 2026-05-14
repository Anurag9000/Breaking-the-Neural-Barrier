from __future__ import annotations

from typing import Tuple

import torch

try:
    from torch_geometric.datasets import TUDataset
    from torch_geometric.loader import DataLoader
    PYG_OK = True
except Exception as exc:  # pragma: no cover - import guard for environments without PyG
    PYG_OK = False
    PYG_ERR = exc


def _ensure_node_features(ds):
    if getattr(ds, 'num_features', 0) > 0:
        return ds
    for graph in ds:
        if getattr(graph, 'x', None) is not None:
            continue
        deg = torch.bincount(graph.edge_index[0], minlength=graph.num_nodes).float().unsqueeze(1)
        graph.x = deg
    return ds


def make_real_graph_loaders(
    dataset_name: str = 'PROTEINS',
    batch_size: int = 8,
    *,
    root: str = './data',
    val_split: float = 0.1,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    if not PYG_OK:
        raise RuntimeError(f'torch_geometric is required for real graph loaders. Import error: {PYG_ERR}')
    ds = TUDataset(root=root, name=dataset_name)
    ds = _ensure_node_features(ds)
    n = len(ds)
    n_val = max(1, int(n * val_split))
    n_train = max(1, n - n_val)
    train_ds = ds[:n_train]
    val_ds = ds[n_train:]
    test_ds = ds[n_train:]
    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dl_test = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    in_dim = int(ds.num_features) if getattr(ds, 'num_features', 0) > 0 else 1
    return dl_train, dl_val, dl_test, in_dim
