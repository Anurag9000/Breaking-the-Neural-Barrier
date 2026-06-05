from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import urllib.request

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split

from sklearn.cluster import KMeans
from sklearn.datasets import fetch_california_housing, fetch_covtype, fetch_openml
from sklearn.metrics import normalized_mutual_info_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


@dataclass
class Task:
    name: str
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    in_dim: int
    out_dim: int
    task_type: str  # classification | regression | reconstruction
    loss_fn: Callable
    metrics_fn: Optional[Callable]
    extra: Dict[str, float]


def _split_dataset(ds: Dataset, seed: int, val_split: float = 0.1, test_split: float = 0.1):
    n = len(ds)
    n_val = int(n * val_split)
    n_test = int(n * test_split)
    n_train = n - n_val - n_test
    g = torch.Generator().manual_seed(int(seed))
    return random_split(ds, [n_train, n_val, n_test], generator=g)


def _resolve_pin_memory(pin_memory: Optional[bool]) -> bool:
    if pin_memory is None:
        return bool(torch.cuda.is_available())
    return bool(pin_memory)


def _make_loaders(
    ds: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    pin_memory: Optional[bool] = None,
):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=_resolve_pin_memory(pin_memory),
    )


def clone_loader(
    loader: DataLoader,
    batch_size: int,
    shuffle: bool,
    generator: Optional[torch.Generator] = None,
) -> DataLoader:
    return DataLoader(
        loader.dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(loader.num_workers),
        pin_memory=bool(getattr(loader, "pin_memory", False)),
        drop_last=bool(getattr(loader, "drop_last", False)),
        collate_fn=getattr(loader, "collate_fn", None),
        generator=generator,
    )


def refresh_task_loaders(task: Task, batch_size: int) -> None:
    task.train_loader = clone_loader(task.train_loader, batch_size, shuffle=True)
    task.val_loader = clone_loader(task.val_loader, batch_size, shuffle=False)
    task.test_loader = clone_loader(task.test_loader, batch_size, shuffle=False)


def _split_numpy_arrays(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    val_split: float = 0.1,
    test_split: float = 0.1,
):
    n = int(x.shape[0])
    n_val = int(n * val_split)
    n_test = int(n * test_split)
    n_train = n - n_val - n_test
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train : n_train + n_val]
    test_idx = perm[n_train + n_val :]
    return (
        x[train_idx],
        y[train_idx],
        x[val_idx],
        y[val_idx],
        x[test_idx],
        y[test_idx],
    )


def _standardize_from_train(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    val_x = scaler.transform(val_x)
    test_x = scaler.transform(test_x)
    return (
        train_x.astype(np.float32),
        val_x.astype(np.float32),
        test_x.astype(np.float32),
    )


class ArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        if np.asarray(y).ndim == 1:
            self.y = torch.as_tensor(y, dtype=torch.long)
        else:
            self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return int(self.x.size(0))

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class PairArrayDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return int(self.x.size(0))

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class OneClassAnomalyDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, anomaly_label: int):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.anomaly_label = int(anomaly_label)

    def __len__(self):
        return int(self.x.size(0))

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], torch.tensor(self.anomaly_label, dtype=torch.long)


def _data_home_from_dir(data_dir: str) -> str:
    path = Path(data_dir).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _load_covtype(seed: int, data_dir: str):
    data = fetch_covtype(download_if_missing=True, data_home=_data_home_from_dir(data_dir))
    x = np.asarray(data.data, dtype=np.float32)
    y = np.asarray(data.target, dtype=np.int64) - 1
    return _split_numpy_arrays(x, y, seed)


def _load_year_prediction(seed: int, data_dir: str):
    cache_dir = Path(_data_home_from_dir(data_dir)) / "year_prediction_msd"
    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "YearPredictionMSD.txt.zip"
    try:
        if not zip_path.exists():
            urllib.request.urlretrieve(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip",
                zip_path,
            )
        frame = pd.read_csv(zip_path, header=None, compression="zip")
        y = frame.iloc[:, 0].to_numpy(dtype=np.float32).reshape(-1, 1)
        x = frame.iloc[:, 1:].to_numpy(dtype=np.float32)
    except Exception:
        try:
            data = fetch_openml(data_id=46672, as_frame=False, data_home=_data_home_from_dir(data_dir))
            x = np.asarray(data.data, dtype=np.float32)
            y = np.asarray(data.target, dtype=np.float32).reshape(-1, 1)
        except Exception:
            try:
                data = fetch_openml(name="Year_Prediction_MSD", version=1, as_frame=False, data_home=_data_home_from_dir(data_dir))
                x = np.asarray(data.data, dtype=np.float32)
                y = np.asarray(data.target, dtype=np.float32).reshape(-1, 1)
            except Exception:
                data = fetch_california_housing(data_home=_data_home_from_dir(data_dir))
                x = np.asarray(data.data, dtype=np.float32)
                y = np.asarray(data.target, dtype=np.float32).reshape(-1, 1)
    return _split_numpy_arrays(x, y, seed)


def _load_california_housing(seed: int, data_dir: str):
    data = fetch_california_housing(data_home=_data_home_from_dir(data_dir))
    x = np.asarray(data.data, dtype=np.float32)
    y = np.asarray(data.target, dtype=np.float32).reshape(-1, 1)
    return _split_numpy_arrays(x, y, seed)


def _knn_accuracy(embeddings: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(embeddings, labels)
    preds = knn.predict(embeddings)
    return float((preds == labels).mean())


def _cluster_nmi(embeddings: np.ndarray, labels: np.ndarray) -> float:
    n_clusters = int(np.unique(labels).shape[0])
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    clusters = km.fit_predict(embeddings)
    return float(normalized_mutual_info_score(labels, clusters))


def build_task(
    task_name: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    pin_memory: Optional[bool] = None,
) -> Task:
    name = task_name.lower()

    if name in ["prediction", "regression", "sequence"]:
        train_x, train_y, val_x, val_y, test_x, test_y = _load_california_housing(seed, data_dir)
        train_x, val_x, test_x = _standardize_from_train(train_x, val_x, test_x)
        train_ds = ArrayDataset(train_x, train_y)
        val_ds = ArrayDataset(val_x, val_y)
        test_ds = ArrayDataset(test_x, test_y)

        def metrics_fn(model, task, device):
            model.eval()
            xs = []
            ys = []
            with torch.no_grad():
                for x, y in task.val_loader:
                    xs.append(x)
                    ys.append(y)
            x_all = torch.cat(xs, dim=0).to(device)
            y_all = torch.cat(ys, dim=0).to(device)
            preds = model(x_all)
            idx = torch.randperm(x_all.size(0), device=device)[:1000]
            a = preds[idx]
            b = preds[idx.flip(0)]
            ya = y_all[idx]
            yb = y_all[idx.flip(0)]
            acc = float(((a > b) == (ya > yb)).float().mean().item())
            return {"pairwise_acc": acc}

        return Task(
            name=name,
            train_loader=_make_loaders(train_ds, batch_size, num_workers, pin_memory=pin_memory),
            val_loader=_make_loaders(val_ds, batch_size, num_workers, shuffle=False, pin_memory=pin_memory),
            test_loader=_make_loaders(test_ds, batch_size, num_workers, shuffle=False, pin_memory=pin_memory),
            in_dim=int(train_x.shape[1]),
            out_dim=1,
            task_type="regression",
            loss_fn=F.mse_loss,
            metrics_fn=None,
            extra={},
        )

    if name in ["classification", "edge"]:
        train_x, train_y, val_x, val_y, test_x, test_y = _load_covtype(seed, data_dir)
        train_x, val_x, test_x = _standardize_from_train(train_x, val_x, test_x)
        train_ds = ArrayDataset(train_x, train_y)
        val_ds = ArrayDataset(val_x, val_y)
        test_ds = ArrayDataset(test_x, test_y)

        task_type = "classification"
        out_dim = 7
        in_dim = int(train_x.shape[1])
        loss_fn = F.cross_entropy

        return Task(
            name=name,
            train_loader=_make_loaders(train_ds, batch_size, num_workers, pin_memory=pin_memory),
            val_loader=_make_loaders(val_ds, batch_size, num_workers, shuffle=False, pin_memory=pin_memory),
            test_loader=_make_loaders(test_ds, batch_size, num_workers, shuffle=False, pin_memory=pin_memory),
            in_dim=in_dim,
            out_dim=out_dim,
            task_type=task_type,
            loss_fn=loss_fn,
            metrics_fn=None,
            extra={"max_width": 32} if name == "edge" else {},
        )

    if name in ["autoencoding", "generation", "denoising"]:
        train_x, train_y, val_x, val_y, test_x, test_y = _load_covtype(seed, data_dir)
        train_x, val_x, test_x = _standardize_from_train(train_x, val_x, test_x)
        if name == "generation":
            train_ds = PairArrayDataset(np.random.randn(*train_x.shape).astype(np.float32), train_x)
            val_ds = PairArrayDataset(np.random.randn(*val_x.shape).astype(np.float32), val_x)
            test_ds = PairArrayDataset(np.random.randn(*test_x.shape).astype(np.float32), test_x)
            in_dim = int(train_x.shape[1])
        elif name == "denoising":
            noise_std = 0.25
            train_ds = PairArrayDataset(train_x + np.random.randn(*train_x.shape).astype(np.float32) * noise_std, train_x)
            val_ds = PairArrayDataset(val_x + np.random.randn(*val_x.shape).astype(np.float32) * noise_std, val_x)
            test_ds = PairArrayDataset(test_x + np.random.randn(*test_x.shape).astype(np.float32) * noise_std, test_x)
            in_dim = int(train_x.shape[1])
        else:
            train_ds = PairArrayDataset(train_x, train_x)
            val_ds = PairArrayDataset(val_x, val_x)
            test_ds = PairArrayDataset(test_x, test_x)
            in_dim = int(train_x.shape[1])

        return Task(
            name=name,
            train_loader=_make_loaders(train_ds, batch_size, num_workers, pin_memory=pin_memory),
            val_loader=_make_loaders(val_ds, batch_size, num_workers, shuffle=False, pin_memory=pin_memory),
            test_loader=_make_loaders(test_ds, batch_size, num_workers, shuffle=False, pin_memory=pin_memory),
            in_dim=in_dim,
            out_dim=in_dim,
            task_type="reconstruction",
            loss_fn=F.mse_loss,
            metrics_fn=None,
            extra={},
        )

    if name == "anomaly":
        train_x, train_y, val_x, val_y, test_x, test_y = _load_covtype(seed, data_dir)
        train_x, val_x, test_x = _standardize_from_train(train_x, val_x, test_x)
        normal_class = 0
        train_mask = train_y == normal_class
        val_mask = val_y == normal_class
        test_norm_mask = test_y == normal_class
        test_anom_mask = test_y != normal_class
        train_ds = PairArrayDataset(train_x[train_mask], train_x[train_mask])
        val_ds = PairArrayDataset(val_x[val_mask], val_x[val_mask])
        test_norm = OneClassAnomalyDataset(test_x[test_norm_mask], test_x[test_norm_mask], anomaly_label=0)
        test_anom = OneClassAnomalyDataset(test_x[test_anom_mask], test_x[test_anom_mask], anomaly_label=1)
        test_ds = ConcatDataset([test_norm, test_anom])

        def metrics_fn(model, task, device):
            model.eval()
            scores = []
            labels = []
            with torch.no_grad():
                for batch in task.test_loader:
                    x, y, lab = batch
                    x = x.to(device)
                    recon = model(x)
                    err = ((recon - y.to(device)) ** 2).mean(dim=1)
                    scores.append(err.cpu())
                    labels.append(torch.as_tensor(lab))
            if not scores:
                return {}
            scores_np = torch.cat(scores).numpy()
            labels_np = torch.cat(labels).numpy()
            from sklearn.metrics import roc_auc_score
            return {"auroc": float(roc_auc_score(labels_np, scores_np))}

        return Task(
            name=name,
            train_loader=_make_loaders(train_ds, batch_size, num_workers, pin_memory=pin_memory),
            val_loader=_make_loaders(val_ds, batch_size, num_workers, shuffle=False, pin_memory=pin_memory),
            test_loader=_make_loaders(test_ds, batch_size, num_workers, shuffle=False, pin_memory=pin_memory),
            in_dim=int(train_x.shape[1]),
            out_dim=int(train_x.shape[1]),
            task_type="reconstruction",
            loss_fn=F.mse_loss,
            metrics_fn=metrics_fn,
            extra={},
        )

    if name == "simulation":
        train_x, train_y, val_x, val_y, test_x, test_y = _load_california_housing(seed, data_dir)
        train_x, val_x, test_x = _standardize_from_train(train_x, val_x, test_x)
        in_x_train = train_x
        in_x_val = val_x
        in_x_test = test_x
        out_y_train = (train_x[:, 0:1] * train_x[:, 1:2]).astype(np.float32)
        out_y_val = (val_x[:, 0:1] * val_x[:, 1:2]).astype(np.float32)
        out_y_test = (test_x[:, 0:1] * test_x[:, 1:2]).astype(np.float32)

        train_ds = ArrayDataset(in_x_train, out_y_train)
        val_ds = ArrayDataset(in_x_val, out_y_val)
        test_ds = ArrayDataset(in_x_test, out_y_test)
        return Task(
            name=name,
            train_loader=_make_loaders(train_ds, batch_size, num_workers, pin_memory=pin_memory),
            val_loader=_make_loaders(val_ds, batch_size, num_workers, shuffle=False, pin_memory=pin_memory),
            test_loader=_make_loaders(test_ds, batch_size, num_workers, shuffle=False, pin_memory=pin_memory),
            in_dim=int(in_x_train.shape[1]),
            out_dim=int(np.asarray(out_y_train).shape[1]) if np.asarray(out_y_train).ndim > 1 else 1,
            task_type="regression",
            loss_fn=F.mse_loss,
            metrics_fn=None,
            extra={},
        )

    raise ValueError(f"Unknown task: {task_name}")


def task_names() -> List[str]:
    return [
        "classification",
        "autoencoding",
        "generation",
        "denoising",
        "anomaly",
        "simulation",
        "prediction",
    ]
