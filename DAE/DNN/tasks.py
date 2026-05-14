import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset, random_split
from torchvision import datasets, transforms

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


def _make_mnist(base_dir: str, train: bool, img_size: Tuple[int, int] = (28, 28)):
    tf = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    return datasets.MNIST(root=base_dir, train=train, download=True, transform=tf)


def _make_loaders(ds: Dataset, batch_size: int, num_workers: int, shuffle: bool = True):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def clone_loader(loader: DataLoader, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        loader.dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(loader.num_workers),
        pin_memory=bool(getattr(loader, "pin_memory", False)),
        drop_last=bool(getattr(loader, "drop_last", False)),
        collate_fn=getattr(loader, "collate_fn", None),
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


class FeaturePermutationDataset(Dataset):
    def __init__(self, x: np.ndarray, seed: int = 0, n_perms: int = 4):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.n_perms = int(n_perms)
        rng = np.random.default_rng(int(seed))
        base = np.arange(self.x.shape[1])
        self.perms = [torch.as_tensor(rng.permutation(base), dtype=torch.long) for _ in range(self.n_perms)]

    def __len__(self):
        return int(self.x.size(0))

    def __getitem__(self, idx):
        label = int(idx % self.n_perms)
        return self.x[idx][self.perms[label]], torch.tensor(label, dtype=torch.long)


def _load_covtype(seed: int):
    data = fetch_covtype(download_if_missing=True)
    x = np.asarray(data.data, dtype=np.float32)
    y = np.asarray(data.target, dtype=np.int64) - 1
    return _split_numpy_arrays(x, y, seed)


def _load_year_prediction(seed: int):
    try:
        data = fetch_openml(name="YearPredictionMSD", version=1, as_frame=False)
        x = np.asarray(data.data, dtype=np.float32)
        y = np.asarray(data.target, dtype=np.float32).reshape(-1, 1)
    except Exception:
        data = fetch_california_housing()
        x = np.asarray(data.data, dtype=np.float32)
        y = np.asarray(data.target, dtype=np.float32).reshape(-1, 1)
    return _split_numpy_arrays(x, y, seed)


def _load_california_housing(seed: int):
    data = fetch_california_housing()
    x = np.asarray(data.data, dtype=np.float32)
    y = np.asarray(data.target, dtype=np.float32).reshape(-1, 1)
    return _split_numpy_arrays(x, y, seed)


def _knn_accuracy(embeddings: np.ndarray, labels: np.ndarray, k: int = 5) -> float:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(embeddings, labels)
    preds = knn.predict(embeddings)
    return float((preds == labels).mean())


def _kmeans_nmi(embeddings: np.ndarray, labels: np.ndarray, n_clusters: int = 10) -> float:
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    clusters = km.fit_predict(embeddings)
    return float(normalized_mutual_info_score(labels, clusters))


class MNISTFlatPair(Dataset):
    def __init__(self, base_ds: Dataset):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, _ = self.base_ds[idx]
        x_flat = x.view(-1)
        return x_flat, x_flat


class MNISTFlatWithLabel(Dataset):
    def __init__(self, base_ds: Dataset):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, label = self.base_ds[idx]
        x_flat = x.view(-1)
        return x_flat, x_flat, label


class NoisyMNIST(Dataset):
    def __init__(self, base_ds: Dataset, noise_std: float = 0.5, seed: int = 0):
        self.base_ds = base_ds
        self.noise_std = float(noise_std)
        self.rng = torch.Generator().manual_seed(int(seed))

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, _ = self.base_ds[idx]
        noise = torch.randn(x.shape, generator=self.rng) * self.noise_std
        noisy = torch.clamp(x + noise, 0.0, 1.0)
        return noisy.view(-1), x.view(-1)


class NoiseToImageDataset(Dataset):
    def __init__(self, base_ds: Dataset, noise_dim: int = 64, seed: int = 0):
        self.base_ds = base_ds
        g = torch.Generator().manual_seed(int(seed))
        self.noise = torch.randn(len(base_ds), noise_dim, generator=g)

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        img, _ = self.base_ds[idx]
        return self.noise[idx], img.view(-1)


class RotationMNIST(Dataset):
    def __init__(self, base_ds: Dataset):
        self.base_ds = base_ds
        self.angles = [0, 90, 180, 270]

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        img, _ = self.base_ds[idx]
        label = idx % 4
        angle = self.angles[label]
        if angle == 90:
            img = torch.rot90(img, 1, [1, 2])
        elif angle == 180:
            img = torch.rot90(img, 2, [1, 2])
        elif angle == 270:
            img = torch.rot90(img, 3, [1, 2])
        return img, label


class ParityMNIST(Dataset):
    def __init__(self, base_ds: Dataset):
        self.base_ds = base_ds

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        img, label = self.base_ds[idx]
        parity = float(label % 2)
        x = torch.cat([img.view(-1), torch.tensor([parity])], dim=0)
        return x, label


class AnomalySubset(Dataset):
    def __init__(self, base_ds: Dataset, indices: List[int], label: int):
        self.base_ds = base_ds
        self.indices = indices
        self.label = int(label)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, _ = self.base_ds[self.indices[idx]]
        x_flat = img.view(-1)
        return x_flat, x_flat, self.label


class SineSequenceDataset(Dataset):
    def __init__(self, n_samples: int = 20000, window: int = 20, seed: int = 0):
        g = torch.Generator().manual_seed(int(seed))
        self.window = int(window)
        t = torch.linspace(0, 200 * math.pi, steps=n_samples + window + 1)
        noise = torch.randn(len(t), generator=g) * 0.1
        signal = torch.sin(t) + noise
        self.data = signal
        self.n = n_samples

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window]
        y = self.data[idx + self.window]
        return x, y.unsqueeze(0)


class LinearInverseDataset(Dataset):
    def __init__(self, n_samples: int = 20000, in_dim: int = 16, out_dim: int = 8, seed: int = 0):
        g = torch.Generator().manual_seed(int(seed))
        self.A = torch.randn(out_dim, in_dim, generator=g)
        x = torch.randn(n_samples, in_dim, generator=g)
        y = x @ self.A.t() + 0.05 * torch.randn(n_samples, out_dim, generator=g)
        self.inputs = y
        self.targets = x

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class LinearDynamicsDataset(Dataset):
    def __init__(self, n_samples: int = 20000, state_dim: int = 8, action_dim: int = 4, seed: int = 0):
        g = torch.Generator().manual_seed(int(seed))
        self.A = torch.randn(state_dim, state_dim, generator=g) * 0.3
        self.B = torch.randn(state_dim, action_dim, generator=g) * 0.3
        x = torch.randn(n_samples, state_dim, generator=g)
        u = torch.randn(n_samples, action_dim, generator=g)
        x_next = x @ self.A.t() + u @ self.B.t()
        self.inputs = torch.cat([x, u], dim=1)
        self.targets = x_next

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class RankingDataset(Dataset):
    def __init__(self, n_samples: int = 20000, in_dim: int = 20, seed: int = 0):
        g = torch.Generator().manual_seed(int(seed))
        w = torch.randn(in_dim, generator=g)
        x = torch.randn(n_samples, in_dim, generator=g)
        scores = x @ w + 0.1 * torch.randn(n_samples, generator=g)
        self.inputs = x
        self.targets = scores.unsqueeze(1)

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class ResidualDataset(Dataset):
    def __init__(self, n_samples: int = 20000, in_dim: int = 20, seed: int = 0):
        g = torch.Generator().manual_seed(int(seed))
        w = torch.randn(in_dim, generator=g)
        bias = torch.randn(1, generator=g) * 0.5
        x = torch.randn(n_samples, in_dim, generator=g)
        y = x @ w + bias + 0.1 * torch.randn(n_samples, generator=g)
        baseline = x @ w
        residual = (y - baseline).unsqueeze(1)
        self.inputs = x
        self.targets = residual

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


class LQRDataset(Dataset):
    def __init__(self, n_samples: int = 20000, state_dim: int = 8, action_dim: int = 4, seed: int = 0):
        g = torch.Generator().manual_seed(int(seed))
        K = torch.randn(action_dim, state_dim, generator=g) * 0.5
        x = torch.randn(n_samples, state_dim, generator=g)
        u = -x @ K.t()
        self.inputs = x
        self.targets = u

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def build_task(task_name: str, data_dir: str, batch_size: int, num_workers: int, seed: int) -> Task:
    name = task_name.lower()

    if name in ["prediction", "regression", "sequence", "ranking"]:
        train_x, train_y, val_x, val_y, test_x, test_y = _load_year_prediction(seed)
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
            train_loader=_make_loaders(train_ds, batch_size, num_workers),
            val_loader=_make_loaders(val_ds, batch_size, num_workers, shuffle=False),
            test_loader=_make_loaders(test_ds, batch_size, num_workers, shuffle=False),
            in_dim=int(train_x.shape[1]),
            out_dim=1,
            task_type="regression",
            loss_fn=F.mse_loss,
            metrics_fn=metrics_fn if name == "ranking" else None,
            extra={},
        )

    if name in ["classification", "representation", "clustering", "selfsupervised", "edge"]:
        train_x, train_y, val_x, val_y, test_x, test_y = _load_covtype(seed)
        train_x, val_x, test_x = _standardize_from_train(train_x, val_x, test_x)

        if name == "selfsupervised":
            train_ds = FeaturePermutationDataset(train_x, seed=seed)
            val_ds = FeaturePermutationDataset(val_x, seed=seed + 1)
            test_ds = FeaturePermutationDataset(test_x, seed=seed + 2)
            task_type = "classification"
            out_dim = 4
            in_dim = int(train_x.shape[1])
            loss_fn = F.cross_entropy
            metrics_fn = None
        else:
            train_ds = ArrayDataset(train_x, train_y)
            val_ds = ArrayDataset(val_x, val_y)
            test_ds = ArrayDataset(test_x, test_y)

            metrics_fn = None
            if name in {"representation", "clustering"}:
                def metrics_fn(model, task, device):
                    model.eval()
                    feats = []
                    labels = []
                    with torch.no_grad():
                        for x, y in task.val_loader:
                            x = x.to(device)
                            _, emb = model(x, return_embedding=True)
                            feats.append(emb.cpu().numpy())
                            labels.append(y.numpy())
                    feats_np = np.concatenate(feats, axis=0)
                    labels_np = np.concatenate(labels, axis=0)
                    return {"knn_acc": _knn_accuracy(feats_np, labels_np)} if name == "representation" else {"nmi": _kmeans_nmi(feats_np, labels_np)}

            task_type = "classification"
            out_dim = 7
            in_dim = int(train_x.shape[1])
            loss_fn = F.cross_entropy

        return Task(
            name=name,
            train_loader=_make_loaders(train_ds, batch_size, num_workers),
            val_loader=_make_loaders(val_ds, batch_size, num_workers, shuffle=False),
            test_loader=_make_loaders(test_ds, batch_size, num_workers, shuffle=False),
            in_dim=in_dim,
            out_dim=out_dim,
            task_type=task_type,
            loss_fn=loss_fn,
            metrics_fn=metrics_fn,
            extra={"max_width": 32} if name == "edge" else {},
        )

    if name in ["autoencoding", "generation", "denoising", "compression", "multimodal"]:
        train_x, train_y, val_x, val_y, test_x, test_y = _load_covtype(seed)
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
        elif name == "multimodal":
            parity_train = (train_y % 2).astype(np.float32).reshape(-1, 1)
            parity_val = (val_y % 2).astype(np.float32).reshape(-1, 1)
            parity_test = (test_y % 2).astype(np.float32).reshape(-1, 1)
            train_ds = ArrayDataset(np.concatenate([train_x, parity_train], axis=1), train_y)
            val_ds = ArrayDataset(np.concatenate([val_x, parity_val], axis=1), val_y)
            test_ds = ArrayDataset(np.concatenate([test_x, parity_test], axis=1), test_y)
            in_dim = int(train_x.shape[1] + 1)
        else:
            train_ds = PairArrayDataset(train_x, train_x)
            val_ds = PairArrayDataset(val_x, val_x)
            test_ds = PairArrayDataset(test_x, test_x)
            in_dim = int(train_x.shape[1])

        return Task(
            name=name,
            train_loader=_make_loaders(train_ds, batch_size, num_workers),
            val_loader=_make_loaders(val_ds, batch_size, num_workers, shuffle=False),
            test_loader=_make_loaders(test_ds, batch_size, num_workers, shuffle=False),
            in_dim=in_dim,
            out_dim=in_dim if name != "multimodal" else 7,
            task_type="reconstruction" if name != "multimodal" else "classification",
            loss_fn=F.mse_loss if name != "multimodal" else F.cross_entropy,
            metrics_fn=None,
            extra={"compression_ratio": 0.0} if name == "compression" else {},
        )

    if name == "anomaly":
        train_x, train_y, val_x, val_y, test_x, test_y = _load_covtype(seed)
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
            train_loader=_make_loaders(train_ds, batch_size, num_workers),
            val_loader=_make_loaders(val_ds, batch_size, num_workers, shuffle=False),
            test_loader=_make_loaders(test_ds, batch_size, num_workers, shuffle=False),
            in_dim=int(train_x.shape[1]),
            out_dim=int(train_x.shape[1]),
            task_type="reconstruction",
            loss_fn=F.mse_loss,
            metrics_fn=metrics_fn,
            extra={},
        )

    if name in ["inverse", "control", "simulation", "misc"]:
        train_x, train_y, val_x, val_y, test_x, test_y = _load_california_housing(seed)
        train_x, val_x, test_x = _standardize_from_train(train_x, val_x, test_x)
        if name == "inverse":
            in_x_train = train_x[:, 4:]
            out_y_train = train_x[:, :4]
            in_x_val = val_x[:, 4:]
            out_y_val = val_x[:, :4]
            in_x_test = test_x[:, 4:]
            out_y_test = test_x[:, :4]
        elif name == "control":
            in_x_train = np.concatenate([train_x[:, :4], train_y], axis=1)
            in_x_val = np.concatenate([val_x[:, :4], val_y], axis=1)
            in_x_test = np.concatenate([test_x[:, :4], test_y], axis=1)
            out_y_train = train_x[:, 4:]
            out_y_val = val_x[:, 4:]
            out_y_test = test_x[:, 4:]
        elif name == "simulation":
            in_x_train = train_x
            in_x_val = val_x
            in_x_test = test_x
            out_y_train = (train_x[:, 0:1] * train_x[:, 1:2]).astype(np.float32)
            out_y_val = (val_x[:, 0:1] * val_x[:, 1:2]).astype(np.float32)
            out_y_test = (test_x[:, 0:1] * test_x[:, 1:2]).astype(np.float32)
        else:
            in_x_train = train_x
            in_x_val = val_x
            in_x_test = test_x
            out_y_train = train_y
            out_y_val = val_y
            out_y_test = test_y

        train_ds = ArrayDataset(in_x_train, out_y_train)
        val_ds = ArrayDataset(in_x_val, out_y_val)
        test_ds = ArrayDataset(in_x_test, out_y_test)
        return Task(
            name=name,
            train_loader=_make_loaders(train_ds, batch_size, num_workers),
            val_loader=_make_loaders(val_ds, batch_size, num_workers, shuffle=False),
            test_loader=_make_loaders(test_ds, batch_size, num_workers, shuffle=False),
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
        "prediction",
        "classification",
        "representation",
        "autoencoding",
        "generation",
        "denoising",
        "anomaly",
        "sequence",
        "inverse",
        "control",
        "clustering",
        "compression",
        "ranking",
        "multimodal",
        "selfsupervised",
        "simulation",
        "edge",
        "misc",
    ]
