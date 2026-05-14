import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset, random_split
from torchvision import datasets, transforms

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.neighbors import KNeighborsClassifier


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

    if name in ["prediction", "regression"]:
        g = torch.Generator().manual_seed(int(seed))
        x = torch.randn(20000, 20, generator=g)
        w = torch.randn(20, 1, generator=g)
        y = x @ w + 0.1 * torch.randn(20000, 1, generator=g)
        base = TensorDataset(x, y)
        train_ds, val_ds, test_ds = _split_dataset(base, seed)
        return Task(
            name=name,
            train_loader=_make_loaders(train_ds, batch_size, num_workers),
            val_loader=_make_loaders(val_ds, batch_size, num_workers, shuffle=False),
            test_loader=_make_loaders(test_ds, batch_size, num_workers, shuffle=False),
            in_dim=20,
            out_dim=1,
            task_type="regression",
            loss_fn=F.mse_loss,
            metrics_fn=None,
            extra={},
        )

    if name == "classification":
        train_base = _make_mnist(data_dir, True)
        test_base = _make_mnist(data_dir, False)
        train_ds, val_ds, _ = _split_dataset(train_base, seed)
        return Task(
            name=name,
            train_loader=_make_loaders(train_ds, batch_size, num_workers),
            val_loader=_make_loaders(val_ds, batch_size, num_workers, shuffle=False),
            test_loader=_make_loaders(test_base, batch_size, num_workers, shuffle=False),
            in_dim=28 * 28,
            out_dim=10,
            task_type="classification",
            loss_fn=F.cross_entropy,
            metrics_fn=None,
            extra={},
        )

    if name == "representation":
        train_base = _make_mnist(data_dir, True)
        test_base = _make_mnist(data_dir, False)
        train_ds, val_ds, _ = _split_dataset(train_base, seed)

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
            return {"knn_acc": _knn_accuracy(feats_np, labels_np)}

        return Task(
            name=name,
            train_loader=_make_loaders(train_ds, batch_size, num_workers),
            val_loader=_make_loaders(val_ds, batch_size, num_workers, shuffle=False),
            test_loader=_make_loaders(test_base, batch_size, num_workers, shuffle=False),
            in_dim=28 * 28,
            out_dim=10,
            task_type="classification",
            loss_fn=F.cross_entropy,
            metrics_fn=metrics_fn,
            extra={},
        )

    if name == "autoencoding":
        train_base = _make_mnist(data_dir, True)
        test_base = _make_mnist(data_dir, False)
        train_ds, val_ds, _ = _split_dataset(train_base, seed)
        return Task(
            name=name,
            train_loader=_make_loaders(MNISTFlatWithLabel(train_ds), batch_size, num_workers),
            val_loader=_make_loaders(MNISTFlatWithLabel(val_ds), batch_size, num_workers, shuffle=False),
            test_loader=_make_loaders(MNISTFlatWithLabel(test_base), batch_size, num_workers, shuffle=False),
            in_dim=28 * 28,
            out_dim=28 * 28,
            task_type="reconstruction",
            loss_fn=F.mse_loss,
            metrics_fn=None,
            extra={},
        )

    if name == "generation":
        train_base = _make_mnist(data_dir, True)
        test_base = _make_mnist(data_dir, False)
        train_ds, val_ds, _ = _split_dataset(train_base, seed)
        return Task(
            name=name,
            train_loader=_make_loaders(NoiseToImageDataset(train_ds, noise_dim=64, seed=seed), batch_size, num_workers),
            val_loader=_make_loaders(NoiseToImageDataset(val_ds, noise_dim=64, seed=seed + 1), batch_size, num_workers, shuffle=False),
            test_loader=_make_loaders(NoiseToImageDataset(test_base, noise_dim=64, seed=seed + 2), batch_size, num_workers, shuffle=False),
            in_dim=64,
            out_dim=28 * 28,
            task_type="reconstruction",
            loss_fn=F.mse_loss,
            metrics_fn=None,
            extra={},
        )

    if name == "denoising":
        train_base = _make_mnist(data_dir, True)
        test_base = _make_mnist(data_dir, False)
        train_ds, val_ds, _ = _split_dataset(train_base, seed)
        return Task(
            name=name,
            train_loader=_make_loaders(NoisyMNIST(train_ds, noise_std=0.5, seed=seed), batch_size, num_workers),
            val_loader=_make_loaders(NoisyMNIST(val_ds, noise_std=0.5, seed=seed + 1), batch_size, num_workers, shuffle=False),
            test_loader=_make_loaders(NoisyMNIST(test_base, noise_std=0.5, seed=seed + 2), batch_size, num_workers, shuffle=False),
            in_dim=28 * 28,
            out_dim=28 * 28,
            task_type="reconstruction",
            loss_fn=F.mse_loss,
            metrics_fn=None,
            extra={},
        )

    if name == "anomaly":
        train_base = _make_mnist(data_dir, True)
        test_base = _make_mnist(data_dir, False)
        train_indices = [i for i, (_, y) in enumerate(train_base) if y < 5]
        test_norm_indices = [i for i, (_, y) in enumerate(test_base) if y < 5]
        test_anom_indices = [i for i, (_, y) in enumerate(test_base) if y >= 5]

        train_subset = torch.utils.data.Subset(train_base, train_indices)
        train_ds, val_ds, _ = _split_dataset(train_subset, seed)

        test_norm = AnomalySubset(test_base, test_norm_indices, label=0)
        test_anom = AnomalySubset(test_base, test_anom_indices, label=1)
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
            train_loader=_make_loaders(MNISTFlatPair(train_ds), batch_size, num_workers),
            val_loader=_make_loaders(MNISTFlatPair(val_ds), batch_size, num_workers, shuffle=False),
            test_loader=_make_loaders(test_ds, batch_size, num_workers, shuffle=False),
            in_dim=28 * 28,
            out_dim=28 * 28,
            task_type="reconstruction",
            loss_fn=F.mse_loss,
            metrics_fn=metrics_fn,
            extra={},
        )

    if name == "sequence":
        base = SineSequenceDataset(n_samples=20000, window=20, seed=seed)
        train_ds, val_ds, test_ds = _split_dataset(base, seed)
        return Task(
            name=name,
            train_loader=_make_loaders(train_ds, batch_size, num_workers),
            val_loader=_make_loaders(val_ds, batch_size, num_workers, shuffle=False),
            test_loader=_make_loaders(test_ds, batch_size, num_workers, shuffle=False),
            in_dim=20,
            out_dim=1,
            task_type="regression",
            loss_fn=F.mse_loss,
            metrics_fn=None,
            extra={},
        )

    if name == "inverse":
        base = LinearInverseDataset(n_samples=20000, in_dim=16, out_dim=8, seed=seed)
        train_ds, val_ds, test_ds = _split_dataset(base, seed)
        return Task(
            name=name,
            train_loader=_make_loaders(train_ds, batch_size, num_workers),
            val_loader=_make_loaders(val_ds, batch_size, num_workers, shuffle=False),
            test_loader=_make_loaders(test_ds, batch_size, num_workers, shuffle=False),
            in_dim=8,
            out_dim=16,
            task_type="regression",
            loss_fn=F.mse_loss,
            metrics_fn=None,
            extra={},
        )

    if name == "control":
        base = LQRDataset(n_samples=20000, state_dim=8, action_dim=4, seed=seed)
        train_ds, val_ds, test_ds = _split_dataset(base, seed)
        return Task(
            name=name,
            train_loader=_make_loaders(train_ds, batch_size, num_workers),
            val_loader=_make_loaders(val_ds, batch_size, num_workers, shuffle=False),
            test_loader=_make_loaders(test_ds, batch_size, num_workers, shuffle=False),
            in_dim=8,
            out_dim=4,
            task_type="regression",
            loss_fn=F.mse_loss,
            metrics_fn=None,
            extra={},
        )

    if name == "clustering":
        train_base = _make_mnist(data_dir, True)
        test_base = _make_mnist(data_dir, False)
        train_ds, val_ds, _ = _split_dataset(train_base, seed)

        def metrics_fn(model, task, device):
            model.eval()
            feats = []
            labels = []
            with torch.no_grad():
                for batch in task.val_loader:
                    if isinstance(batch, (list, tuple)) and len(batch) == 3:
                        x, _, y = batch
                    else:
                        x, y = batch
                    x = x.to(device)
                    _, emb = model(x, return_embedding=True)
                    feats.append(emb.cpu().numpy())
                    labels.append(torch.as_tensor(y).numpy())
            feats_np = np.concatenate(feats, axis=0)
            labels_np = np.concatenate(labels, axis=0)
            return {"nmi": _kmeans_nmi(feats_np, labels_np)}

        return Task(
            name=name,
            train_loader=_make_loaders(MNISTFlatWithLabel(train_ds), batch_size, num_workers),
            val_loader=_make_loaders(MNISTFlatWithLabel(val_ds), batch_size, num_workers, shuffle=False),
            test_loader=_make_loaders(MNISTFlatWithLabel(test_base), batch_size, num_workers, shuffle=False),
            in_dim=28 * 28,
            out_dim=28 * 28,
            task_type="reconstruction",
            loss_fn=F.mse_loss,
            metrics_fn=metrics_fn,
            extra={},
        )

    if name == "compression":
        train_base = _make_mnist(data_dir, True)
        test_base = _make_mnist(data_dir, False)
        train_ds, val_ds, _ = _split_dataset(train_base, seed)
        return Task(
            name=name,
            train_loader=_make_loaders(MNISTFlatPair(train_ds), batch_size, num_workers),
            val_loader=_make_loaders(MNISTFlatPair(val_ds), batch_size, num_workers, shuffle=False),
            test_loader=_make_loaders(MNISTFlatPair(test_base), batch_size, num_workers, shuffle=False),
            in_dim=28 * 28,
            out_dim=28 * 28,
            task_type="reconstruction",
            loss_fn=F.mse_loss,
            metrics_fn=None,
            extra={"compression_ratio": 0.0},
        )

    if name == "ranking":
        base = RankingDataset(n_samples=20000, in_dim=20, seed=seed)
        train_ds, val_ds, test_ds = _split_dataset(base, seed)

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
            idx = torch.randperm(x_all.size(0))[:1000]
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
            in_dim=20,
            out_dim=1,
            task_type="regression",
            loss_fn=F.mse_loss,
            metrics_fn=metrics_fn,
            extra={},
        )

    if name == "multimodal":
        train_base = _make_mnist(data_dir, True)
        test_base = _make_mnist(data_dir, False)
        train_ds, val_ds, _ = _split_dataset(train_base, seed)
        return Task(
            name=name,
            train_loader=_make_loaders(ParityMNIST(train_ds), batch_size, num_workers),
            val_loader=_make_loaders(ParityMNIST(val_ds), batch_size, num_workers, shuffle=False),
            test_loader=_make_loaders(ParityMNIST(test_base), batch_size, num_workers, shuffle=False),
            in_dim=28 * 28 + 1,
            out_dim=10,
            task_type="classification",
            loss_fn=F.cross_entropy,
            metrics_fn=None,
            extra={},
        )

    if name == "selfsupervised":
        train_base = _make_mnist(data_dir, True)
        test_base = _make_mnist(data_dir, False)
        train_ds, val_ds, _ = _split_dataset(train_base, seed)
        return Task(
            name=name,
            train_loader=_make_loaders(RotationMNIST(train_ds), batch_size, num_workers),
            val_loader=_make_loaders(RotationMNIST(val_ds), batch_size, num_workers, shuffle=False),
            test_loader=_make_loaders(RotationMNIST(test_base), batch_size, num_workers, shuffle=False),
            in_dim=28 * 28,
            out_dim=4,
            task_type="classification",
            loss_fn=F.cross_entropy,
            metrics_fn=None,
            extra={},
        )

    if name == "simulation":
        base = LinearDynamicsDataset(n_samples=20000, state_dim=8, action_dim=4, seed=seed)
        train_ds, val_ds, test_ds = _split_dataset(base, seed)
        return Task(
            name=name,
            train_loader=_make_loaders(train_ds, batch_size, num_workers),
            val_loader=_make_loaders(val_ds, batch_size, num_workers, shuffle=False),
            test_loader=_make_loaders(test_ds, batch_size, num_workers, shuffle=False),
            in_dim=12,
            out_dim=8,
            task_type="regression",
            loss_fn=F.mse_loss,
            metrics_fn=None,
            extra={},
        )

    if name == "edge":
        train_base = _make_mnist(data_dir, True)
        test_base = _make_mnist(data_dir, False)
        train_ds, val_ds, _ = _split_dataset(train_base, seed)
        return Task(
            name=name,
            train_loader=_make_loaders(train_ds, batch_size, num_workers),
            val_loader=_make_loaders(val_ds, batch_size, num_workers, shuffle=False),
            test_loader=_make_loaders(test_base, batch_size, num_workers, shuffle=False),
            in_dim=28 * 28,
            out_dim=10,
            task_type="classification",
            loss_fn=F.cross_entropy,
            metrics_fn=None,
            extra={"max_width": 32},
        )

    if name == "misc":
        base = ResidualDataset(n_samples=20000, in_dim=20, seed=seed)
        train_ds, val_ds, test_ds = _split_dataset(base, seed)
        return Task(
            name=name,
            train_loader=_make_loaders(train_ds, batch_size, num_workers),
            val_loader=_make_loaders(val_ds, batch_size, num_workers, shuffle=False),
            test_loader=_make_loaders(test_ds, batch_size, num_workers, shuffle=False),
            in_dim=20,
            out_dim=1,
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
