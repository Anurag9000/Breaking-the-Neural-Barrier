"""
Repo-wide benchmark guardrails.

Legacy toy datasets and toy demo entrypoints are blocked so that new runs
cannot silently fall back to MNIST/CIFAR/STL-style benchmarks or synthetic
smoke tests.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _die(message: str) -> None:
    try:
        sys.stderr.write(message + "\n")
    except Exception:
        pass
    os._exit(1)


_repo_root = Path(__file__).resolve().parent
_script_path = Path(sys.argv[0]) if sys.argv and sys.argv[0] else None
if _script_path is not None and _script_path.exists() and _script_path.is_file():
    try:
        _resolved_script = _script_path.resolve()
    except Exception:
        _resolved_script = _script_path
    try:
        _inside_repo = _resolved_script.is_relative_to(_repo_root)
    except Exception:
        _inside_repo = str(_repo_root) in str(_resolved_script)
    if _inside_repo:
        _argv0 = _resolved_script.name.lower()
        if any(token in _argv0 for token in ("toy", "dummy", "mock")):
            _die(
                f"{_resolved_script} is blocked. Use a real benchmark entrypoint instead of a toy/demo script."
            )
        try:
            _script_text = _resolved_script.read_text(encoding="utf-8", errors="ignore").lower()
        except Exception:
            _script_text = ""
        if any(
            marker in _script_text
            for marker in (
                "synthetic toy",
                "toy dataset",
                "dummy dataset",
                "dummy data",
                "random images",
                "random noise",
                "synthesize tsv pairs",
                "synthesiz",
                "mock data",
                "fake data",
            )
        ):
            _die(
                f"{_resolved_script} contains synthetic/demo data code and is blocked."
            )

        try:
            import torchvision.datasets as _tvds
        except Exception:
            _tvds = None

        if _tvds is not None:
            def _resolve_split_root(root, split: str) -> Path:
                base = Path(root)
                candidates = []
                split = split.lower()
                if split in {"train", "training"}:
                    candidates.extend([base / "train", base / "training"])
                elif split in {"val", "valid", "validation"}:
                    candidates.extend([base / "val", base / "valid", base / "validation"])
                elif split in {"test", "testing"}:
                    candidates.extend([base / "test", base / "testing"])
                elif split in {"train+unlabeled", "train_unlabeled", "trainunlabeled"}:
                    candidates.extend([base / "train", base / "training"])
                for candidate in candidates:
                    if candidate.exists():
                        return candidate
                return base

            def _imagefolder_from_split(name: str):
                def _ctor(root, *args, **kwargs):
                    transform = kwargs.pop("transform", None)
                    target_transform = kwargs.pop("target_transform", None)
                    if kwargs.get("train", True):
                        split = "train"
                    else:
                        split = "test"
                    if "split" in kwargs:
                        split = str(kwargs.pop("split"))
                    split_root = _resolve_split_root(root, split)
                    from torchvision.datasets import ImageFolder

                    if not split_root.exists():
                        _die(
                            f"{name} redirect expected a real folder-backed benchmark under {split_root}, "
                            "but it does not exist."
                        )
                    return ImageFolder(
                        root=str(split_root),
                        transform=transform,
                        target_transform=target_transform,
                    )

                return _ctor

            def _blocked(name: str):
                def _ctor(*args, **kwargs):
                    _die(
                        f"{name} is blocked. Use a publishable benchmark loader instead of a toy dataset."
                    )

                return _ctor

            for _ds_name in ["CIFAR10", "CIFAR100", "STL10", "SVHN"]:
                if hasattr(_tvds, _ds_name):
                    setattr(_tvds, _ds_name, _imagefolder_from_split(_ds_name))

            for _ds_name in ["MNIST", "FashionMNIST", "KMNIST"]:
                if hasattr(_tvds, _ds_name):
                    setattr(_tvds, _ds_name, _blocked(_ds_name))
