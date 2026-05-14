"""
Repo-wide benchmark guardrails.

When `BBNB_STRICT_BENCHMARKS=1` is set, legacy toy datasets and toy demo
entrypoints are blocked so that new runs cannot silently fall back to
MNIST/CIFAR/STL-style benchmarks or synthetic smoke tests. This is
intentionally opt-in to avoid breaking old scripts until they are migrated to
the shared benchmark loaders.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _enabled() -> bool:
    return os.environ.get("BBNB_STRICT_BENCHMARKS", "").strip().lower() in {"1", "true", "yes", "on"}


if _enabled():
    def _die(message: str) -> None:
        try:
            sys.stderr.write(message + "\n")
        except Exception:
            pass
        os._exit(1)

    _argv0 = Path(sys.argv[0]).name.lower() if sys.argv else ""
    if any(token in _argv0 for token in ("toy", "dummy", "mock")):
        _die(
            f"{sys.argv[0]} is blocked by BBNB_STRICT_BENCHMARKS=1. "
            "Use a real benchmark entrypoint instead of a toy/demo script."
        )
    _script_path = Path(sys.argv[0]) if sys.argv and sys.argv[0] else None
    if _script_path is not None and _script_path.exists() and _script_path.is_file():
        try:
            _script_text = _script_path.read_text(encoding="utf-8", errors="ignore").lower()
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
                f"{_script_path} contains synthetic/demo data code and is blocked by "
                "BBNB_STRICT_BENCHMARKS=1."
            )

    try:
        import torchvision.datasets as _tvds
    except Exception:
        _tvds = None

    if _tvds is not None:
        def _blocked(name: str):
            def _ctor(*args, **kwargs):
                raise RuntimeError(
                    f"{name} is blocked by BBNB_STRICT_BENCHMARKS=1. "
                    "Use a publishable benchmark loader instead of a toy dataset."
                )

            return _ctor

        for _ds_name in ["MNIST", "FashionMNIST", "KMNIST", "CIFAR10", "CIFAR100", "STL10", "SVHN"]:
            if hasattr(_tvds, _ds_name):
                setattr(_tvds, _ds_name, _blocked(_ds_name))
