from __future__ import annotations

import argparse
import os
from pathlib import Path

from sklearn.datasets import fetch_california_housing, fetch_covtype, fetch_openml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prefetch the sklearn datasets used by DAE/DNN tabular tasks.")
    p.add_argument(
        "--data-home",
        type=str,
        default=None,
        help="Optional sklearn data cache directory. Defaults to sklearn's normal cache location.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Prefetch all tabular benchmark datasets used by the active DAE/DNN tasks.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.data_home:
        os.environ["SCIKIT_LEARN_DATA"] = str(Path(args.data_home).expanduser())

    datasets = [
        ("Covertype", lambda: fetch_covtype(download_if_missing=True)),
        ("California Housing", lambda: fetch_california_housing()),
        ("YearPredictionMSD", lambda: fetch_openml(name="YearPredictionMSD", version=1, as_frame=False)),
    ]

    if not args.all:
        datasets = [("Covertype", lambda: fetch_covtype(download_if_missing=True))]

    for name, loader in datasets:
        print(f"Prefetching {name}...")
        loader()
        print(f"Cached {name}.")

    print("Done.")


if __name__ == "__main__":
    main()
