"""Alias runner for multi-head DAE classifier.

Reuses the group-sparse MLP DAE supervised runner.
"""

from .run_dae_groupsparse_mlp_sup_stl import main as _base_main


def main() -> None:
    _base_main()


if __name__ == "__main__":
    main()

