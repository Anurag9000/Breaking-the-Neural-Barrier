"""Alias runner for semi-supervised graph DAE classifier (plan item 14).

This reuses the supervised graph node-feature DAE runner.
"""

from .run_dae_graph_nodefeat_sup_stl import main as _base_main


def main() -> None:
    _base_main()


if __name__ == "__main__":
    main()

