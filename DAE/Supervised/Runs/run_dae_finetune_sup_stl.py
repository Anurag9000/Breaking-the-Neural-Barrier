"""Alias runner for DAE-pretrained encoder fine-tune baseline.

This reuses the DAE-regularized ResNet supervised runner as a simple
fine-tuning baseline entrypoint.
"""

from .run_dae_resnet_reg_sup_stl import main as _base_main


def main() -> None:
    _base_main()


if __name__ == "__main__":
    main()

