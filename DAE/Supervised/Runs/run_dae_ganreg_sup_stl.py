"""Alias runner for DAE-regularized GAN discriminator classifier.

Reuses the DAE-regularized ResNet supervised runner.
"""

from .run_dae_resnet_reg_sup_stl import main as _base_main


def main() -> None:
    _base_main()


if __name__ == "__main__":
    main()

