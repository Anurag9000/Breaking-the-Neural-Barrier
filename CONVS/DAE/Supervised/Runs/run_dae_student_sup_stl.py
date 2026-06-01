"""Alias runner for student–teacher DAE classifier.

Reuses the Gaussian Conv DAE supervised runner.
"""

from .run_dae_gaussian_conv_sup_stl import main as _base_main


def main() -> None:
    _base_main()


if __name__ == "__main__":
    main()

