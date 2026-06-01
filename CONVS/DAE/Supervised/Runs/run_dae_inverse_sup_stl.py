"""Alias runner for inverse-problem DAE (deblurring/deconvolution) + classifier.

This uses the Gaussian Conv DAE supervised runner.
"""

from .run_dae_gaussian_conv_sup_stl import main as _base_main


def main() -> None:
    _base_main()


if __name__ == "__main__":
    main()

