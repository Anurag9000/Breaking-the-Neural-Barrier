"""Alias runner for super-resolution DAE + classifier.

Reuses the Gaussian Conv DAE supervised runner; super-resolution behaviour
is controlled by the data pipeline, but width/depth training is identical.
"""

from .run_dae_gaussian_conv_sup_stl import main as _base_main


def main() -> None:
    _base_main()


if __name__ == "__main__":
    main()

