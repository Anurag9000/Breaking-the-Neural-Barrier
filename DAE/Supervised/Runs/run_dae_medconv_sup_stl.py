"""Alias runner for medical-image Conv DAE + classifier.

This uses the Gaussian Conv DAE supervised runner but lives under a
separate name so it lines up with the medical-image entry in the plan.
"""

from .run_dae_gaussian_conv_sup_stl import main as _base_main


def main() -> None:
    _base_main()


if __name__ == "__main__":
    main()

