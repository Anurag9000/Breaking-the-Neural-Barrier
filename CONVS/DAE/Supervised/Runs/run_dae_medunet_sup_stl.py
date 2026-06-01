"""Alias runner for medical-image U-Net DAE + classifier.

This reuses the U-Net Conv supervised runner.
"""

from .run_dae_unet_conv_sup_stl import main as _base_main


def main() -> None:
    _base_main()


if __name__ == "__main__":
    main()

