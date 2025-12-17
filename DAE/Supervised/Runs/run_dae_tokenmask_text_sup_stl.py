"""
Supervised text DAE classifier (BERT-style), aliasing the semi-supervised runner.

This reuses the CLI and training loop from run_dae_tokenmask_text_semisup_stl.py
but is exposed under a separate module path so it can correspond to the
\"BERT-style DAE pretrain + classifier fine-tune\" entry in the plan.
"""

from .run_dae_tokenmask_text_semisup_stl import main as _base_main


def main() -> None:
    _base_main()


if __name__ == "__main__":
    main()

