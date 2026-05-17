from __future__ import annotations

from MLPS.tabular.shared.dae_dnn._staged_phase_entry import run_single_phase


def main() -> None:
    run_single_phase("ae_width_to_depth")


if __name__ == "__main__":
    main()
