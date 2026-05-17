from __future__ import annotations

from MLPS.tabular.shared.dae_dnn._staged_phase_entry import run_single_phase


def main() -> None:
    run_single_phase("ae_depth_only")


if __name__ == "__main__":
    main()
