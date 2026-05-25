from __future__ import annotations

import sys
from typing import List

from MLPS.tabular.shared.dae_dnn.run_goliath_staged_width import main as staged_main

SUPPORTED_STAGED_PHASES = {"ae_alt_width", "ae_width_to_depth"}


def _force_single_phase(argv: List[str], phase_name: str) -> List[str]:
    rewritten: List[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--phases":
            rewritten.extend(["--phases", phase_name])
            i += 1
            while i < len(argv) and not argv[i].startswith("-"):
                i += 1
            continue
        rewritten.append(arg)
        i += 1
    if "--phases" not in rewritten:
        rewritten.extend(["--phases", phase_name])
    return rewritten


def run_single_phase(phase_name: str) -> None:
    if phase_name not in SUPPORTED_STAGED_PHASES:
        raise SystemExit(
            f"Unsupported staged ADP phase '{phase_name}'. Supported phases are: "
            f"{', '.join(sorted(SUPPORTED_STAGED_PHASES))}."
        )
    sys.argv = [sys.argv[0], *_force_single_phase(sys.argv[1:], phase_name)]
    staged_main()
