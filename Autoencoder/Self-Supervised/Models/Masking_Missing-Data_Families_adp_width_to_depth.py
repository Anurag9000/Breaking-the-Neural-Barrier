import subprocess
import sys
from pathlib import Path

# Delegate to the unified ADP AE SSL core, forcing masked/missing-data algo and the requested ADP mode.
BASE_PATH = Path(__file__).with_name("adp_ae_ssl_core.py").resolve()


def main():
    # Always select the masking/missing-data family; allow downstream args to override dataset, etc.
    cmd = [sys.executable, str(BASE_PATH), "--algo", "masked", "--adp-mode", "width_to_depth"]
    cmd.extend(sys.argv[1:])
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
