import subprocess, sys
from pathlib import Path

BASE_PATH = Path(__file__).with_name("adp_ae_ssl_core.py").resolve()

def main():
    cmd = [sys.executable, str(BASE_PATH), "--algo", "masked", "--adp-mode", "depth_only"]
    cmd.extend(sys.argv[1:])
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
