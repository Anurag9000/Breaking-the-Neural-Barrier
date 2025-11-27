from pathlib import Path
import importlib.util
import subprocess, sys

BASE_PATH = Path(__file__).with_name("ae_lowrank_stl_py_low_rank_nuclear_norm_regularized_ae_adp_width_to_depth.py").resolve()
_spec = importlib.util.spec_from_file_location("adp_impl", BASE_PATH)
adp_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adp_impl)

for _k, _v in list(adp_impl.__dict__.items()):
    if _k.startswith("__"):
        continue
    globals()[_k] = _v


def main():
    cmd = [sys.executable, str(BASE_PATH), "--adp-mode", "alt_depth"]
    cmd.extend(sys.argv[1:])
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
