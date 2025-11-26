from pathlib import Path
import importlib.util
import torch.nn as nn

BASE_PATH = Path(__file__).with_name("nlp_ssl_common_adp_width_to_depth.py").resolve()
_spec = importlib.util.spec_from_file_location("adp_impl", BASE_PATH)
adp_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adp_impl)

ADPConfig = adp_impl.ADPConfig  # type: ignore
MLPTextSSL = adp_impl.MLPTextSSL  # type: ignore
adp_search = adp_impl.adp_search  # type: ignore


def main():
    import subprocess, sys
    subprocess.call([sys.executable, str(BASE_PATH), "--adp-mode", "depth_to_width"])


if __name__ == "__main__":
    main()
