from pathlib import Path
import importlib.util
import torch.nn as nn

BASE_PATH = Path(__file__).with_name("mlp_ae_stl_adp_width_to_depth.py").resolve()
_spec = importlib.util.spec_from_file_location("adp_impl", BASE_PATH)
adp_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adp_impl)


class ADP_MlpAeStl(adp_impl.ADPConfig):  # type: ignore
    pass


class ADPMLPAE(adp_impl.MLPAutoencoder):  # type: ignore
    pass


def main():
    # delegate to width_to_depth script but set default adp_mode to depth_to_width
    import argparse
    import torch
    p = argparse.ArgumentParser(description="ADP MLP Autoencoder depth_to_width")
    p.add_argument("--hidden", type=int, nargs="+", default=[1024, 512])
    p.add_argument("--bottleneck", type=int, default=256)
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--trials-width", type=int, default=2)
    p.add_argument("--trials-depth", type=int, default=2)
    p.add_argument("--ex-k", type=int, default=128)
    p.add_argument("--max-width", type=int, default=4096)
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--max-neurons", type=int, default=10_000_000)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()
    args.adp_mode = "depth_to_width"
    # call underlying main with adjusted args
    import subprocess, sys
    cmd = [sys.executable, str(BASE_PATH), "--hidden", *map(str, args.hidden),
           "--bottleneck", str(args.bottleneck),
           "--adp-mode", "depth_to_width",
           "--delta", str(args.delta),
           "--patience", str(args.patience),
           "--trials-width", str(args.trials_width),
           "--trials-depth", str(args.trials_depth),
           "--ex-k", str(args.ex_k),
           "--max-width", str(args.max_width),
           "--max-depth", str(args.max_depth),
           "--max-neurons", str(args.max_neurons),
           "--max-epochs", str(args.max_epochs),
           "--batch-size", str(args.batch_size)]
    subprocess.call(cmd)


if __name__ == "__main__":
    main()
