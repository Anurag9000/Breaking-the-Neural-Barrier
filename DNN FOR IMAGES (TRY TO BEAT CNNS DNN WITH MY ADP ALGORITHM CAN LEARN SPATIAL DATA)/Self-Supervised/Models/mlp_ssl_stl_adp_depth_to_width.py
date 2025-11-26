from pathlib import Path
import importlib.util
import torch.nn as nn

BASE_PATH = Path(__file__).with_name("mlp_ssl_stl_adp_width_to_depth.py").resolve()
_spec = importlib.util.spec_from_file_location("adp_impl", BASE_PATH)
adp_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adp_impl)


class ADPConfig(adp_impl.ADPConfig):  # type: ignore
    pass


class ADPMLPSSL(adp_impl.MLPSSL):  # type: ignore
    pass


def main():
    import argparse
    import subprocess, sys
    p = argparse.ArgumentParser(description="ADP MLP SSL depth_to_width")
    p.add_argument("--hidden", type=int, nargs="+", default=[1024, 512])
    p.add_argument("--rep-dim", type=int, default=256)
    p.add_argument("--proj-dim", type=int, default=128)
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
    cmd = [sys.executable, str(BASE_PATH), "--hidden", *map(str, args.hidden),
           "--rep-dim", str(args.rep_dim), "--proj-dim", str(args.proj_dim),
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
