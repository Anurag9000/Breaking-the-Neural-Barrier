from pathlib import Path
import importlib.util

BASE_PATH = Path(__file__).with_name("nlp_ae_common_adp_width_to_depth.py").resolve()
_spec = importlib.util.spec_from_file_location("adp_impl", BASE_PATH)
adp_impl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adp_impl)


ADPConfig = adp_impl.ADPConfig  # type: ignore
TextAE = adp_impl.TextAE  # type: ignore
adp_search = adp_impl.adp_search  # type: ignore


def main():
    import argparse
    import subprocess, sys
    p = argparse.ArgumentParser(description="ADP NLP AE depth_to_width")
    p.add_argument("--hidden", type=int, nargs="+", default=[512, 256])
    p.add_argument("--out-dim", type=int, default=128)
    p.add_argument("--vocab-size", type=int, default=30000)
    p.add_argument("--emb-dim", type=int, default=128)
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--trials-width", type=int, default=2)
    p.add_argument("--trials-depth", type=int, default=2)
    p.add_argument("--ex-k", type=int, default=64)
    p.add_argument("--max-width", type=int, default=4096)
    p.add_argument("--max-depth", type=int, default=12)
    p.add_argument("--max-neurons", type=int, default=5_000_000)
    p.add_argument("--max-epochs", type=int, default=30)
    args = p.parse_args()
    cmd = [sys.executable, str(BASE_PATH),
           "--hidden", *map(str, args.hidden),
           "--out-dim", str(args.out_dim),
           "--vocab-size", str(args.vocab_size),
           "--emb-dim", str(args.emb_dim),
           "--adp-mode", "depth_to_width",
           "--delta", str(args.delta),
           "--patience", str(args.patience),
           "--trials-width", str(args.trials_width),
           "--trials-depth", str(args.trials_depth),
           "--ex-k", str(args.ex_k),
           "--max-width", str(args.max_width),
           "--max-depth", str(args.max_depth),
           "--max-neurons", str(args.max_neurons),
           "--max-epochs", str(args.max_epochs)]
    subprocess.call(cmd)


if __name__ == "__main__":
    main()
