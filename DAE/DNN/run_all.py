import argparse
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from DAE.DNN.tasks import task_names


def main() -> None:
    p = argparse.ArgumentParser(description="Run STL + 6 ADP modes for all DNN tasks")
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--results-dir", type=str, default="DAE/DNN/results")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--max-epochs", type=int, default=100000000)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--ex-k", type=int, default=1)
    p.add_argument("--trials-width", type=int, default=0)
    p.add_argument("--trials-depth", type=int, default=0)
    p.add_argument("--max-width", type=int, default=4096)
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--max-neurons", type=int, default=10000000)
    p.add_argument("--hidden", type=int, nargs="+", default=[50, 50])
    p.add_argument("--tasks", type=str, nargs="+", default=["all"], help="Task names or 'all'")
    args = p.parse_args()

    tasks = task_names() if "all" in [t.lower() for t in args.tasks] else args.tasks
    adp_modes = ["width_only", "depth_only", "width_to_depth", "depth_to_width", "alt_width", "alt_depth"]

    for task in tasks:
        for mode in ["stl"] + adp_modes:
            cmd = [
                sys.executable,
                str(Path("DAE/DNN/run_task.py")),
                "--task",
                task,
                "--mode",
                "adp" if mode != "stl" else "stl",
                "--adp-mode",
                mode if mode != "stl" else "width_only",
                "--data-dir",
                args.data_dir,
                "--results-dir",
                args.results_dir,
                "--batch-size",
                str(args.batch_size),
                "--max-epochs",
                str(args.max_epochs),
                "--patience",
                str(args.patience),
                "--ex-k",
                str(args.ex_k),
                "--trials-width",
                str(args.trials_width),
                "--trials-depth",
                str(args.trials_depth),
                "--max-width",
                str(args.max_width),
                "--max-depth",
                str(args.max_depth),
                "--max-neurons",
                str(args.max_neurons),
            ]
            cmd += ["--hidden"] + [str(h) for h in args.hidden]
            print("Running:", " ".join(cmd))
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
