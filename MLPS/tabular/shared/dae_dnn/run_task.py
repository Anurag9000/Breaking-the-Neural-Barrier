import argparse
import json
import datetime as _dt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch

from utils.adp_logging import ContinuousLogger
from utils.adp_plot import plot_best_loss_per_neurons_from_csv, plot_val_loss_from_csv

from MLPS.tabular.shared.dae_dnn.mlp import MLP
from MLPS.tabular.shared.dae_dnn.runtime_tuning import bootstrap_runtime
from MLPS.tabular.shared.dae_dnn.tasks import build_task, refresh_task_loaders, stl_batch_size_for_task
from MLPS.tabular.shared.dae_dnn.adp_search import ADPConfig, adp_search, train_with_early_stopping
from MLPS.tabular.shared.dae_dnn.train_utils import AdaptiveBatchController
from MLPS.tabular.shared.dae_dnn.train_utils import eval_epoch


def main() -> None:
    bootstrap_runtime("run_task")

    def format_hidden(hidden):
        return str([int(w) for w in hidden])

    p = argparse.ArgumentParser(description="DNN STL/ADP task runner (plain MLP)")
    p.add_argument("--task", type=str, required=True)
    p.add_argument("--mode", type=str, default="adp", choices=["stl", "adp"])
    p.add_argument(
        "--adp-mode",
        type=str,
        default="width_to_depth",
        choices=["alt_width", "alt_depth", "width_to_depth", "depth_to_width"],
    )
    p.add_argument("--hidden", type=int, nargs="+", default=[50, 50])
    p.add_argument("--batch-size", type=int, default=0, help="Batch size override. 0 (default) defers to per-task target-batches computation.")
    p.add_argument("--run-root", type=str, default=None, help="Optional fixed output root for resumable runs.")
    p.add_argument("--max-epochs", type=int, default=100000000)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--trials-width", type=int, default=10)
    p.add_argument("--trials-depth", type=int, default=2)
    p.add_argument("--ex-k", type=int, default=1)
    p.add_argument("--max-width", type=int, default=4096)
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--max-neurons", type=int, default=10000000)
    p.add_argument("--width-stage-margin-patience", type=int, default=10)
    p.add_argument("--width-stage-min-improve-pct", type=float, default=1.0)
    p.add_argument("--metrics-interval", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--results-dir", type=str, default="MLPS/tabular/shared/dae_dnn/results")
    args = p.parse_args()

    torch.manual_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Build a probe task at batch_size=1 to discover train-set size, then
    # apply the canonical target-based default when batch_size==0.
    probe_task = build_task(
        args.task,
        args.data_dir,
        1,
        args.num_workers,
        args.seed,
        pin_memory=False,
    )
    train_size = len(probe_task.train_loader.dataset)
    initial_batch_size = stl_batch_size_for_task(args.task, train_size, override=int(args.batch_size))

    task = build_task(
        args.task,
        args.data_dir,
        initial_batch_size,
        args.num_workers,
        args.seed,
        pin_memory=False,
    )

    max_width = args.max_width
    if "max_width" in task.extra:
        max_width = min(int(task.extra["max_width"]), int(args.max_width))

    hidden = list(args.hidden)
    if args.mode == "adp" and not hidden:
        hidden = [1]
    model = MLP(in_dim=task.in_dim, hidden_widths=hidden, out_dim=task.out_dim)

    if args.run_root:
        results_dir = Path(args.run_root)
        run_name = results_dir.name
    else:
        run_name = (
            f"{task.name}_{args.mode}_{args.adp_mode}_d{len(hidden)}"
            f"_w{max(hidden) if hidden else 0}_exk{args.ex_k}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        results_dir = Path(args.results_dir) / run_name
    batch_state_path = results_dir / "_batch_size_state.json"

    task_state_path = results_dir / "task_state.json"
    existing_task_state = {}
    if task_state_path.exists():
        try:
            existing_task_state = json.loads(task_state_path.read_text(encoding="utf-8"))
        except Exception:
            existing_task_state = {}
    if bool(existing_task_state.get("completed", False)) and not bool(existing_task_state.get("failed", False)):
        print(f"[RESUME] Task already completed at {task_state_path}; skipping.", flush=True)
        return

    logger = ContinuousLogger(results_dir, f"dnn_{task.name}", args.adp_mode, resume=results_dir.exists())

    logger.log_console(
        f"Task={task.name} mode={args.mode} adp_mode={args.adp_mode} hidden={format_hidden(hidden)} in_dim={task.in_dim} out_dim={task.out_dim}"
    )
    logger.log_console(
        f"ADP: ex_k={args.ex_k} trials_width={args.trials_width} trials_depth={args.trials_depth} max_width={max_width} max_depth={args.max_depth} max_neurons={args.max_neurons}"
    )
    logger.log_console(
        f"Train: batch_size={args.batch_size} lr=1e-3 weight_decay=1e-4 es_patience={args.patience} max_epochs={args.max_epochs}"
    )
    logger.log_console(f"Device: {device}")

    batch_controller = AdaptiveBatchController(
        initial_batch_size,
        threshold_gb=5.5,
        poll_interval_sec=30.0,
        shrink_factor=1.0,
        state_path=batch_state_path,
        restore_state=True,
    )
    batch_controller.start()

    cfg = ADPConfig(
        adp_mode=args.adp_mode,
        delta=1e-4,
        patience=args.patience,
        trials_width=args.trials_width,
        trials_depth=args.trials_depth,
        ex_k=args.ex_k,
        max_width=max_width,
        max_depth=args.max_depth,
        max_neurons=args.max_neurons,
        width_stage_margin_patience=args.width_stage_margin_patience,
        width_stage_min_improve_pct=args.width_stage_min_improve_pct,
        max_epochs=args.max_epochs,
        metrics_interval=args.metrics_interval,
    )

    task_state = {
        "task": task.name,
        "mode": args.mode,
        "adp_mode": args.adp_mode,
        "hidden": [int(w) for w in hidden],
        "batch_size": int(args.batch_size),
        "run_root": str(results_dir),
        "completed": False,
        "failed": False,
    }

    try:
        task_state_path.parent.mkdir(parents=True, exist_ok=True)
        task_state_path.write_text(json.dumps({**task_state, "status": "running"}, indent=2, sort_keys=True), encoding="utf-8")
        refresh_task_loaders(task, batch_controller.current_batch_size)
        if args.mode == "stl":
            best_val, best_state, _ = train_with_early_stopping(
                model.to(device), task, cfg, device, logger, batch_controller=batch_controller
            )
            model.load_state_dict(best_state)
            logger.log_console(f"[STL] best_val_loss={best_val:.6f} hidden={format_hidden(model.hidden_widths)}")
        else:
            best_val, model = adp_search(
                model.to(device),
                task,
                cfg,
                device,
                logger,
                batch_controller=batch_controller,
                results_dir=results_dir,
            )
            logger.log_console(f"[ADP] best_val_loss={best_val:.6f} hidden={format_hidden(model.hidden_widths)}")

        val_loss, val_acc, throughput = eval_epoch(model, task.val_loader, task.loss_fn, device, task.task_type, measure_throughput=(task.name == "edge"))
        logger.log_console(f"[VAL] loss={val_loss:.6f} acc={val_acc if val_acc is not None else 'na'}")
        if throughput is not None:
            logger.log_console(f"[VAL] throughput={throughput:.2f} samples/sec")

        if task.metrics_fn is not None:
            metrics = task.metrics_fn(model, task, device)
            if metrics:
                logger.log_console(f"[METRICS] {metrics}")
                logger.log_epoch_stats({"epoch": 0, **metrics})

        task_state.update(
            {
                "completed": True,
                "best_val": float(best_val),
                "best_hidden": [int(w) for w in model.hidden_widths],
                "val_loss": float(val_loss),
                "val_acc": None if val_acc is None else float(val_acc),
                "throughput": None if throughput is None else float(throughput),
            }
        )
        task_state_path.write_text(json.dumps(task_state, indent=2, sort_keys=True), encoding="utf-8")
        (results_dir / "task_summary.json").write_text(json.dumps(task_state, indent=2, sort_keys=True), encoding="utf-8")

        plot_val_loss_from_csv(logger.csv_file, results_dir / "val_loss_vs_step.png", title=f"{run_name} - val_loss")
        plot_best_loss_per_neurons_from_csv(
            logger.csv_file, results_dir / "loss_vs_neurons_best.png", title=f"{run_name} - best val_loss per neurons"
        )
    except BaseException as exc:
        task_state.update({"completed": False, "failed": True, "error": repr(exc)})
        task_state_path.write_text(json.dumps(task_state, indent=2, sort_keys=True), encoding="utf-8")
        raise
    finally:
        batch_controller.stop()
        logger.close()


if __name__ == "__main__":
    try:
        import os as _os, sys as _sys
        if _os.name == "posix" and _sys.platform.startswith("linux"):
            import ctypes as _ctypes
            _ctypes.CDLL("libc.so.6", use_errno=True).mlockall(3)
        elif _os.name == "nt":
            import ctypes as _ctypes
            _ctypes.windll.kernel32.SetProcessWorkingSetSize(_ctypes.windll.kernel32.GetCurrentProcess(), -1, -1)
    except Exception:
        pass
    main()
