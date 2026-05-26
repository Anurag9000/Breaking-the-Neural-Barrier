import argparse
import json
import datetime as _dt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import torch

from utils.adp_logging import ContinuousLogger
from utils.adp_plot import plot_best_loss_per_neurons_from_csv, plot_val_loss_from_csv

from DAE.DNN.mlp import MLP
from DAE.DNN.tasks import build_task, refresh_task_loaders
from DAE.DNN.adp_search import ADPConfig, adp_search, train_with_early_stopping
from DAE.DNN.train_utils import AdaptiveBatchController
from DAE.DNN.train_utils import eval_epoch


def main() -> None:
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
    p.add_argument("--batch-size", type=int, default=32768)
    p.add_argument("--max-epochs", type=int, default=100000000)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--trials-width", type=int, default=10)
    p.add_argument("--trials-depth", type=int, default=5)
    p.add_argument("--ex-k", type=int, default=1)
    p.add_argument("--max-width", type=int, default=4096)
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--max-neurons", type=int, default=10000000)
    p.add_argument("--width-stage-margin-patience", type=int, default=5)
    p.add_argument("--width-stage-min-improve-pct", type=float, default=1.0)
    p.add_argument("--metrics-interval", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--results-dir", type=str, default="DAE/DNN/results")
    args = p.parse_args()

    torch.manual_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_state_path = Path(args.results_dir) / "_batch_size_state.json"
    initial_batch_size = int(args.batch_size)
    if batch_state_path.exists():
        try:
            payload = json.loads(batch_state_path.read_text(encoding="utf-8"))
            initial_batch_size = min(initial_batch_size, int(payload.get("batch_size", initial_batch_size)))
        except Exception:
            pass

    task = build_task(args.task, args.data_dir, initial_batch_size, args.num_workers, args.seed)

    max_width = args.max_width
    if "max_width" in task.extra:
        max_width = min(int(task.extra["max_width"]), int(args.max_width))

    model = MLP(in_dim=task.in_dim, hidden_widths=args.hidden, out_dim=task.out_dim)

    run_name = (
        f"{task.name}_{args.mode}_{args.adp_mode}_d{len(args.hidden)}"
        f"_w{max(args.hidden) if args.hidden else 0}_exk{args.ex_k}_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    results_dir = Path(args.results_dir) / run_name
    logger = ContinuousLogger(results_dir, f"dnn_{task.name}", args.adp_mode)

    logger.log_console(
        f"Task={task.name} mode={args.mode} adp_mode={args.adp_mode} hidden={format_hidden(args.hidden)} in_dim={task.in_dim} out_dim={task.out_dim}"
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
        shrink_factor=0.75,
        state_path=batch_state_path,
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

    try:
        refresh_task_loaders(task, batch_controller.current_batch_size)
        if args.mode == "stl":
            best_val, best_state, _ = train_with_early_stopping(
                model.to(device), task, cfg, device, logger, batch_controller=batch_controller
            )
            model.load_state_dict(best_state)
            logger.log_console(f"[STL] best_val_loss={best_val:.6f} hidden={format_hidden(model.hidden_widths)}")
        else:
            best_val, model = adp_search(
                model.to(device), task, cfg, device, logger, batch_controller=batch_controller
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

        plot_val_loss_from_csv(logger.csv_file, results_dir / "val_loss_vs_step.png", title=f"{run_name} - val_loss")
        plot_best_loss_per_neurons_from_csv(
            logger.csv_file, results_dir / "loss_vs_neurons_best.png", title=f"{run_name} - best val_loss per neurons"
        )
    finally:
        batch_controller.stop()
        logger.close()


if __name__ == "__main__":
    main()
