from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from DAE.DNN.tasks import build_task
from utils.adp_logging import ContinuousLogger

import run_goliath as rg


DEFAULT_TASKS = ["representation", "autoencoding", "generation"]

# Quick but still broad: tiny, medium, wide, shallow, and deep.
QUICK_ARCHITECTURES = [
    [1],
    [4],
    [16],
    [64],
    [128],
    [1, 1],
    [4, 4],
    [16, 16],
    [64, 64],
    [128, 128],
    [4, 4, 4, 4],
    [16, 16, 16, 16],
    [64, 64, 64, 64],
    [16, 16, 16, 16, 16, 16],
    [64, 64, 64, 64, 64, 64],
    [16, 16, 16, 16, 16, 16, 16, 16, 16, 16],
    [64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
]


def parse_csv_ints(text: str) -> List[int]:
    return [int(part.strip()) for part in str(text).split(",") if part.strip()]


def parse_architectures(values: Sequence[str]) -> List[List[int]]:
    architectures: List[List[int]] = []
    for item in values:
        arch = parse_csv_ints(item)
        if arch:
            architectures.append([max(1, int(v)) for v in arch])
    return architectures


def dedupe_architectures(architectures: Iterable[Sequence[int]]) -> List[List[int]]:
    seen = set()
    out: List[List[int]] = []
    for arch in architectures:
        key = tuple(int(v) for v in arch)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(list(key))
    return out


def build_architectures(args) -> List[List[int]]:
    if args.architecture:
        return dedupe_architectures(parse_architectures(args.architecture))
    if args.widths and args.depths:
        widths = parse_csv_ints(args.widths)
        depths = parse_csv_ints(args.depths)
        return dedupe_architectures([[int(width)] * int(depth) for depth in depths for width in widths])
    return [list(arch) for arch in QUICK_ARCHITECTURES]


def phase_name_for_architecture(architecture: Sequence[int]) -> str:
    depth = len(architecture)
    width = max(int(v) for v in architecture)
    return f"stl_ablation_d{depth:02d}_w{width:03d}_{'_'.join(str(int(v)) for v in architecture)}"


def load_source_task_summary(source_run_root: Path, task_name: str) -> Dict[str, Any]:
    path = source_run_root / task_name / "task_summary.json"
    return rg.load_json_if_exists(path) or {}


def comparison_row(
    *,
    task_name: str,
    ablation_phase: str,
    ablation_architecture: Sequence[int],
    ablation_best_val: float,
    ref_phase: str,
    ref_kind: str,
    ref_architecture: Optional[Sequence[int]],
    ref_best_val: float,
) -> Dict[str, Any]:
    winner = "ablation_stl" if float(ablation_best_val) <= float(ref_best_val) else ref_kind
    return {
        "task": task_name,
        "ablation_phase": ablation_phase,
        "ablation_architecture": rg.format_architecture_for_report(ablation_architecture),
        "ablation_best_val": float(ablation_best_val),
        "reference_kind": ref_kind,
        "reference_phase": ref_phase,
        "reference_architecture": rg.format_architecture_for_report(ref_architecture),
        "reference_best_val": float(ref_best_val),
        "winner": winner,
        "winner_value": min(float(ablation_best_val), float(ref_best_val)),
    }


def make_cfg(args, tasks: List[str], run_root: Path) -> rg.RunConfig:
    return rg.RunConfig(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        run_root=str(run_root),
        tasks=tasks,
        phases=[],
        batch_size=int(args.batch_size),
        num_workers=int(args.num_workers),
        seed=int(args.seed),
        stl_width=int(args.stl_width),
        stl_depth=int(args.stl_depth),
        alt_start_width=1,
        alt_start_depth=1,
        patience=int(args.patience),
        width_expansion_patience=10,
        depth_expansion_patience=2,
        delta=float(args.delta),
        max_epochs=int(args.max_epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_clip=float(args.grad_clip),
        max_width=int(args.max_width),
        max_depth=int(args.max_depth),
        max_neurons=int(args.max_neurons),
        width_stage_margin_patience=int(args.width_stage_margin_patience),
        width_stage_min_improve_pct=float(args.width_stage_min_improve_pct),
        use_bn=bool(args.use_bn),
        demo=False,
    )


def run_task_ablation(
    *,
    task_name: str,
    task_root: Path,
    cfg: rg.RunConfig,
    source_run_root: Path,
    architectures: Sequence[Sequence[int]],
    device,
    log: ContinuousLogger,
    batch_controller,
) -> Dict[str, Any]:
    task_batch_size = rg.batch_size_for_task(task_name, cfg.batch_size)
    task = build_task(task_name, cfg.data_dir, task_batch_size, cfg.num_workers, cfg.seed)
    source_summary = load_source_task_summary(source_run_root, task_name)

    ablation_runs: List[Dict[str, Any]] = []
    comparisons: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []

    source_adp_runs = list(source_summary.get("adp_runs", []))
    source_paired_stl_runs = list(source_summary.get("paired_stl_runs", []))
    source_refs = [
        ("adp", entry.get("phase"), entry.get("architecture"), float(entry.get("best_val", float("inf"))))
        for entry in source_adp_runs
    ] + [
        ("paired_stl", entry.get("phase"), entry.get("architecture"), float(entry.get("best_val", float("inf"))))
        for entry in source_paired_stl_runs
    ]

    best_ablation: Optional[Dict[str, Any]] = None

    for architecture in architectures:
        phase_name = phase_name_for_architecture(architecture)
        log.log_console(f"[ABLATION:{task_name}] STL phase start: {phase_name} architecture={rg.format_architecture_for_report(architecture)}")
        summary = rg.run_stl_phase(
            task,
            task_root,
            cfg,
            device,
            list(architecture),
            phase_name=phase_name,
            source_phase=None,
            batch_controller=batch_controller,
        )
        ablation_runs.append(summary)

        if best_ablation is None or float(summary.get("best_val", float("inf"))) < float(best_ablation.get("best_val", float("inf"))):
            best_ablation = summary

        rows.append(
            {
                "task": task_name,
                "row_type": "ablation_stl",
                "phase": phase_name,
                "architecture": rg.format_architecture_for_report(architecture),
                "best_val": float(summary.get("best_val", float("inf"))),
                "best_epoch": int(summary.get("best_epoch", 0)),
                "final_epoch": int(summary.get("final_epoch", summary.get("best_epoch", 0))),
                "test_loss": (summary.get("test_metrics") or {}).get("test_loss"),
                "test_acc": (summary.get("test_metrics") or {}).get("test_acc"),
            }
        )

        for ref_kind, ref_phase, ref_arch, ref_best_val in source_refs:
            comparisons.append(
                comparison_row(
                    task_name=task_name,
                    ablation_phase=phase_name,
                    ablation_architecture=architecture,
                    ablation_best_val=float(summary.get("best_val", float("inf"))),
                    ref_phase=str(ref_phase),
                    ref_kind=ref_kind,
                    ref_architecture=ref_arch,
                    ref_best_val=float(ref_best_val),
                )
            )

    rg.write_csv(
        task_root / "ablation_summary.csv",
        rows,
        fieldnames=["task", "row_type", "phase", "architecture", "best_val", "best_epoch", "final_epoch", "test_loss", "test_acc"],
    )
    rg.write_json(
        task_root / "ablation_summary.json",
        {
            "task": task_name,
            "source_task_summary": str(source_run_root / task_name / "task_summary.json"),
            "source_adp_runs": source_adp_runs,
            "source_paired_stl_runs": source_paired_stl_runs,
            "ablation_stl_runs": ablation_runs,
            "comparisons": comparisons,
            "best_ablation": best_ablation,
        },
    )

    return {
        "task": task_name,
        "source_adp_runs": source_adp_runs,
        "source_paired_stl_runs": source_paired_stl_runs,
        "ablation_stl_runs": ablation_runs,
        "comparisons": comparisons,
        "best_ablation": best_ablation,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick STL architecture ablation for selected tabular DAE/DNN tasks.")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--results-dir", default="MLPS/tabular/shared/dae_dnn/results")
    p.add_argument("--run-root", default=None)
    p.add_argument("--source-run-root", default="MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current")
    p.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    p.add_argument("--batch-size", type=int, default=32768)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--delta", type=float, default=1e-4)
    p.add_argument("--max-epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-width", type=int, default=512)
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--max-neurons", type=int, default=10_000_000)
    p.add_argument("--width-stage-margin-patience", type=int, default=10)
    p.add_argument("--width-stage-min-improve-pct", type=float, default=1.0)
    p.add_argument("--stl-width", type=int, default=128)
    p.add_argument("--stl-depth", type=int, default=2)
    p.add_argument("--use-bn", action="store_true", default=True)
    p.add_argument("--no-bn", dest="use_bn", action="store_false")
    p.add_argument("--widths", default="")
    p.add_argument("--depths", default="")
    p.add_argument("--architecture", action="append", default=[], help="Explicit hidden widths, e.g. --architecture 64,64,64")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tasks = [str(t).lower() for t in args.tasks]
    architectures = build_architectures(args)
    if not architectures:
        raise SystemExit("No architectures requested.")

    source_run_root = Path(args.source_run_root)
    run_root = Path(args.run_root) if args.run_root else Path(args.results_dir) / f"stl_ablation_{rg.now_stamp()}"
    run_root.mkdir(parents=True, exist_ok=True)

    cfg = make_cfg(args, tasks, run_root)
    rg.seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_controller = None

    logger = ContinuousLogger(run_root, "stl_ablation", "stl_ablation")
    logger.log_console(f"Run root: {run_root}")
    logger.log_console(f"Tasks: {tasks}")
    logger.log_console(f"Architectures: {[rg.format_architecture_for_report(a) for a in architectures]}")
    logger.log_console(f"Source run root: {source_run_root}")
    logger.log_console(f"Device: {device}")
    logger.log_console(f"Git commit: {rg.git_commit()}")

    task_reports: List[Dict[str, Any]] = []
    comparison_rows: List[Dict[str, Any]] = []
    for task_name in tasks:
        task_root = run_root / task_name
        task_root.mkdir(parents=True, exist_ok=True)
        report = run_task_ablation(
            task_name=task_name,
            task_root=task_root,
            cfg=cfg,
            source_run_root=source_run_root,
            architectures=architectures,
            device=device,
            log=logger,
            batch_controller=batch_controller,
        )
        task_reports.append(report)
        comparison_rows.extend(report["comparisons"])

    if comparison_rows:
        rg.write_csv(
            run_root / "comparison_summary.csv",
            comparison_rows,
            fieldnames=[
                "task",
                "ablation_phase",
                "ablation_architecture",
                "ablation_best_val",
                "reference_kind",
                "reference_phase",
                "reference_architecture",
                "reference_best_val",
                "winner",
                "winner_value",
            ],
        )

    rg.write_json(
        run_root / "comparison_summary.json",
        {
            "tasks": tasks,
            "architectures": architectures,
            "source_run_root": str(source_run_root),
            "reports": task_reports,
        },
    )
    logger.close()


if __name__ == "__main__":
    main()
