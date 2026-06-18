from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from MLPS.tabular.shared.dae_dnn.tasks import build_task
from MLPS.tabular.shared.dae_dnn.runtime_tuning import bootstrap_runtime
from utils.adp_logging import ContinuousLogger

try:  # pragma: no cover - import shim for direct script execution
    import run_goliath as rg
    import run_stl_ablation_parallel as launcher
    import run_stl_ablation as stl
except ModuleNotFoundError:  # pragma: no cover - import shim for package-style imports
    from MLPS.tabular.shared.dae_dnn import run_goliath as rg
    from MLPS.tabular.shared.dae_dnn import run_stl_ablation_parallel as launcher
    from MLPS.tabular.shared.dae_dnn import run_stl_ablation as stl


@dataclass(frozen=True)
class ProbeCandidate:
    task_name: str
    architecture: Tuple[int, ...]
    parameter_count: int
    depth: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stress-test STL parallelism with the largest models first.")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--results-dir", default="MLPS/tabular/shared/dae_dnn/results")
    p.add_argument("--run-root", default=None)
    p.add_argument("--source-run-root", default="MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current")
    p.add_argument("--tasks", nargs="+", default=list(stl.DEFAULT_TASKS))
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--pin-memory", dest="pin_memory", action="store_true", default=False)
    p.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--delta", type=float, default=1e-4)
    p.add_argument("--probe-epochs", type=int, default=2)
    p.add_argument("--max-width", type=int, default=1024)
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--max-neurons", type=int, default=10_000_000)
    p.add_argument("--width-stage-margin-patience", type=int, default=10)
    p.add_argument("--width-stage-min-improve-pct", type=float, default=1.0)
    p.add_argument("--min-width", type=int, default=1)
    p.add_argument("--width-step", type=int, default=1)
    p.add_argument("--width-count-per-depth", type=int, default=10)
    p.add_argument("--min-depth", type=int, default=1)
    p.add_argument(
        "--param-band",
        nargs=2,
        type=int,
        metavar=("PARAM_EXP_START", "PARAM_EXP_END"),
        default=None,
        help="Parameter-count decade band to probe, e.g. 1 3 for 10^1 through 10^3.",
    )
    p.add_argument("--start-n", type=int, default=2, help="Start probing at this parallelism.")
    p.add_argument("--stl-width", type=int, default=128)
    p.add_argument("--stl-depth", type=int, default=2)
    p.add_argument("--metrics-every", type=int, default=0)
    p.add_argument("--use-bn", action="store_true", default=True)
    p.add_argument("--no-bn", dest="use_bn", action="store_false")
    p.add_argument(
        "--legacy-architecture-grid",
        action="store_true",
        default=False,
        help="Use the old fixed width x depth sweep instead of parameter-matched depth families.",
    )
    return p.parse_args()


def normalize_tasks(tasks: Sequence[str]) -> List[str]:
    return [str(task).lower() for task in tasks]


def resolve_run_root(args: argparse.Namespace, param_band: Optional[Tuple[int, int]]) -> Path:
    if args.run_root:
        return Path(args.run_root)
    band_label = stl.param_band_label(param_band)
    suffix = f"_{band_label}" if band_label else ""
    return Path(args.results_dir) / f"stl_parallelism_probe{suffix}_{rg.now_stamp()}"


def make_cfg(args: argparse.Namespace, tasks: Sequence[str], run_root: Path) -> rg.RunConfig:
    param_band = stl.normalize_param_band(getattr(args, "param_band", None))
    return rg.RunConfig(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        run_root=str(run_root),
        tasks=list(tasks),
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
        max_epochs=int(args.probe_epochs),
        lr=1e-3,
        weight_decay=1e-4,
        grad_clip=1.0,
        max_width=int(args.max_width),
        max_depth=int(args.max_depth),
        max_neurons=int(args.max_neurons),
        width_stage_margin_patience=int(args.width_stage_margin_patience),
        width_stage_min_improve_pct=float(args.width_stage_min_improve_pct),
        use_bn=bool(args.use_bn),
        demo=False,
        metrics_every=int(args.metrics_every),
        min_width=int(args.min_width),
        width_step=int(args.width_step),
        width_count_per_depth=int(args.width_count_per_depth),
        parameter_matched=not bool(getattr(args, "legacy_architecture_grid", False)),
        parameter_band=param_band,
    )


def probe_candidate_sort_key(candidate: ProbeCandidate) -> Tuple[int, int, str, Tuple[int, ...]]:
    return (-int(candidate.parameter_count), int(candidate.depth), str(candidate.task_name), tuple(candidate.architecture))


def probe_trial_sizes(start_n: int, candidate_count: int) -> List[int]:
    start_n = max(1, int(start_n))
    candidate_count = max(0, int(candidate_count))
    if candidate_count == 0:
        return []
    if start_n > candidate_count:
        start_n = candidate_count
    return list(range(start_n, candidate_count + 1))


def select_probe_candidates(candidates: Sequence[ProbeCandidate], n: int) -> List[ProbeCandidate]:
    n = max(0, int(n))
    if n <= 0:
        return []
    return list(candidates[: min(n, len(candidates))])


def build_probe_candidates(args: argparse.Namespace, run_root: Path) -> List[ProbeCandidate]:
    tasks = normalize_tasks(args.tasks)
    cfg = make_cfg(args, tasks, run_root)
    base_architectures = stl.build_architectures(args)
    candidates: List[ProbeCandidate] = []

    for task_name in tasks:
        task = build_task(task_name, cfg.data_dir, 1, cfg.num_workers, cfg.seed, pin_memory=bool(args.pin_memory))
        for architecture in base_architectures:
            family = [list(architecture)]
            if cfg.parameter_matched and len(architecture) == 1:
                family = stl.parameter_matched_architectures(task, int(architecture[0]), cfg)
            for expanded_architecture in family:
                architecture_tuple = tuple(int(v) for v in expanded_architecture)
                parameter_count = int(rg.count_model_parameters(rg.make_stl_model(task, expanded_architecture, cfg.use_bn)))
                candidates.append(
                    ProbeCandidate(
                        task_name=task_name,
                        architecture=architecture_tuple,
                        parameter_count=parameter_count,
                        depth=len(architecture_tuple),
                    )
                )

    candidates.sort(key=probe_candidate_sort_key)
    return candidates


def count_candidates_by_task(candidates: Sequence[ProbeCandidate]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.task_name] = counts.get(candidate.task_name, 0) + 1
    return counts


def trial_command(
    *,
    args: argparse.Namespace,
    trial_root: Path,
    candidate: ProbeCandidate,
    rank: int,
) -> List[str]:
    child_root = trial_root / f"{rank:03d}_{candidate.task_name}_d{candidate.depth:02d}_p{candidate.parameter_count}"
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_stl_ablation.py"),
        "--data-dir",
        str(args.data_dir),
        "--results-dir",
        str(args.results_dir),
        "--run-root",
        str(child_root),
        "--source-run-root",
        str(args.source_run_root),
        "--tasks",
        candidate.task_name,
        "--architecture",
        ",".join(str(int(v)) for v in candidate.architecture),
        "--repeat-count",
        "1",
        "--repeat-index",
        "1",
        "--batch-size",
        str(int(args.batch_size)),
        "--pin-memory" if bool(args.pin_memory) else "--no-pin-memory",
        "--num-workers",
        str(int(args.num_workers)),
        "--seed",
        str(int(args.seed)),
        "--patience",
        str(int(args.patience)),
        "--delta",
        str(float(args.delta)),
        "--max-epochs",
        str(int(args.probe_epochs)),
        "--lr",
        "1e-3",
        "--weight-decay",
        "1e-4",
        "--grad-clip",
        "1.0",
        "--max-width",
        str(int(args.max_width)),
        "--max-depth",
        str(int(args.max_depth)),
        "--max-neurons",
        str(int(args.max_neurons)),
        "--width-stage-margin-patience",
        str(int(args.width_stage_margin_patience)),
        "--width-stage-min-improve-pct",
        str(float(args.width_stage_min_improve_pct)),
        "--min-width",
        str(int(args.min_width)),
        "--width-step",
        str(int(args.width_step)),
        "--width-count-per-depth",
        str(int(args.width_count_per_depth)),
        "--min-depth",
        str(int(args.min_depth)),
        "--stl-width",
        str(int(args.stl_width)),
        "--stl-depth",
        str(int(args.stl_depth)),
        "--metrics-every",
        str(int(args.metrics_every)),
    ]
    if not bool(args.use_bn):
        command.append("--no-bn")
    if bool(getattr(args, "legacy_architecture_grid", False)):
        command.append("--legacy-architecture-grid")
    if getattr(args, "param_band", None):
        command.extend(
            [
                "--param-band",
                str(int(args.param_band[0])),
                str(int(args.param_band[1])),
            ]
        )
    return command


def terminate_processes(processes: Sequence[subprocess.Popen[Any]]) -> None:
    for proc in processes:
        try:
            launcher.terminate_child_process(proc)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def run_trial(args: argparse.Namespace, trial_root: Path, selected: Sequence[ProbeCandidate]) -> Tuple[bool, List[Dict[str, Any]]]:
    trial_root.mkdir(parents=True, exist_ok=True)
    active: Dict[subprocess.Popen[Any], Tuple[ProbeCandidate, List[str]]] = {}
    results: List[Dict[str, Any]] = []

    for rank, candidate in enumerate(selected, start=1):
        cmd = trial_command(args=args, trial_root=trial_root, candidate=candidate, rank=rank)
        proc = subprocess.Popen(cmd, start_new_session=True)
        active[proc] = (candidate, cmd)

    failed: Optional[Dict[str, Any]] = None
    while active:
        finished: List[subprocess.Popen[Any]] = []
        for proc in list(active):
            code = proc.poll()
            if code is None:
                continue
            candidate, cmd = active[proc]
            if code != 0:
                failed = {
                    "task": candidate.task_name,
                    "architecture": list(candidate.architecture),
                    "parameter_count": int(candidate.parameter_count),
                    "depth": int(candidate.depth),
                    "exit_code": int(code),
                    "command": cmd,
                }
                finished.append(proc)
                continue
            finished.append(proc)
            results.append(
                {
                    "task": candidate.task_name,
                    "architecture": list(candidate.architecture),
                    "parameter_count": int(candidate.parameter_count),
                    "depth": int(candidate.depth),
                    "exit_code": int(code),
                    "command": cmd,
                }
            )

        for proc in finished:
            active.pop(proc, None)

        if failed is not None:
            terminate_processes(list(active.keys()))
            results.append(failed)
            return False, results

        if active:
            time.sleep(2.0)

    return True, results


def main() -> None:
    bootstrap_runtime("run_stl_parallelism_probe")

    args = parse_args()
    tasks = normalize_tasks(args.tasks)
    param_band = stl.normalize_param_band(getattr(args, "param_band", None))
    run_root = resolve_run_root(args, param_band)
    run_root.mkdir(parents=True, exist_ok=True)
    logger = ContinuousLogger(run_root, "stl_parallelism_probe", "stl_parallelism_probe")
    logger.log_console(f"Run root: {run_root}")
    logger.log_console(f"Tasks: {tasks}")
    logger.log_console(f"Probe epochs: {int(args.probe_epochs)}")
    if param_band is not None:
        logger.log_console(f"Parameter decade band: {list(param_band)}")
    logger.log_console(f"Git commit: {rg.git_commit()}")
    logger.log_console(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

    candidates = build_probe_candidates(args, run_root)
    if not candidates:
        raise SystemExit("No probe candidates were generated.")

    logger.log_console(f"Probe candidates: {len(candidates)}")
    counts_by_task = count_candidates_by_task(candidates)
    logger.log_console("Candidates by task: " + ", ".join(f"{task}={count}" for task, count in sorted(counts_by_task.items())))
    logger.log_console("Largest candidates: " + ", ".join(
        f"{candidate.task_name}:{rg.format_architecture_for_report(candidate.architecture)}@{candidate.parameter_count}"
        for candidate in candidates[: min(5, len(candidates))]
    ))

    start_n = max(2, int(args.start_n))
    trial_sizes = probe_trial_sizes(start_n, len(candidates))

    trials: List[Dict[str, Any]] = []
    recommended_parallelism = 1
    for n in trial_sizes:
        selected = select_probe_candidates(candidates, n)
        trial_root = run_root / f"probe_n{n:02d}"
        logger.log_console(f"[PROBE] start n={n} trial_root={trial_root}")
        success, job_results = run_trial(args, trial_root, selected)
        trial_state = {
            "parallelism": int(n),
            "trial_root": str(trial_root),
            "selected_candidates": [
                {
                    "task": candidate.task_name,
                    "architecture": list(candidate.architecture),
                    "parameter_count": int(candidate.parameter_count),
                    "depth": int(candidate.depth),
                }
                for candidate in selected
            ],
            "success": bool(success),
            "job_results": job_results,
        }
        rg.write_json(trial_root / "trial_state.json", trial_state)
        trials.append(trial_state)
        if success:
            recommended_parallelism = int(n)
            logger.log_console(f"[PROBE] success n={n}")
            continue
        logger.log_console(f"[PROBE] failed n={n}")
        break

    recommended_file = run_root / "recommended_parallelism.txt"
    recommended_file.write_text(f"{int(recommended_parallelism)}\n", encoding="utf-8")
    summary = {
        "tasks": tasks,
        "param_band": list(param_band) if param_band is not None else None,
        "probe_epochs": int(args.probe_epochs),
        "start_n": int(start_n),
        "trial_sizes": trial_sizes,
        "candidate_count": len(candidates),
        "candidate_count_by_task": counts_by_task,
        "recommended_parallelism": int(recommended_parallelism),
        "recommended_parallelism_file": str(recommended_file),
        "trials": trials,
    }
    rg.write_json(run_root / "parallelism_probe_summary.json", summary)
    logger.log_console(f"Recommended parallelism: {recommended_parallelism}")
    logger.log_console(f"Recommended parallelism file: {recommended_file}")
    logger.close()


if __name__ == "__main__":
    main()
