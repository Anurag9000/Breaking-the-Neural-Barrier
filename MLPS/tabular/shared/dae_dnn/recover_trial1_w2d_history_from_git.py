from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any, Dict, List, Sequence

import run_goliath as rg


DEFAULT_COMMIT = "26105187d"
DEFAULT_CURRENT_RUN_ROOT = "MLPS/tabular/shared/dae_dnn/results/goliath_w2d_anomaly_onward_gpu"
DEFAULT_OUTPUT_SUBDIR = "analysis/recovered_trial1_w2d_history"
REPO_ROOT = Path(__file__).resolve().parents[4]

TASKS = ["representation", "autoencoding", "generation", "denoising"]
PHASES = ["ae_width_to_depth", "stl_from_ae_width_to_depth"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Recover old trial-1 width-to-depth candidate history from Git.")
    p.add_argument("--commit", default=DEFAULT_COMMIT)
    p.add_argument("--current-run-root", default=DEFAULT_CURRENT_RUN_ROOT)
    p.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    return p.parse_args()


def resolve_repo_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def run_git(*args: str) -> bytes:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.stdout


def extract_subtree(commit: str, repo_path: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    archive_bytes = run_git("archive", "--format=tar", commit, repo_path)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as tf:
        tf.extractall(destination)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def architecture_from_metadata(meta: Dict[str, Any], candidate_state: Dict[str, Any]) -> List[int]:
    model = meta.get("model", {})
    hidden = model.get("hidden_widths")
    if hidden:
        return [int(v) for v in hidden]
    raw = candidate_state.get("architecture") or meta.get("hidden_widths")
    if isinstance(raw, list):
        return [int(v) for v in raw]
    if isinstance(raw, str):
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, int):
            return [int(parsed)]
        return [int(v) for v in parsed]
    raise ValueError(f"Could not determine architecture for {meta.get('phase_name', '?')}")


def parameter_count(meta: Dict[str, Any], architecture: Sequence[int]) -> int:
    model_cfg = meta["model"]
    model = rg.make_model(
        int(model_cfg["in_dim"]),
        [int(v) for v in architecture],
        int(model_cfg["out_dim"]),
        bool(model_cfg["use_bn"]),
    )
    return int(sum(p.numel() for p in model.parameters()))


def summarise_candidate(task: str, phase: str, candidate_dir: Path) -> Dict[str, Any]:
    candidate_state = load_json(candidate_dir / "candidate_state.json")
    metadata = load_json(candidate_dir / "metadata.json")
    architecture = architecture_from_metadata(metadata, candidate_state)
    summary = {
        "task": task,
        "family": "ADP" if phase == "ae_width_to_depth" else "STL_paired",
        "phase": phase,
        "candidate_dir": str(candidate_dir),
        "candidate_name": candidate_dir.name,
        "architecture": architecture,
        "best_val": float(candidate_state["best_val"]),
        "best_epoch": int(candidate_state["best_epoch"]),
        "final_epoch": int(candidate_state["final_epoch"]),
        "completed": bool(candidate_state.get("completed", False)),
        "parameter_count": parameter_count(metadata, architecture),
        "training_stats_csv": str(candidate_dir / "training_stats.csv"),
        "training_log_txt": str(candidate_dir / "training_log.txt"),
        "metadata_json": str(candidate_dir / "metadata.json"),
        "candidate_state_json": str(candidate_dir / "candidate_state.json"),
    }
    return summary


def write_summary(output_root: Path, rows: List[Dict[str, Any]]) -> None:
    csv_path = output_root / "recovered_candidate_summary.csv"
    json_path = output_root / "recovered_candidate_summary.json"
    fieldnames = [
        "task",
        "family",
        "phase",
        "candidate_dir",
        "candidate_name",
        "architecture",
        "best_val",
        "best_epoch",
        "final_epoch",
        "completed",
        "parameter_count",
        "training_stats_csv",
        "training_log_txt",
        "metadata_json",
        "candidate_state_json",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_copy = dict(row)
            row_copy["architecture"] = rg.format_architecture_for_report(row["architecture"])
            writer.writerow(row_copy)
    rg.write_json(json_path, {"candidates": rows})


def main() -> None:
    args = parse_args()
    current_run_root = resolve_repo_path(args.current_run_root)
    output_root = current_run_root / args.output_subdir
    raw_root = output_root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    for task in TASKS:
        for phase in PHASES:
            repo_path = f"MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current/{task}/{phase}"
            extract_subtree(args.commit, repo_path, raw_root)
            extracted_phase_root = raw_root / repo_path
            for candidate_state_path in sorted(extracted_phase_root.glob("cand_*/candidate_state.json")):
                candidate_dir = candidate_state_path.parent
                summary_rows.append(summarise_candidate(task, phase, candidate_dir))

    summary_rows.sort(key=lambda row: (row["task"], row["phase"], row["candidate_name"]))
    write_summary(output_root, summary_rows)
    rg.write_json(
        output_root / "recovery_index.json",
        {
            "commit": args.commit,
            "tasks": TASKS,
            "phases": PHASES,
            "raw_root": str(raw_root),
            "summary_csv": str(output_root / "recovered_candidate_summary.csv"),
            "summary_json": str(output_root / "recovered_candidate_summary.json"),
        },
    )

    print(f"Recovered trial-1 width-to-depth history into: {output_root}")
    print(f"Recovered candidates: {len(summary_rows)}")


if __name__ == "__main__":
    main()
