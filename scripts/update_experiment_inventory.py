#!/usr/bin/env python3
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = REPO_ROOT / "MLPS" / "tabular" / "shared" / "dae_dnn" / "results"
EXPERIMENTS_ROOT = REPO_ROOT / "experiments"
ACTIVE_ADP_ROOT = RESULTS_ROOT / "adp" / "w2d" / "repeat5_v1"
DOCS_DIR = REPO_ROOT / "docs" / "tabular_dae_dnn"
CSV_PATH = DOCS_DIR / "experiment_inventory.csv"
MD_PATH = DOCS_DIR / "experiment_inventory.md"


@dataclass(frozen=True)
class InventoryRow:
    root_path: str
    source: str
    status: str
    markers: str
    total_files: int
    json_files: int
    csv_files: int
    txt_files: int
    png_files: int
    pt_files: int


def _count_files(root: Path) -> tuple[int, int, int, int, int, int]:
    total = json_files = csv_files = txt_files = png_files = pt_files = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        total += 1
        suffix = path.suffix.lower()
        if suffix == ".json":
            json_files += 1
        elif suffix == ".csv":
            csv_files += 1
        elif suffix == ".txt":
            txt_files += 1
        elif suffix == ".png":
            png_files += 1
        elif suffix in {".pt", ".pth", ".ckpt"}:
            pt_files += 1
    return total, json_files, csv_files, txt_files, png_files, pt_files


def _marker_flags(root: Path) -> list[str]:
    markers: list[str] = []
    for name in (
        "run_metadata.json",
        "task_metadata.json",
        "task_state.json",
        "task_summary.json",
        "final_report.json",
        "training_log.txt",
        "training_stats.csv",
        "phase_progress.csv",
        "search_state.json",
        "candidate_state.json",
        "_batch_size_state.json",
    ):
        if (root / name).exists():
            markers.append(name)
    return markers


def _status_for(root: Path) -> str:
    if root == ACTIVE_ADP_ROOT or root.is_relative_to(ACTIVE_ADP_ROOT):
        return "active"
    if (root / "final_report.json").exists():
        return "complete"
    if (root / "task_state.json").exists() or (root / "training_log.txt").exists():
        return "in_progress"
    return "unknown"


def _discover_roots() -> list[Path]:
    roots: set[Path] = set()
    for base in (RESULTS_ROOT, EXPERIMENTS_ROOT):
        if not base.exists():
            continue
        for marker in base.rglob("run_metadata.json"):
            roots.add(marker.parent)
    if ACTIVE_ADP_ROOT.exists():
        roots.add(ACTIVE_ADP_ROOT)
        for repeat_dir in sorted(ACTIVE_ADP_ROOT.glob("repeat_*")):
            if not repeat_dir.is_dir():
                continue
            for task_dir in sorted(repeat_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                if (task_dir / "task_state.json").exists() or (
                    task_dir / "training_log.txt"
                ).exists():
                    roots.add(task_dir)
    return sorted(roots, key=lambda p: p.as_posix())


def build_inventory() -> list[InventoryRow]:
    rows: list[InventoryRow] = []
    for root in _discover_roots():
        rel = root.relative_to(REPO_ROOT).as_posix()
        if root.is_relative_to(RESULTS_ROOT):
            source = "results"
        elif root.is_relative_to(EXPERIMENTS_ROOT):
            source = "experiments"
        else:
            source = rel.split("/", 1)[0]
        markers = _marker_flags(root)
        total, json_files, csv_files, txt_files, png_files, pt_files = _count_files(root)
        rows.append(
            InventoryRow(
                root_path=rel,
                source=source,
                status=_status_for(root),
                markers=";".join(markers),
                total_files=total,
                json_files=json_files,
                csv_files=csv_files,
                txt_files=txt_files,
                png_files=png_files,
                pt_files=pt_files,
            )
        )
    return rows


def write_csv(rows: list[InventoryRow]) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "root_path",
                "source",
                "status",
                "markers",
                "total_files",
                "json_files",
                "csv_files",
                "txt_files",
                "png_files",
                "pt_files",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.root_path,
                    row.source,
                    row.status,
                    row.markers,
                    row.total_files,
                    row.json_files,
                    row.csv_files,
                    row.txt_files,
                    row.png_files,
                    row.pt_files,
                ]
            )


def write_markdown(rows: list[InventoryRow]) -> None:
    lines = [
        "# Experiment Inventory",
        "",
        "This catalog is generated from the tracked result roots and the active ADP run root.",
        "",
        "| Root | Source | Status | Markers | Files |",
        "|---|---|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row.root_path}` | `{row.source}` | `{row.status}` | "
            f"`{row.markers or '-'}` | {row.total_files} |"
        )
    lines.extend(
        [
            "",
            "Counts include files under each root. Checkpoint binaries are counted in `pt_files` but are not meant to be committed.",
        ]
    )
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = build_inventory()
    write_csv(rows)
    write_markdown(rows)


if __name__ == "__main__":
    main()
