from __future__ import annotations

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parent
RUN_GOLIATH_PATH = REPO_ROOT / "MLPS" / "tabular" / "shared" / "dae_dnn" / "run_goliath.py"

_SPEC = importlib.util.spec_from_file_location("run_goliath_under_test", RUN_GOLIATH_PATH)
assert _SPEC is not None and _SPEC.loader is not None
rg = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = rg
_SPEC.loader.exec_module(rg)


class DummyLoader:
    def __init__(self) -> None:
        self.batch_size = 1
        self.dataset = [0]


class ADPSemanticsTests(unittest.TestCase):
    def make_task(self):
        loader = DummyLoader()
        return SimpleNamespace(
            name="prediction",
            in_dim=2,
            out_dim=1,
            task_type="regression",
            extra={},
            loss_fn=None,
            metrics_fn=None,
            train_loader=loader,
            val_loader=loader,
            test_loader=loader,
        )

    def make_cfg(self, root: Path, patience: int = 2, max_width: int = 8, max_depth: int = 8):
        return rg.RunConfig(
            data_dir="./data",
            results_dir=str(root),
            run_root=str(root),
            tasks=["prediction"],
            phases=["ae_width_to_depth"],
            batch_size=1,
            num_workers=0,
            seed=0,
            stl_width=2,
            stl_depth=2,
            alt_start_width=2,
            alt_start_depth=2,
            patience=patience,
            delta=1e-6,
            max_epochs=1,
            lr=1e-3,
            weight_decay=0.0,
            grad_clip=1.0,
            max_width=max_width,
            max_depth=max_depth,
            max_neurons=10_000,
            use_bn=False,
            demo=False,
        )

    def fake_training_loop_factory(self, values):
        values = list(values)

        def fake_training_loop(*, task, model, candidate_dir, cfg, device, logger, reconstruct, resume=True, batch_controller=None):
            idx = int(candidate_dir.name.split("_")[1])
            val = float(values[idx])
            candidate_dir.mkdir(parents=True, exist_ok=True)
            state = model.state_dict()
            payload = {
                "model_state": state,
                "best_state": state,
                "optimizer_state": {},
                "epoch": 1,
                "best_val": val,
                "best_epoch": 1,
                "es_counter": 0,
            }
            torch.save(payload, candidate_dir / "checkpoint_best.pt")
            torch.save(payload, candidate_dir / "checkpoint_last.pt")
            rg.write_json(
                candidate_dir / "candidate_state.json",
                {
                    "completed": True,
                    "task": task.name,
                    "phase": candidate_dir.parent.name,
                    "candidate_dir": str(candidate_dir),
                    "best_val": val,
                    "best_epoch": 1,
                    "final_epoch": 1,
                    "architecture": [int(w) for w in model.hidden_widths],
                    "reconstruct": reconstruct,
                    "checkpoint_best": str(candidate_dir / "checkpoint_best.pt"),
                    "checkpoint_last": str(candidate_dir / "checkpoint_last.pt"),
                },
            )
            return rg.CandidateResult(
                best_val=val,
                best_epoch=1,
                final_epoch=1,
                best_checkpoint=candidate_dir / "checkpoint_best.pt",
                last_checkpoint=candidate_dir / "checkpoint_last.pt",
                candidate_dir=candidate_dir,
                architecture=[int(w) for w in model.hidden_widths],
            )

        return fake_training_loop

    def run_mode(self, mode: str, values, patience: int = 2, max_width: int = 8, max_depth: int = 8):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            task = self.make_task()
            cfg = self.make_cfg(root, patience=patience, max_width=max_width, max_depth=max_depth)
            task_root = root / "prediction"
            task_root.mkdir(parents=True, exist_ok=True)
            phase_name = next(name for name, phase_mode in rg.GOLIATH_ADP_PHASES if phase_mode == mode)
            fake_training_loop = self.fake_training_loop_factory(values)

            with mock.patch.object(rg, "training_loop", side_effect=fake_training_loop), \
                mock.patch.object(rg, "eval_final", return_value={"test_loss": 0.0}), \
                mock.patch.object(rg, "plot_candidate_stats", return_value=None):
                rg.run_growth_phase(
                    task=task,
                    task_root=task_root,
                    cfg=cfg,
                    device=torch.device("cpu"),
                    base_hidden=[2, 2],
                    phase_name=phase_name,
                    mode=mode,
                    reconstruct=False,
                    batch_controller=None,
                )

            progress_path = task_root / phase_name / "phase_progress.csv"
            with progress_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            return rows

    def assert_phases_and_architectures(self, rows, expected_phases, expected_architectures):
        self.assertEqual([row["search_phase"] for row in rows], expected_phases)
        self.assertEqual([row["architecture"] for row in rows], expected_architectures)

    def test_width_only_stops_after_patience_failures(self):
        rows = self.run_mode("width_only", values=[1.0, 2.0, 3.0], patience=2, max_width=4)
        self.assert_phases_and_architectures(rows, ["width", "width", "width"], ["[2, 2]", "[3, 3]", "[4, 4]"])

    def test_depth_only_stops_after_patience_failures(self):
        rows = self.run_mode("depth_only", values=[1.0, 2.0, 3.0], patience=2, max_depth=4)
        self.assert_phases_and_architectures(rows, ["depth", "depth", "depth"], ["[2, 2]", "[2, 2, 2]", "[2, 2, 2, 2]"])

    def test_width_to_depth_switches_after_width_failures_and_stops_after_depth_failures(self):
        rows = self.run_mode("width_to_depth", values=[1.0, 2.0, 3.0, 4.0], patience=2, max_width=4, max_depth=3)
        self.assert_phases_and_architectures(rows, ["width", "width", "width", "depth"], ["[2, 2]", "[3, 3]", "[4, 4]", "[4, 4, 4]"])

    def test_depth_to_width_switches_after_depth_failures_and_stops_after_width_failures(self):
        rows = self.run_mode("depth_to_width", values=[1.0, 2.0, 3.0, 4.0], patience=2, max_width=3, max_depth=4)
        self.assert_phases_and_architectures(rows, ["depth", "depth", "depth", "width"], ["[2, 2]", "[2, 2, 2]", "[2, 2, 2, 2]", "[3, 3, 3, 3]"])

    def test_alt_width_runs_width_then_depth_blocks_and_terminates_when_no_expansions_remain(self):
        rows = self.run_mode("alt_width", values=[1.0, 2.0, 3.0, 4.0, 5.0], patience=1, max_width=4, max_depth=4)
        self.assert_phases_and_architectures(
            rows,
            ["width", "width", "depth", "width", "depth"],
            ["[2, 2]", "[3, 3]", "[3, 3, 3]", "[4, 4, 4]", "[4, 4, 4, 4]"],
        )

    def test_alt_depth_runs_depth_then_width_blocks_and_terminates_when_no_expansions_remain(self):
        rows = self.run_mode("alt_depth", values=[1.0, 2.0, 3.0, 4.0, 5.0], patience=1, max_width=4, max_depth=4)
        self.assert_phases_and_architectures(
            rows,
            ["depth", "depth", "width", "depth", "width"],
            ["[2, 2]", "[2, 2, 2]", "[3, 3, 3]", "[3, 3, 3, 3]", "[4, 4, 4, 4]"],
        )


if __name__ == "__main__":
    unittest.main()
