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

from DAE.DNN.mlp import MLP

from MLPS.tabular.shared.dae_dnn.adp_staged_width import can_widen_staged, expand_width_staged, next_staged_widths

REPO_ROOT = Path(__file__).resolve().parent
RUNNER_PATH = REPO_ROOT / "MLPS" / "tabular" / "shared" / "dae_dnn" / "run_goliath_staged_width.py"

_SPEC = importlib.util.spec_from_file_location("run_goliath_staged_width_under_test", RUNNER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
staged_runner = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = staged_runner
_SPEC.loader.exec_module(staged_runner)


class DummyLoader:
    def __init__(self) -> None:
        self.batch_size = 1
        self.dataset = [0]


class StagedWidthTests(unittest.TestCase):
    def test_next_staged_widths_fills_one_layer_at_a_time(self):
        self.assertEqual(next_staged_widths([2, 2], 1, 10), [3, 2])
        self.assertEqual(next_staged_widths([3, 2], 1, 10), [3, 3])
        self.assertEqual(next_staged_widths([3, 3], 1, 10), [4, 3])
        self.assertEqual(next_staged_widths([4, 3], 1, 10), [4, 4])

    def test_next_staged_widths_respects_max_width_per_layer(self):
        self.assertEqual(next_staged_widths([4, 3], 1, 4), [4, 4])
        self.assertIsNone(next_staged_widths([4, 4], 1, 4))

    def test_expand_width_staged_rebuilds_model(self):
        model = MLP(in_dim=2, hidden_widths=[2, 2], out_dim=1, use_bn=False)
        widened = expand_width_staged(model, 1, 10)
        self.assertIsNotNone(widened)
        self.assertEqual(widened.hidden_widths, [3, 2])

    def test_can_widen_staged_allows_finishing_a_partially_filled_depth(self):
        model = MLP(in_dim=2, hidden_widths=[4, 3], out_dim=1, use_bn=False)
        self.assertTrue(can_widen_staged(model, 4, 10_000))

    def test_depth_waits_until_widths_are_uniform(self):
        loader = DummyLoader()
        task = SimpleNamespace(
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

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = staged_runner.rg.RunConfig(
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
                alt_start_width=1,
                alt_start_depth=1,
                patience=1,
                delta=1e-6,
                max_epochs=1,
                lr=1e-3,
                weight_decay=0.0,
                grad_clip=1.0,
                max_width=3,
                max_depth=3,
                max_neurons=10_000,
                use_bn=False,
                demo=False,
            )
            values = [1.0, 2.0, 0.5, 1.5, 2.5, 3.5]

            def fake_training_loop(*, task, model, candidate_dir, cfg, device, logger, reconstruct, resume=True, batch_controller=None, display_best_floor=None):
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
                staged_runner.rg.write_json(
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
                return staged_runner.rg.CandidateResult(
                    best_val=val,
                    best_epoch=1,
                    final_epoch=1,
                    best_checkpoint=candidate_dir / "checkpoint_best.pt",
                    last_checkpoint=candidate_dir / "checkpoint_last.pt",
                    candidate_dir=candidate_dir,
                    architecture=[int(w) for w in model.hidden_widths],
                )

            task_root = root / "prediction"
            task_root.mkdir(parents=True, exist_ok=True)
            with mock.patch.object(staged_runner.rg, "training_loop", side_effect=fake_training_loop), \
                mock.patch.object(staged_runner.rg, "eval_final", return_value={"test_loss": 0.0}), \
                mock.patch.object(staged_runner.rg, "plot_candidate_stats", return_value=None):
                staged_runner.run_growth_phase(
                    task=task,
                    task_root=task_root,
                    cfg=cfg,
                    device=torch.device("cpu"),
                    base_hidden=[1],
                    phase_name="ae_width_to_depth",
                    mode="width_to_depth",
                    reconstruct=False,
                    batch_controller=None,
                )

            with (task_root / "ae_width_to_depth" / "phase_progress.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(
                [row["architecture"] for row in rows],
                ["[1]", "[2]", "[2, 2]", "[3, 2]", "[3, 3]", "[3, 3, 3]"],
            )
            self.assertEqual(
                [row["search_phase"] for row in rows],
                ["width", "width", "depth", "width", "width", "depth"],
            )
            self.assertEqual(
                [int(row["width_fail"]) for row in rows],
                [0, 1, 1, 0, 1, 1],
            )


if __name__ == "__main__":
    unittest.main()
