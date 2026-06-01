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

from MLPS.tabular.shared.dae_dnn.mlp import MLP

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
    def test_candidate_directories_are_sorted_by_numeric_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["cand_9_d1_w9", "cand_10_d1_w10", "cand_2_d1_w2", "cand_100_d1_w100"]:
                (root / name).mkdir(parents=True, exist_ok=True)

            ordered = [path.name for path in staged_runner.rg.list_candidate_dirs(root)]

            self.assertEqual(
                ordered,
                ["cand_2_d1_w2", "cand_9_d1_w9", "cand_10_d1_w10", "cand_100_d1_w100"],
            )

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

    def test_load_candidate_model_falls_back_to_checkpoint_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = Path(tmp) / "cand_001_d2_w2"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            model = MLP(in_dim=2, hidden_widths=[2, 2], out_dim=1, use_bn=True)
            payload = {
                "model_state": model.state_dict(),
                "best_state": model.state_dict(),
                "optimizer_state": {},
                "epoch": 1,
                "best_val": 0.123,
                "best_epoch": 1,
                "es_counter": 0,
            }
            torch.save(payload, candidate_dir / "checkpoint_best.pt")
            staged_runner.rg.write_json(
                candidate_dir / "metadata.json",
                {
                    "candidate_dir": str(candidate_dir),
                    "model": {
                        "in_dim": 2,
                        "hidden_widths": [3, 2],
                        "out_dim": 1,
                        "use_bn": True,
                    },
                },
            )

            loaded_model, meta, ckpt = staged_runner.rg.load_candidate_model(candidate_dir, torch.device("cpu"))

            self.assertEqual([int(w) for w in loaded_model.hidden_widths], [2, 2])
            self.assertEqual(meta["source"], "inferred_from_checkpoint_fallback")
            self.assertEqual(int(ckpt["best_epoch"]), 1)

    def test_load_candidate_model_falls_back_to_checkpoint_last_when_best_is_corrupt(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = Path(tmp) / "cand_001_d2_w2"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            model = MLP(in_dim=2, hidden_widths=[2, 2], out_dim=1, use_bn=True)
            payload = {
                "model_state": model.state_dict(),
                "best_state": model.state_dict(),
                "optimizer_state": {},
                "epoch": 1,
                "best_val": 0.123,
                "best_epoch": 1,
                "es_counter": 0,
            }
            torch.save(payload, candidate_dir / "checkpoint_last.pt")
            (candidate_dir / "checkpoint_best.pt").write_text("corrupt", encoding="utf-8")
            staged_runner.rg.write_json(
                candidate_dir / "metadata.json",
                {
                    "candidate_dir": str(candidate_dir),
                    "model": {
                        "in_dim": 2,
                        "hidden_widths": [2, 2],
                        "out_dim": 1,
                        "use_bn": True,
                    },
                },
            )

            loaded_model, meta, ckpt = staged_runner.rg.load_candidate_model(candidate_dir, torch.device("cpu"))

            self.assertEqual([int(w) for w in loaded_model.hidden_widths], [2, 2])
            self.assertEqual(meta["checkpoint_source"], "checkpoint_last.pt")
            self.assertEqual(int(ckpt["best_epoch"]), 1)

    def test_training_loop_resume_falls_back_to_last_checkpoint_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = Path(tmp) / "cand_001_d2_w2"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            true_model = MLP(in_dim=2, hidden_widths=[2, 2], out_dim=1, use_bn=True)
            optimizer = torch.optim.AdamW(true_model.parameters(), lr=1e-3, weight_decay=0.0)
            payload = {
                "model_state": true_model.state_dict(),
                "best_state": true_model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": 1,
                "best_val": 0.123,
                "best_epoch": 1,
                "es_counter": 0,
            }
            torch.save(payload, candidate_dir / "checkpoint_last.pt")
            staged_runner.rg.write_json(
                candidate_dir / "metadata.json",
                {
                    "candidate_dir": str(candidate_dir),
                    "model": {
                        "in_dim": 2,
                        "hidden_widths": [3, 2],
                        "out_dim": 1,
                        "use_bn": True,
                    },
                },
            )

            cfg = staged_runner.rg.RunConfig(
                data_dir="./data",
                results_dir=str(candidate_dir.parent),
                run_root=str(candidate_dir.parent),
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
                max_depth=4,
                max_neurons=10_000,
                width_stage_margin_patience=5,
                width_stage_min_improve_pct=0.0,
                use_bn=True,
                demo=False,
            )
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
            wrong_model = MLP(in_dim=2, hidden_widths=[3, 2], out_dim=1, use_bn=True)
            logger = staged_runner.rg.ContinuousLogger(candidate_dir, "prediction_ae_width_to_depth", "ae_width_to_depth")
            try:
                result = staged_runner.rg.training_loop(
                    task=task,
                    model=wrong_model,
                    candidate_dir=candidate_dir,
                    cfg=cfg,
                    device=torch.device("cpu"),
                    logger=logger,
                    reconstruct=False,
                    resume=True,
                )
            finally:
                logger.close()

            self.assertEqual(result.architecture, [2, 2])
            metadata = staged_runner.rg.read_json(candidate_dir / "metadata.json")
            self.assertEqual(metadata["source"], "inferred_from_resume_checkpoint_fallback")

    def test_restore_rng_state_coerces_torch_tensor_dtype(self):
        original = torch.get_rng_state()
        try:
            payload = {
                "rng_state": {
                    "python": None,
                    "numpy": None,
                    "torch": original.to(dtype=torch.int64),
                    "cuda": None,
                }
            }
            staged_runner.rg.restore_rng_state(payload)
            restored = torch.get_rng_state()
            self.assertTrue(torch.equal(restored, original))
        finally:
            torch.set_rng_state(original)

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
                max_depth=4,
                max_neurons=10_000,
                width_stage_margin_patience=5,
                width_stage_min_improve_pct=0.0,
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

    def test_width_to_depth_switches_when_uniform_width_stages_improve_by_too_little(self):
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
                max_width=4,
                max_depth=4,
                max_neurons=10_000,
                width_stage_margin_patience=2,
                width_stage_min_improve_pct=5.0,
                use_bn=False,
                demo=False,
            )
            values = [1.0, 0.99, 0.98, 0.979, 0.978, 0.977, 0.976, 0.975, 0.974, 0.973]

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
                    base_hidden=[1, 1],
                    phase_name="ae_width_to_depth",
                    mode="width_to_depth",
                    reconstruct=False,
                    batch_controller=None,
                )

            with (task_root / "ae_width_to_depth" / "phase_progress.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(
                [row["architecture"] for row in rows[:6]],
                ["[1, 1]", "[2, 1]", "[2, 2]", "[3, 2]", "[3, 3]", "[3, 3, 3]"],
            )
            self.assertEqual(
                [row["search_phase"] for row in rows[:6]],
                ["width", "width", "width", "width", "width", "depth"],
            )
            self.assertEqual(rows[4]["width_stage_margin_fail"], "2")

    def test_alt_width_switches_to_depth_when_uniform_width_stage_improves_too_little(self):
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
                phases=["ae_alt_width"],
                batch_size=1,
                num_workers=0,
                seed=0,
                stl_width=2,
                stl_depth=2,
                alt_start_width=1,
                alt_start_depth=1,
                patience=50,
                delta=1e-6,
                max_epochs=1,
                lr=1e-3,
                weight_decay=0.0,
                grad_clip=1.0,
                max_width=4,
                max_depth=4,
                max_neurons=10_000,
                width_stage_margin_patience=1,
                width_stage_min_improve_pct=5.0,
                use_bn=False,
                demo=False,
            )
            values = [
                1.0, 0.99, 0.989, 1.1, 0.988, 0.987, 0.986, 0.985, 0.984, 0.983,
                0.982, 0.981, 0.98, 0.979, 0.978, 0.977, 0.976, 0.975, 0.974, 0.973,
            ]

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
                    phase_name="ae_alt_width",
                    mode="alt_width",
                    reconstruct=False,
                    batch_controller=None,
                )

            with (task_root / "ae_alt_width" / "phase_progress.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(
                [row["architecture"] for row in rows[:8]],
                ["[1]", "[2]", "[2, 2]", "[2, 2, 2]", "[2, 2, 2, 2]", "[3, 2, 2, 2]", "[3, 3, 2, 2]", "[3, 3, 3, 2]"],
            )
            self.assertEqual(
                [row["search_phase"] for row in rows[:8]],
                ["width", "width", "depth", "depth", "depth", "width", "width", "width"],
            )
            self.assertEqual(rows[8]["width_stage_margin_fail"], "1")
            self.assertEqual(rows[8]["search_phase"], "width")

    def test_alt_depth_switches_to_depth_when_uniform_width_stage_improves_too_little(self):
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
                phases=["ae_alt_depth"],
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
                max_width=4,
                max_depth=4,
                max_neurons=10_000,
                width_stage_margin_patience=1,
                width_stage_min_improve_pct=5.0,
                use_bn=False,
                demo=False,
            )
            values = [
                1.0, 0.9, 1.1, 0.89, 0.889, 0.888, 0.887, 0.886, 0.885, 0.884,
                0.883, 0.882, 0.881, 0.88, 0.879, 0.878, 0.877, 0.876, 0.875, 0.874,
            ]

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
                    phase_name="ae_alt_depth",
                    mode="alt_depth",
                    reconstruct=False,
                    batch_controller=None,
                )

            with (task_root / "ae_alt_depth" / "phase_progress.csv").open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(
                [row["search_phase"] for row in rows[:10]],
                ["depth", "depth", "depth", "width", "width", "width", "width", "width", "width", "depth"],
            )
            self.assertEqual(
                [row["architecture"] for row in rows[:10]],
                ["[1]", "[1, 1]", "[1, 1, 1]", "[2, 1, 1]", "[2, 2, 1]", "[2, 2, 2]", "[3, 2, 2]", "[3, 3, 2]", "[3, 3, 3]", "[3, 3, 3, 3]"],
            )
            self.assertEqual(rows[8]["width_stage_margin_fail"], "1")


if __name__ == "__main__":
    unittest.main()
