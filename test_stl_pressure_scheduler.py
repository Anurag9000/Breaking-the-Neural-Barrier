from __future__ import annotations

import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from MLPS.tabular.shared.dae_dnn import run_stl_ablation_parallel as parallel


class FakeProc:
    _next_pid = 1000

    def __init__(self, job_name: str, child_root: Path, completed_roots: set[Path], launch_counts: dict[str, int]) -> None:
        self.job_name = job_name
        self.child_root = child_root
        self.completed_roots = completed_roots
        self.launch_counts = launch_counts
        self.pid = FakeProc._next_pid
        FakeProc._next_pid += 1
        self.terminated = False
        self.completed = False
        self.poll_calls = 0

    def poll(self) -> int | None:
        if self.terminated:
            return 143
        if self.completed:
            return 0
        self.poll_calls += 1
        if self.job_name == "small" and self.launch_counts["large"] >= 2 and self.poll_calls >= 3:
            self.completed_roots.add(self.child_root)
            self.completed = True
            return 0
        if self.job_name == "large" and self.launch_counts[self.job_name] >= 2 and self.poll_calls >= 2:
            self.completed_roots.add(self.child_root)
            self.completed = True
            return 0
        return None


class STLPressureSchedulerTests(unittest.TestCase):
    def make_args(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(
            batch_size=0,
            pin_memory=False,
            num_workers=0,
            seed=0,
            patience=10,
            delta=1e-4,
            max_epochs=100000000,
            lr=1e-3,
            weight_decay=1e-4,
            grad_clip=1.0,
            max_width=1024,
            max_depth=10,
            max_neurons=10_000_000,
            width_stage_margin_patience=10,
            width_stage_min_improve_pct=1.0,
            min_width=1,
            width_step=1,
            width_count_per_depth=10,
            min_depth=1,
            param_band=None,
            repeat_count=5,
            stl_width=128,
            stl_depth=2,
            metrics_every=0,
            use_bn=True,
            legacy_architecture_grid=False,
            scheduler="pressure_aware",
            max_active_jobs=0,
            host_ram_pressure_limit_pct=90.0,
            host_ram_resume_pct=85.0,
            pressure_poll_interval_sec=0.0,
            pressure_settle_sec=0.0,
            max_retries_per_job=1,
            data_dir="./data",
            results_dir="MLPS/tabular/shared/dae_dnn/results",
            source_run_root="MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current",
        )

    def test_pressure_pause_requeues_same_child_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            small_root = run_root / "task_a" / "_children" / "small"
            large_root = run_root / "task_a" / "_children" / "large"
            jobs_by_task = {
                "task_a": [
                    parallel.ChildJob("task_a", (1,), small_root, 10, 1, "small"),
                    parallel.ChildJob("task_a", (2,), large_root, 20, 1, "large"),
                ]
            }
            completed_roots: set[Path] = set()
            launch_counts = {"small": 0, "large": 0}
            live: list[FakeProc] = []

            def fake_build_jobs(args, tasks, root):
                self.assertEqual(list(tasks), ["task_a"])
                self.assertEqual(root, run_root)
                return jobs_by_task

            def fake_build_cmd(*, args, task_name, architecture, child_run_root):
                name = "large" if child_run_root == large_root else "small"
                return [name]

            def fake_launch(cmd):
                name = cmd[0]
                root = large_root if name == "large" else small_root
                launch_counts[name] += 1
                proc = FakeProc(name, root, completed_roots, launch_counts)
                live.append(proc)
                return proc

            def fake_terminate(proc):
                proc.terminated = True

            def fake_child_completed(child_root, task_name):
                return child_root in completed_roots

            def fake_memory_sample():
                active_names = {proc.job_name for proc in live if not proc.terminated and not proc.completed}
                if active_names == {"small", "large"}:
                    return parallel.MemoryPressureSample(total_mib=1000, available_mib=50, used_pct=95.0)
                if active_names == {"small"}:
                    return parallel.MemoryPressureSample(total_mib=1000, available_mib=400, used_pct=60.0)
                if active_names == {"large"}:
                    return parallel.MemoryPressureSample(total_mib=1000, available_mib=300, used_pct=70.0)
                return parallel.MemoryPressureSample(total_mib=1000, available_mib=500, used_pct=50.0)

            def fake_aggregate(task_name, task_root, child_roots):
                self.assertEqual(task_name, "task_a")
                self.assertIn(small_root, child_roots)
                self.assertIn(large_root, child_roots)
                return {"task": task_name, "comparisons": []}

            logger = mock.Mock()
            args = self.make_args()

            with mock.patch.object(parallel, "build_task_jobs", side_effect=fake_build_jobs), \
                mock.patch.object(parallel, "build_worker_command", side_effect=fake_build_cmd), \
                mock.patch.object(parallel, "launch_child_process", side_effect=fake_launch), \
                mock.patch.object(parallel, "terminate_child_process", side_effect=fake_terminate), \
                mock.patch.object(parallel, "child_completed", side_effect=fake_child_completed), \
                mock.patch.object(parallel, "sample_host_memory_pressure", side_effect=fake_memory_sample), \
                mock.patch.object(parallel, "aggregate_task", side_effect=fake_aggregate), \
                mock.patch.object(parallel.time, "sleep", return_value=None):
                reports = parallel.run_pressure_aware(args, run_root, ["task_a"], logger)

            self.assertEqual(len(reports), 1)
            self.assertEqual(launch_counts["small"], 1)
            self.assertEqual(launch_counts["large"], 2)
            large_state = parallel.load_child_state(large_root)
            self.assertTrue(large_state.get("completed"))
            self.assertEqual(int(large_state.get("pause_count", 0)), 1)
            self.assertIn(large_root, completed_roots)

    def test_launch_child_process_uses_new_session(self) -> None:
        proc = parallel.launch_child_process(["sleep", "30"])
        try:
            self.assertNotEqual(os.getpgid(proc.pid), os.getpgrp())
        finally:
            parallel.terminate_child_process(proc)
            proc.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
