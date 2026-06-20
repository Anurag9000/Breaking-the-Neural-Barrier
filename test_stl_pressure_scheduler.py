from __future__ import annotations

import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from MLPS.tabular.shared.dae_dnn import run_stl_ablation_parallel as parallel
from MLPS.tabular.shared.dae_dnn import runtime_tuning


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
        if self.job_name == "small" and self.launch_counts["large"] >= 1 and self.poll_calls >= 3:
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
            gpu_memory_pressure_limit_pct=90.0,
            gpu_memory_resume_pct=85.0,
            gpu_device_index=0,
            pressure_poll_interval_sec=0.0,
            post_launch_sample_delay_sec=0.0,
            batch_backoff_factor=0.5,
            max_retries_per_job=0,
            data_dir="./data",
            results_dir="MLPS/tabular/shared/dae_dnn/results",
            source_run_root="MLPS/tabular/shared/dae_dnn/results/goliath_w2d_staged_current",
        )

    def test_pause_does_not_unlock_new_launch_until_real_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            small_root = run_root / "task_c" / "_children" / "small"
            large_root = run_root / "task_c" / "_children" / "large"
            medium_root = run_root / "task_c" / "_children" / "medium"
            jobs_by_task = {
                "task_c": [
                    parallel.ChildJob("task_c", (1,), small_root, 10, 1, "small"),
                    parallel.ChildJob("task_c", (3,), large_root, 20, 1, "large"),
                    parallel.ChildJob("task_c", (2,), medium_root, 30, 1, "medium"),
                ]
            }
            completed_roots: set[Path] = set()
            launch_counts = {"small": 0, "medium": 0, "large": 0}
            live: list["GateProc"] = []
            launch_order: list[str] = []
            pressure_hits = 0

            class GateProc:
                _next_pid = 5000

                def __init__(self, job_name: str, child_root: Path) -> None:
                    self.job_name = job_name
                    self.child_root = child_root
                    self.pid = GateProc._next_pid
                    GateProc._next_pid += 1
                    self.terminated = False
                    self.completed = False
                    self.poll_calls = 0

                def poll(self) -> int | None:
                    if self.terminated:
                        return 143
                    if self.completed:
                        return 0
                    self.poll_calls += 1
                    if self.job_name == "small" and launch_counts["large"] >= 1 and self.poll_calls >= 4:
                        completed_roots.add(self.child_root)
                        self.completed = True
                        return 0
                    if self.job_name == "medium" and self.poll_calls >= 2:
                        completed_roots.add(self.child_root)
                        self.completed = True
                        return 0
                    if self.job_name == "large" and launch_counts["medium"] >= 1 and self.poll_calls >= 2:
                        completed_roots.add(self.child_root)
                        self.completed = True
                        return 0
                    return None

            def fake_build_jobs(args, tasks, root):
                return jobs_by_task

            def fake_build_cmd(*, args, task_name, architecture, child_run_root, device_mode, batch_size):
                if child_run_root == small_root:
                    return ["small"]
                if child_run_root == medium_root:
                    return ["medium"]
                return ["large"]

            def fake_launch(cmd, env=None, log_path=None):
                name = cmd[0]
                if name == "large" and launch_counts["large"] >= 1:
                    self.assertIn(small_root, completed_roots)
                root = {"small": small_root, "medium": medium_root, "large": large_root}[name]
                launch_counts[name] += 1
                launch_order.append(name)
                proc = GateProc(name, root)
                live.append(proc)
                return proc, None

            def fake_terminate(proc):
                proc.terminated = True

            def fake_child_completed(child_root, task_name):
                return child_root in completed_roots

            def fake_memory_sample():
                nonlocal pressure_hits
                active_names = {proc.job_name for proc in live if not proc.terminated and not proc.completed}
                if active_names == {"small", "large"} and pressure_hits == 0:
                    pressure_hits += 1
                    return parallel.MemoryPressureSample(total_mib=1000, available_mib=50, used_pct=95.0)
                return parallel.MemoryPressureSample(total_mib=1000, available_mib=500, used_pct=50.0)

            def fake_aggregate(task_name, task_root, child_roots):
                return {"task": task_name, "comparisons": []}

            logger = mock.Mock()
            args = self.make_args()
            args.max_active_jobs = 2

            with mock.patch.object(parallel, "build_task_jobs", side_effect=fake_build_jobs), \
                mock.patch.object(parallel, "build_worker_command", side_effect=fake_build_cmd), \
                mock.patch.object(parallel, "launch_child_process", side_effect=fake_launch), \
                mock.patch.object(parallel, "terminate_child_process", side_effect=fake_terminate), \
                mock.patch.object(parallel, "child_completed", side_effect=fake_child_completed), \
                mock.patch.object(parallel, "sample_host_memory_pressure", side_effect=fake_memory_sample), \
                mock.patch.object(parallel, "sample_gpu_memory_pressure", return_value=parallel.GpuPressureSample(total_mib=0, used_mib=0, used_pct=0.0)), \
                mock.patch.object(parallel, "aggregate_task", side_effect=fake_aggregate), \
                mock.patch.object(parallel.time, "sleep", return_value=None):
                reports = parallel.run_pressure_aware(args, run_root, ["task_c"], logger)

            self.assertEqual(len(reports), 1)
            self.assertEqual(launch_order[:2], ["small", "large"])
            self.assertGreaterEqual(launch_counts["large"], 1)
            self.assertEqual(launch_counts["medium"], 1)
            self.assertGreaterEqual(len(launch_order), 4)
            self.assertEqual(launch_order[2], "large")
            self.assertEqual(launch_order[3], "medium")
            self.assertIn(small_root, completed_roots)
            self.assertIn(medium_root, completed_roots)

    def test_pressure_stall_halves_batch_size_before_relaunch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            job_root = run_root / "task_a" / "_children" / "job"
            jobs_by_task = {
                "task_a": [
                    parallel.ChildJob("task_a", (4, 4), job_root, 100, 2, "job"),
                ]
            }
            completed_roots: set[Path] = set()
            launch_batches: list[int] = []
            live: list["BackoffProc"] = []

            class BackoffProc:
                _next_pid = 7000

                def __init__(self, batch_size: int, child_root: Path) -> None:
                    self.batch_size = int(batch_size)
                    self.child_root = child_root
                    self.pid = BackoffProc._next_pid
                    BackoffProc._next_pid += 1
                    self.terminated = False
                    self.completed = False
                    self.poll_calls = 0

                def poll(self) -> int | None:
                    if self.terminated:
                        return 143
                    if self.completed:
                        return 0
                    self.poll_calls += 1
                    active_batches = [proc.batch_size for proc in live if not proc.terminated and not proc.completed]
                    if active_batches and max(active_batches) >= 200:
                        return None
                    if self.batch_size <= 100 and self.poll_calls >= 2:
                        self.completed = True
                        completed_roots.add(self.child_root)
                        return 0
                    return None

            def fake_build_jobs(args, tasks, root):
                return jobs_by_task

            def fake_resolve_task_base_batch_sizes(args, tasks):
                return {"task_a": 200}

            def fake_build_cmd(*, args, task_name, architecture, child_run_root, device_mode, batch_size):
                launch_batches.append(int(batch_size))
                return [str(int(batch_size))]

            def fake_launch(cmd, env=None, log_path=None):
                batch_size = int(cmd[0])
                proc = BackoffProc(batch_size, job_root)
                live.append(proc)
                return proc, None

            def fake_terminate(proc):
                proc.terminated = True

            def fake_child_completed(child_root, task_name):
                return child_root in completed_roots

            def fake_memory_sample():
                active_batches = [proc.batch_size for proc in live if not proc.terminated and not proc.completed]
                if active_batches and max(active_batches) >= 200:
                    return parallel.MemoryPressureSample(total_mib=1000, available_mib=50, used_pct=95.0)
                return parallel.MemoryPressureSample(total_mib=1000, available_mib=500, used_pct=50.0)

            logger = mock.Mock()
            args = self.make_args()
            args.max_active_jobs = 1
            args.batch_size = 200
            args.post_launch_sample_delay_sec = 0.0
            args.pressure_poll_interval_sec = 0.0

            with mock.patch.object(parallel, "build_task_jobs", side_effect=fake_build_jobs), \
                mock.patch.object(parallel, "resolve_task_base_batch_sizes", side_effect=fake_resolve_task_base_batch_sizes), \
                mock.patch.object(parallel, "build_worker_command", side_effect=fake_build_cmd), \
                mock.patch.object(parallel, "launch_child_process", side_effect=fake_launch), \
                mock.patch.object(parallel, "terminate_child_process", side_effect=fake_terminate), \
                mock.patch.object(parallel, "child_completed", side_effect=fake_child_completed), \
                mock.patch.object(parallel, "sample_host_memory_pressure", side_effect=fake_memory_sample), \
                mock.patch.object(parallel, "sample_gpu_memory_pressure", return_value=parallel.GpuPressureSample(total_mib=0, used_mib=0, used_pct=0.0)), \
                mock.patch.object(parallel, "aggregate_task", return_value={"task": "task_a", "comparisons": []}), \
                mock.patch.object(parallel.time, "sleep", return_value=None):
                parallel.run_pressure_aware(args, run_root, ["task_a"], logger)

            self.assertEqual(launch_batches[:2], [200, 100])
            self.assertTrue((run_root / "batch_backoff_state.json").exists())
            state = parallel.load_batch_backoff_state(run_root)
            self.assertGreaterEqual(int(state.get("backoff_count", 0)), 1)
            self.assertAlmostEqual(float(state.get("batch_scale", 0.0)), 0.5, places=6)

    def test_launch_child_process_uses_new_session(self) -> None:
        proc, handle = parallel.launch_child_process(["sleep", "30"])
        try:
            self.assertNotEqual(os.getpgid(proc.pid), os.getpgrp())
        finally:
            parallel.terminate_child_process(proc)
            parallel.close_child_log(handle)
            proc.wait(timeout=5)

    def test_child_completed_prefers_task_state_completion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            child_root = Path(tmp) / "child"
            task_root = child_root / "task_x"
            task_root.mkdir(parents=True, exist_ok=True)
            parallel.rg.write_json(
                task_root / "ablation_state.json",
                {
                    "task": "task_x",
                    "completed": True,
                    "failed": False,
                },
            )

            self.assertTrue(parallel.child_completed(child_root, "task_x"))

    def test_child_failure_is_requeued_without_parent_abort(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            failing_root = run_root / "task_b" / "_children" / "failing"
            success_root = run_root / "task_b" / "_children" / "success"
            jobs_by_task = {
                "task_b": [
                    parallel.ChildJob("task_b", (3,), failing_root, 30, 1, "failing"),
                    parallel.ChildJob("task_b", (1,), success_root, 10, 1, "success"),
                ]
            }
            launch_counts = {"failing": 0, "success": 0}
            completed_roots: set[Path] = set()

            class RetryProc:
                _next_pid = 3000

                def __init__(self, job_name: str, child_root: Path) -> None:
                    self.job_name = job_name
                    self.child_root = child_root
                    self.pid = RetryProc._next_pid
                    RetryProc._next_pid += 1
                    self.done = False
                    self.poll_calls = 0

                def poll(self):
                    if not self.done:
                        self.poll_calls += 1
                        if self.job_name == "success" and self.poll_calls >= 3:
                            self.done = True
                            completed_roots.add(self.child_root)
                            return 0
                        if self.job_name == "failing":
                            self.done = True
                            return 1
                    if self.done:
                        return 0 if self.job_name == "success" else 1
                    return None

            def fake_build_jobs(args, tasks, root):
                return jobs_by_task

            def fake_build_cmd(*, args, task_name, architecture, child_run_root, device_mode, batch_size):
                if child_run_root == success_root:
                    return ["success"]
                return ["failing"]

            def fake_launch(cmd, env=None, log_path=None):
                name = cmd[0]
                launch_counts[name] += 1
                if name == "failing" and launch_counts["failing"] >= 2:
                    raise KeyboardInterrupt
                root = success_root if name == "success" else failing_root
                return RetryProc(name, root), None

            def fake_child_completed(child_root, task_name):
                return child_root in completed_roots

            logger = mock.Mock()
            args = self.make_args()
            args.max_active_jobs = 1

            with mock.patch.object(parallel, "build_task_jobs", side_effect=fake_build_jobs), \
                mock.patch.object(parallel, "build_worker_command", side_effect=fake_build_cmd), \
                mock.patch.object(parallel, "launch_child_process", side_effect=fake_launch), \
                mock.patch.object(parallel, "child_completed", side_effect=fake_child_completed), \
                mock.patch.object(parallel, "sample_host_memory_pressure", return_value=parallel.MemoryPressureSample(total_mib=1000, available_mib=900, used_pct=10.0)), \
                mock.patch.object(parallel, "sample_gpu_memory_pressure", return_value=parallel.GpuPressureSample(total_mib=0, used_mib=0, used_pct=0.0)), \
                mock.patch.object(parallel.time, "sleep", return_value=None):
                with self.assertRaises(KeyboardInterrupt):
                    parallel.run_pressure_aware(args, run_root, ["task_b"], logger)

            self.assertGreaterEqual(launch_counts["failing"], 2)
            self.assertGreaterEqual(launch_counts["success"], 1)
            self.assertEqual(launch_counts["failing"], 2)
            state = parallel.load_child_state(failing_root)
            self.assertIn(state.get("status"), {"retrying", "running"})
            self.assertFalse(bool(state.get("failed", False)))
            self.assertGreaterEqual(int(state.get("failure_count", 0)), 1)

    def test_num_workers_are_pinned_to_zero(self) -> None:
        with mock.patch.dict(os.environ, {"TABULAR_CPU_WORKERS": "8"}, clear=False):
            self.assertEqual(runtime_tuning.resolve_num_workers(8), 0)
            self.assertEqual(runtime_tuning.resolve_num_workers(0), 0)


if __name__ == "__main__":
    unittest.main()
