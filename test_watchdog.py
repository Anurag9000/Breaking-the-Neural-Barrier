from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parent
WATCHDOG_PATH = REPO_ROOT / "MLPS" / "tabular" / "shared" / "dae_dnn" / "run_with_watchdog.py"

_SPEC = importlib.util.spec_from_file_location("run_with_watchdog_under_test", WATCHDOG_PATH)
assert _SPEC is not None and _SPEC.loader is not None
wd = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = wd
_SPEC.loader.exec_module(wd)


class FakeProcess:
    def __init__(self, exit_code: int) -> None:
        self.exit_code = exit_code

    def poll(self):
        return self.exit_code

    def wait(self):
        return self.exit_code

    def terminate(self):
        return None

    def kill(self):
        return None


class WatchdogTests(unittest.TestCase):
    def test_ensure_run_root_arg_injects_for_goliath(self):
        command = [".venv/bin/python", "MLPS/tabular/shared/dae_dnn/run_goliath.py", "--tasks", "all"]
        prepared = wd.ensure_run_root_arg(command, Path("/tmp/run-root"))
        self.assertEqual(prepared[-2:], ["--run-root", "/tmp/run-root"])

    def test_ensure_run_root_arg_preserves_existing_flag(self):
        command = [
            ".venv/bin/python",
            "MLPS/tabular/shared/dae_dnn/run_goliath.py",
            "--run-root",
            "/tmp/already",
            "--tasks",
            "all",
        ]
        prepared = wd.ensure_run_root_arg(command, Path("/tmp/run-root"))
        self.assertEqual(prepared, command)

    def test_heartbeat_snapshot_counts_epoch_events_and_state_progress(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log_a = root / "a" / "training_log.txt"
            log_a.parent.mkdir(parents=True, exist_ok=True)
            log_a.write_text("epoch=1\nepoch=2\n", encoding="utf-8")
            log_b = root / "b" / "training_log.txt"
            log_b.parent.mkdir(parents=True, exist_ok=True)
            log_b.write_text("epoch=1\n", encoding="utf-8")

            wd.write_json(root / "a" / "candidate_state.json", {"epoch": 2, "completed": False})
            wd.write_json(root / "m" / "method_state.json", {"candidate_index": 3})
            wd.write_json(root / "t" / "task_state.json", {"next_phase_index": 4})

            snapshot = wd.heartbeat_snapshot(root)

            self.assertEqual(snapshot.epoch_events, 3)
            self.assertEqual(snapshot.candidate_epoch_total, 2)
            self.assertEqual(snapshot.method_progress_total, 3)
            self.assertEqual(snapshot.task_progress_total, 4)
            self.assertIsNotNone(snapshot.latest_path)

    def test_register_hiccup_starts_window_and_increments_counts(self):
        restart_count, first_hiccup_at, hiccup_restarts = wd.register_hiccup(
            restart_count=0,
            first_hiccup_at=None,
            hiccup_restarts=0,
            now=12.5,
        )
        self.assertEqual(restart_count, 1)
        self.assertEqual(first_hiccup_at, 12.5)
        self.assertEqual(hiccup_restarts, 1)

    def test_run_supervised_counts_nonzero_exits_against_restart_budget(self):
        popen_calls = []

        def fake_popen(command):
            popen_calls.append(list(command))
            if len(popen_calls) == 1:
                return FakeProcess(7)
            if len(popen_calls) == 2:
                return FakeProcess(7)
            return FakeProcess(0)

        finalize_calls = []

        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp)
            candidate_dir = run_root / "prediction" / "ae_width_to_depth" / "cand_000_d2_w2"
            candidate_dir.mkdir(parents=True, exist_ok=True)

            with mock.patch.object(wd.subprocess, "Popen", side_effect=fake_popen), \
                mock.patch.object(wd, "heartbeat_snapshot", return_value=wd.HeartbeatSnapshot(None, 0, 0, 0, 0, 0)), \
                mock.patch.object(wd, "latest_incomplete_candidate", return_value=candidate_dir), \
                mock.patch.object(wd, "finalize_candidate", side_effect=lambda cand, device: finalize_calls.append(cand) or True), \
                mock.patch.object(wd.torch.cuda, "is_available", return_value=False):
                exit_code = wd.run_supervised(
                    command=["python", "MLPS/tabular/shared/dae_dnn/run_goliath.py"],
                    run_root=run_root,
                    idle_seconds=9999,
                    max_restarts=2,
                    burst_limit=99,
                    burst_window_seconds=9999,
                    poll_seconds=1,
                    grace_seconds=1,
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(finalize_calls), 1)
        self.assertGreaterEqual(len(popen_calls), 3)


if __name__ == "__main__":
    unittest.main()
