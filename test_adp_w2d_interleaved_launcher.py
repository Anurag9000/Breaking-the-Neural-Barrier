from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path

import MLPS.tabular.shared.dae_dnn.run_adp_w2d_suite_parallel_interleaved as interleaved


class InterleavedLauncherTests(unittest.TestCase):
    def test_build_pending_jobs_spans_repeats(self) -> None:
        args = types.SimpleNamespace(repeat_count=4)
        with tempfile.TemporaryDirectory() as tmpdir:
            pending = list(interleaved.build_pending_jobs(args, ["classification", "simulation"], Path(tmpdir)))
        self.assertEqual(pending[0][0:2], (1, "classification"))
        self.assertEqual(pending[1][0:2], (1, "simulation"))
        self.assertEqual(pending[2][0:2], (2, "classification"))
        self.assertEqual(pending[3][0:2], (2, "simulation"))
        self.assertEqual(pending[-1][0:2], (5, "simulation"))

    def test_job_is_done_uses_existing_task_completed_logic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_root = Path(tmpdir)
            (task_root / "task_state.json").write_text('{"completed": true, "failed": false}', encoding="utf-8")
            self.assertTrue(interleaved.job_is_done(task_root))


if __name__ == "__main__":
    unittest.main()
