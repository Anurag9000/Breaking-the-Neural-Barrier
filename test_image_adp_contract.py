import copy
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from utils.adp_contract import run_module_adp


class FakeModel:
    def __init__(self, hidden_widths):
        self.hidden_widths = list(hidden_widths)


def fake_snapshot(model, state_dict=None):
    return {"hidden_widths": list(model.hidden_widths)}


def fake_restore(model, snap, device=None):
    return FakeModel(list(snap["hidden_widths"]))


def fake_total_neurons(model, width=None, depth=None, widths=None):
    use_widths = widths if widths is not None else model.hidden_widths
    return int(sum(use_widths))


def fake_expand_width(model, ex_k, max_width):
    widths = list(model.hidden_widths)
    if not widths:
        return None
    target = min(max(widths) + max(1, int(ex_k)), int(max_width)) if len(set(widths)) == 1 else max(widths)
    next_widths = list(widths)
    for idx, width in enumerate(next_widths):
        if width < target:
            next_widths[idx] = width + 1
            model.hidden_widths = next_widths
            return model
    return None


def fake_expand_depth(model, max_depth):
    if not model.hidden_widths or len(model.hidden_widths) >= max_depth:
        return None
    if len(set(model.hidden_widths)) != 1:
        return None
    model.hidden_widths = list(model.hidden_widths) + [int(model.hidden_widths[-1])]
    return model


@dataclass
class FakeConfig:
    adp_mode: str
    delta: float = 1e-6
    trials_width: int = 5
    trials_depth: int = 5
    ex_k: int = 1
    max_width: int = 10
    max_depth: int = 10
    max_neurons: int = 1000
    width_stage_margin_patience: int = 1
    width_stage_min_improve_pct: float = 20.0


class ADPContractImageTests(unittest.TestCase):
    def make_train_fn(self, losses, visited):
        def _train(model, dl_train=None, dl_val=None, acfg=None, device=None, history=None, logger=None, verbose=True):
            widths = tuple(model.hidden_widths)
            visited.append(widths)
            return losses[widths]

        return _train

    def make_module_globals(self, losses, visited):
        return {
            "train_with_early_stopping": self.make_train_fn(losses, visited),
            "expand_width": fake_expand_width,
            "expand_depth": fake_expand_depth,
            "snapshot_arch_and_state": fake_snapshot,
            "restore_arch_and_state": fake_restore,
            "total_neurons": fake_total_neurons,
            "__name__": "fake_image_adp",
        }

    def test_width_to_depth_deepens_only_after_uniform_width(self):
        losses = {
            (2, 2): 10.0,
            (3, 2): 9.0,
            (3, 3): 9.0,
            (3, 3, 3): 8.0,
        }
        visited = []
        with tempfile.TemporaryDirectory() as tmpdir:
            best_val, model = run_module_adp(
                self.make_module_globals(losses, visited),
                FakeModel([2, 2]),
                None,
                None,
                FakeConfig(adp_mode="width_to_depth", max_width=3, max_depth=3),
                "cpu",
                results_dir=Path(tmpdir),
            )
        self.assertEqual(visited[:4], [(2, 2), (3, 2), (3, 3), (3, 3, 3)])
        self.assertEqual(model.hidden_widths, [3, 3, 3])
        self.assertAlmostEqual(best_val, 8.0)

    def test_width_to_depth_resets_width_patience_after_depth_fill(self):
        losses = {
            (28, 28): 10.2,
            (29, 28): 10.1,
            (29, 29): 10.0,
            (29, 29, 1): 9.9,
            (29, 29, 2): 9.8,
            (29, 29, 3): 9.7,
            (29, 29, 4): 9.6,
            (29, 29, 5): 9.5,
            (29, 29, 6): 9.4,
            (29, 29, 7): 9.3,
            (29, 29, 8): 9.2,
            (29, 29, 9): 9.1,
            (29, 29, 10): 9.0,
            (29, 29, 11): 8.9,
            (29, 29, 12): 8.8,
            (29, 29, 13): 8.7,
            (29, 29, 14): 8.6,
            (29, 29, 15): 8.5,
            (29, 29, 16): 8.4,
            (29, 29, 17): 8.3,
            (29, 29, 18): 8.2,
            (29, 29, 19): 8.1,
            (29, 29, 20): 8.0,
            (29, 29, 21): 7.9,
            (29, 29, 22): 7.8,
            (29, 29, 23): 7.7,
            (29, 29, 24): 7.6,
            (29, 29, 25): 7.5,
            (29, 29, 26): 7.4,
            (29, 29, 27): 7.3,
            (29, 29, 28): 7.2,
            (29, 29, 29): 7.1,
            (30, 29, 29): 7.0,
            (30, 30, 29): 6.9,
            (30, 30, 30): 6.8,
        }
        visited = []
        with tempfile.TemporaryDirectory() as tmpdir:
            best_val, model = run_module_adp(
                self.make_module_globals(losses, visited),
                FakeModel([28, 28]),
                None,
                None,
                FakeConfig(
                    adp_mode="width_to_depth",
                    max_width=30,
                    max_depth=3,
                    width_stage_margin_patience=1,
                    width_stage_min_improve_pct=1_000.0,
                ),
                "cpu",
                results_dir=Path(tmpdir),
            )
        self.assertIn((29, 29, 29), visited)
        self.assertIn((30, 29, 29), visited)
        self.assertEqual(model.hidden_widths, [30, 30, 30])
        self.assertAlmostEqual(best_val, 6.8)

    def test_depth_only_fills_to_uniform_before_depth(self):
        losses = {
            (3, 2): 5.0,
            (3, 3): 4.0,
            (3, 3, 3): 3.0,
        }
        visited = []
        with tempfile.TemporaryDirectory() as tmpdir:
            best_val, model = run_module_adp(
                self.make_module_globals(losses, visited),
                FakeModel([3, 2]),
                None,
                None,
                FakeConfig(adp_mode="depth_only", width_stage_min_improve_pct=1.0, max_width=3, max_depth=3),
                "cpu",
                results_dir=Path(tmpdir),
            )
        self.assertEqual(visited[:3], [(3, 2), (3, 3), (3, 3, 3)])
        self.assertEqual(model.hidden_widths, [3, 3, 3])
        self.assertAlmostEqual(best_val, 3.0)


if __name__ == "__main__":
    unittest.main()
