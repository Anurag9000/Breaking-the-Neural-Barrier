from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn as nn

from utils.adp_contract import run_module_adp


DEPTH_EXPANSIONS = 0


class DummyModel(nn.Module):
    def __init__(self, width: int = 4, depth: int = 1) -> None:
        super().__init__()
        self.width = int(width)
        self.depth = int(depth)
        self.weight = nn.Parameter(torch.ones(1))


class DummyConfig:
    adp_mode = "width_to_depth"
    delta = 1e-3
    patience = 2
    width_expansion_patience = 10
    depth_expansion_patience = 2
    width_stage_margin_patience = 10
    width_stage_min_improve_pct = 1.0
    ex_k = 1
    max_width = 4
    max_depth = 10
    max_neurons = 10_000
    min_new_layer_width = 1


def total_neurons(model: DummyModel, width: int, depth: int, widths=None) -> int:
    del model, widths
    return int(width) * int(depth)


def snapshot_arch_and_state(model: DummyModel, state_dict=None):
    del state_dict
    return {
        "width": int(model.width),
        "depth": int(model.depth),
        "widths": [int(model.width)] * int(model.depth),
        "model": copy.deepcopy(model),
    }


def restore_arch_and_state(model: DummyModel, snap, device=None) -> DummyModel:
    del model
    restored = copy.deepcopy(snap["model"])
    return restored.to(device) if device is not None else restored


def expand_width(model: DummyModel, ex_k: int, max_width: int, device=None, cfg=None):
    del device, cfg
    next_width = min(int(max_width), int(model.width) + int(ex_k))
    if next_width == int(model.width):
        return None
    widened = copy.deepcopy(model)
    widened.width = next_width
    return widened


def expand_depth(model: DummyModel, max_depth: int, device=None, cfg=None):
    del device, cfg
    global DEPTH_EXPANSIONS
    if int(model.depth) >= int(max_depth):
        return None
    DEPTH_EXPANSIONS += 1
    deepened = copy.deepcopy(model)
    deepened.depth += 1
    return deepened


def train_with_early_stopping(model: DummyModel, dl_train, dl_val, acfg, device, history):
    del model, dl_train, dl_val, acfg, device
    history.append(1.0)
    return 1.0, {}


class ADPContractTest(unittest.TestCase):
    def setUp(self) -> None:
        global DEPTH_EXPANSIONS
        DEPTH_EXPANSIONS = 0

    def test_width_to_depth_stops_after_depth_patience_without_improvement(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            best_val, best_model = run_module_adp(
                globals(),
                DummyModel(),
                dl_train=[],
                dl_val=[],
                acfg=DummyConfig(),
                device="cpu",
                results_dir=Path(tmpdir),
            )
        self.assertEqual(best_val, 1.0)
        self.assertEqual(best_model.depth, 1)
        self.assertEqual(DEPTH_EXPANSIONS, 2)


class ResumeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.width = 1
        self.depth = 1
        self.weight = nn.Parameter(torch.tensor([1.0]))


class ResumeConfig:
    adp_mode = "width_only"
    delta = 0.0
    patience = 10
    width_expansion_patience = 1
    width_stage_margin_patience = 1
    ex_k = 1
    max_width = 1
    max_depth = 1
    max_neurons = 10
    max_epochs = 3
    lr = 0.1
    weight_decay = 0.0


RESUME_CALLS = 0
RESUME_INTERRUPT = True
RESUME_INITIAL_STEPS = []


def resume_total_neurons(model: ResumeModel, width: int, depth: int, widths=None) -> int:
    del model, widths
    return int(width) * int(depth)


def resume_snapshot(model: ResumeModel, state_dict=None):
    return {
        "width": model.width,
        "depth": model.depth,
        "widths": [model.width],
        "model": copy.deepcopy(model),
        "state": copy.deepcopy(state_dict if state_dict is not None else model.state_dict()),
    }


def resume_restore(model: ResumeModel, snap, device=None):
    del model
    restored = copy.deepcopy(snap["model"])
    restored.load_state_dict(snap["state"])
    return restored.to(device) if device is not None else restored


def resume_train(model, dl_train, dl_val, acfg, device, history):
    del dl_train, dl_val, acfg, device
    global RESUME_CALLS
    RESUME_CALLS += 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    state = optimizer.state.get(model.weight, {})
    step = state.get("step", 0)
    RESUME_INITIAL_STEPS.append(int(step.item()) if torch.is_tensor(step) else int(step))
    if RESUME_INTERRUPT and RESUME_CALLS == 2:
        raise RuntimeError("simulated interruption")
    optimizer.zero_grad(set_to_none=True)
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    value = float(model.weight.item())
    history.append(value)
    return value, copy.deepcopy(model.state_dict())


class ResumeContractTest(unittest.TestCase):
    def test_interrupted_candidate_restores_last_epoch_and_optimizer_state(self) -> None:
        global RESUME_CALLS, RESUME_INTERRUPT
        RESUME_CALLS = 0
        RESUME_INTERRUPT = True
        RESUME_INITIAL_STEPS.clear()
        module_globals = {
            "__name__": "resume_contract_test",
            "train_with_early_stopping": resume_train,
            "snapshot_arch_and_state": resume_snapshot,
            "restore_arch_and_state": resume_restore,
            "total_neurons": resume_total_neurons,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = Path(tmpdir)
            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                run_module_adp(module_globals, ResumeModel(), [], [], ResumeConfig(), "cpu", results_dir=results_dir)

            checkpoint = torch.load(results_dir / "cand_000_d1_w1" / "checkpoint_last.pt", weights_only=False)
            self.assertEqual(checkpoint["completed_epochs"], 1)
            self.assertIn("optimizer_state", checkpoint)
            self.assertIn("rng_state", checkpoint)
            self.assertIn("hyperparameters", checkpoint)

            RESUME_INTERRUPT = False
            best_val, _ = run_module_adp(
                module_globals,
                ResumeModel(),
                [],
                [],
                ResumeConfig(),
                "cpu",
                results_dir=results_dir,
            )
            self.assertLess(best_val, 1.0)
            self.assertEqual(RESUME_CALLS, 4)
            self.assertEqual(RESUME_INITIAL_STEPS, [0, 1, 1, 2])


if __name__ == "__main__":
    unittest.main()
