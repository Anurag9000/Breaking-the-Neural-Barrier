from __future__ import annotations

import importlib.util
from pathlib import Path


BASE_PATH = Path(__file__).resolve().parents[2] / "Self-Supervised" / "Models" / "dae_tcn_seq_stl_adp_width_to_depth.py"
_spec = importlib.util.spec_from_file_location("dae_tcn_seq_stl_adp_width_to_depth", BASE_PATH)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

ADPConfig = _module.ADPConfig
adp_search = _module.adp_search
make_loaders = _module.make_loaders

BASE_MODEL_PATH = Path(__file__).resolve().parents[2] / "Self-Supervised" / "Models" / "dae_tcn_seq_stl.py"
_base_spec = importlib.util.spec_from_file_location("dae_tcn_seq_stl", BASE_MODEL_PATH)
assert _base_spec is not None and _base_spec.loader is not None
_base_module = importlib.util.module_from_spec(_base_spec)
_base_spec.loader.exec_module(_base_module)

ModelClass = _base_module.DAETCNSeq
