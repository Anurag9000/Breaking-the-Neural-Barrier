"""
Shim module so run_ae_stl_py_train_eval_undercomplete_autoencoder.py can import
`AE_STL` and `ae_total_neurons` using `from AE_STL import ...`.

We load the real implementation from
`Autoencoder/Supervised/Models/ae_stl.py` via an explicit filesystem-based
import so it works when the script is executed directly (no packages).
"""

from pathlib import Path
import importlib.util
import sys

THIS_DIR = Path(__file__).resolve().parent
MODELS_DIR = THIS_DIR.parent / "Models"
AE_PATH = MODELS_DIR / "ae_stl.py"

spec = importlib.util.spec_from_file_location("ae_stl_module", AE_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load ae_stl.py from {AE_PATH}")

module = importlib.util.module_from_spec(spec)
sys.modules["ae_stl_module"] = module
spec.loader.exec_module(module)

AE_STL = module.AE_STL  # type: ignore[attr-defined]
ae_total_neurons = module.ae_total_neurons  # type: ignore[attr-defined]

