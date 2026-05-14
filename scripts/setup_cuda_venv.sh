#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${1:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_PYTHON="${VENV_DIR}/bin/python"
export VENV_DIR

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "python3 not found on PATH" >&2
  exit 1
fi

venv_needs_rebuild=0
if [ ! -x "${VENV_PYTHON}" ]; then
  venv_needs_rebuild=1
elif ! "${VENV_PYTHON}" - <<'PY' >/dev/null 2>&1; then
import os
import pathlib
import sys

prefix = pathlib.Path(sys.prefix).resolve()
expected = (pathlib.Path.cwd().resolve() / os.environ["VENV_DIR"]).resolve()
raise SystemExit(0 if prefix == expected else 1)
PY
  venv_needs_rebuild=1
fi

if [ "${venv_needs_rebuild}" -eq 1 ]; then
  "${PYTHON_BIN}" -m venv --copies "${VENV_DIR}"
fi

if [ ! -x "${VENV_PYTHON}" ]; then
  echo "virtualenv python is missing at ${VENV_PYTHON}" >&2
  exit 1
fi

"${VENV_PYTHON}" -m pip install --upgrade pip setuptools wheel
"${VENV_PYTHON}" -m pip install -r requirements.txt

"${VENV_PYTHON}" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda_device_count:", torch.cuda.device_count())
    print("cuda_device_name:", torch.cuda.get_device_name(0))
PY

echo
echo "Setup complete."
echo "Activate with:"
echo "  source ${VENV_DIR}/bin/activate"
echo "Start training with:"
echo "  ${VENV_DIR}/bin/python DAE/DNN/run_goliath.py \\"
echo "    --tasks all \\"
echo "    --data-dir ./data \\"
echo "    --results-dir DAE/DNN/results \\"
echo "    --batch-size 448 \\"
echo "    --stl-width 128 \\"
echo "    --stl-depth 2 \\"
echo "    --alt-start-width 2 \\"
echo "    --alt-start-depth 2 \\"
echo "    --patience 5 \\"
echo "    --seed 0"
