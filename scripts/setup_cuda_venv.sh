#!/usr/bin/env bash
set -euo pipefail

VENV_DIR="${1:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "python3 not found on PATH" >&2
  exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

python - <<'PY'
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
echo "  python DAE/DNN/run_goliath.py \\"
echo "    --tasks all \\"
echo "    --data-dir ./data \\"
echo "    --results-dir DAE/DNN/results \\"
echo "    --stl-width 128 \\"
echo "    --stl-depth 2 \\"
echo "    --alt-start-width 2 \\"
echo "    --alt-start-depth 2 \\"
echo "    --patience 5 \\"
echo "    --seed 0"
