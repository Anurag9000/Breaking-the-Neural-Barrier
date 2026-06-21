#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
PYTHON_BIN="./.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="python"
fi

echo "Triggering cross-platform MLPS emergency kill switch..."
exec "$PYTHON_BIN" scripts/kill_all_runners.py
