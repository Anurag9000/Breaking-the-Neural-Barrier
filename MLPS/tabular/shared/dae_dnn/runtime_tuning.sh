#!/usr/bin/env bash

# Best-effort shell-side runtime tuning for tabular DAE/DNN launchers.
# Safe to source from wrappers with `set -euo pipefail`.

tabular_runtime_detect_cpu_cores() {
  local cores=""
  if command -v nproc >/dev/null 2>&1; then
    cores="$(nproc 2>/dev/null || true)"
  fi
  if [[ -z "${cores}" ]] && command -v getconf >/dev/null 2>&1; then
    cores="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
  fi
  if [[ -z "${cores}" ]]; then
    cores="1"
  fi
  if [[ "${cores}" -lt 1 ]]; then
    cores="1"
  fi
  printf '%s\n' "${cores}"
}

tabular_runtime_bootstrap() {
  local cpu_cores
  cpu_cores="$(tabular_runtime_detect_cpu_cores)"

  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${cpu_cores}}"
  export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${cpu_cores}}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-${cpu_cores}}"
  export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-${cpu_cores}}"
  export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-${cpu_cores}}"
  export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-${cpu_cores}}"
  export TORCH_INTEROP_THREADS="${TORCH_INTEROP_THREADS:-1}"
  export OMP_DYNAMIC="${OMP_DYNAMIC:-FALSE}"
  export MKL_DYNAMIC="${MKL_DYNAMIC:-FALSE}"
  export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-ACTIVE}"

  if command -v renice >/dev/null 2>&1; then
    renice -n -20 -p "$$" >/dev/null 2>&1 || true
  fi
  if command -v ionice >/dev/null 2>&1; then
    ionice -c2 -n0 -p "$$" >/dev/null 2>&1 || true
  fi
  if command -v chrt >/dev/null 2>&1; then
    chrt -b -p 0 "$$" >/dev/null 2>&1 || true
  fi
}
