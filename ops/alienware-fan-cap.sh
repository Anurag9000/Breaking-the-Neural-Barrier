#!/usr/bin/env bash
set -euo pipefail

CPU_BOOST_PATH="/sys/class/hwmon/hwmon6/fan1_boost"
GPU_BOOST_PATH="/sys/class/hwmon/hwmon6/fan2_boost"

CPU_TARGET="${1:-73}"
GPU_TARGET="${2:-73}"
SLEEP_SEC="${3:-1}"

clamp() {
  local value="$1"
  if (( value < 0 )); then
    echo 0
  elif (( value > 100 )); then
    echo 100
  else
    echo "$value"
  fi
}

CPU_TARGET="$(clamp "$CPU_TARGET")"
GPU_TARGET="$(clamp "$GPU_TARGET")"

while true; do
  if [[ -w "$CPU_BOOST_PATH" ]]; then
    printf '%s\n' "$CPU_TARGET" > "$CPU_BOOST_PATH" || true
  fi
  if [[ -w "$GPU_BOOST_PATH" ]]; then
    printf '%s\n' "$GPU_TARGET" > "$GPU_BOOST_PATH" || true
  fi
  sleep "$SLEEP_SEC"
done
