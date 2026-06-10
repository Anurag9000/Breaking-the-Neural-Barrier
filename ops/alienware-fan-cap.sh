#!/usr/bin/env bash
set -euo pipefail

CPU_BOOST_PATH="/sys/class/hwmon/hwmon6/fan1_boost"
GPU_BOOST_PATH="/sys/class/hwmon/hwmon6/fan2_boost"
CPU_INPUT_PATH="/sys/class/hwmon/hwmon6/fan1_input"
GPU_INPUT_PATH="/sys/class/hwmon/hwmon6/fan2_input"
CPU_MAX_PATH="/sys/class/hwmon/hwmon6/fan1_max"
GPU_MAX_PATH="/sys/class/hwmon/hwmon6/fan2_max"

CPU_TARGET_RPM="${1:-5000}"
GPU_TARGET_RPM="${2:-5000}"
HYSTERESIS_RPM="${3:-200}"
SLEEP_SEC="${4:-1}"

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

read_int() {
  local path="$1"
  if [[ -r "$path" ]]; then
    cat "$path"
  else
    echo 0
  fi
}

rpm_to_boost() {
  local target_rpm="$1"
  local max_rpm="$2"
  if (( target_rpm <= 0 || max_rpm <= 0 )); then
    echo 0
    return
  fi
  local pct=$(( (target_rpm * 100) / max_rpm ))
  clamp "$pct"
}

CPU_MAX_RPM="$(read_int "$CPU_MAX_PATH")"
GPU_MAX_RPM="$(read_int "$GPU_MAX_PATH")"
CPU_CAP_BOOST="$(rpm_to_boost "$CPU_TARGET_RPM" "$CPU_MAX_RPM")"
GPU_CAP_BOOST="$(rpm_to_boost "$GPU_TARGET_RPM" "$GPU_MAX_RPM")"

write_boosts() {
  local cpu_boost="$1"
  local gpu_boost="$2"
  if [[ -w "$CPU_BOOST_PATH" ]]; then
    printf '%s\n' "$cpu_boost" > "$CPU_BOOST_PATH" || true
  fi
  if [[ -w "$GPU_BOOST_PATH" ]]; then
    printf '%s\n' "$gpu_boost" > "$GPU_BOOST_PATH" || true
  fi
}

mode="auto"
trap 'write_boosts 0 0' EXIT

while true; do
  cpu_rpm="$(read_int "$CPU_INPUT_PATH")"
  gpu_rpm="$(read_int "$GPU_INPUT_PATH")"

  if (( cpu_rpm > CPU_TARGET_RPM || gpu_rpm > GPU_TARGET_RPM )); then
    if [[ "$mode" != "capped" ]]; then
      write_boosts "$CPU_CAP_BOOST" "$GPU_CAP_BOOST"
      mode="capped"
    fi
  elif (( cpu_rpm < CPU_TARGET_RPM - HYSTERESIS_RPM && gpu_rpm < GPU_TARGET_RPM - HYSTERESIS_RPM )); then
    if [[ "$mode" != "auto" ]]; then
      write_boosts 0 0
      mode="auto"
    fi
  fi

  sleep "$SLEEP_SEC"
done
