#!/usr/bin/env bash
set -euo pipefail

uid="${1:-$(id -u)}"
username="$(id -un "${uid}")"
cpu_last="$(( $(nproc) - 1 ))"
cpu_range="0-${cpu_last}"

sudo install -d -m 0755 /etc/sysctl.d
sudo tee /etc/sysctl.d/99-mlps-training-priority.conf >/dev/null <<'EOF'
kernel.sched_autogroup_enabled = 0
EOF

sudo install -d -m 0755 /etc/systemd/system/user@.service.d
sudo tee /etc/systemd/system/user@.service.d/96-mlps-delegate.conf >/dev/null <<'EOF'
[Service]
Delegate=
Delegate=cpu cpuset io memory pids
EOF

sudo systemctl set-property "user-${uid}.slice" \
  CPUWeight=10000 \
  StartupCPUWeight=10000 \
  IOWeight=10000 \
  StartupIOWeight=10000 \
  TasksMax=infinity \
  AllowedCPUs="${cpu_range}"

sudo systemctl daemon-reload
sudo sysctl --system >/dev/null

cat <<EOF
Applied:
  user-${uid}.slice CPUWeight=10000
  user-${uid}.slice IOWeight=10000
  user-${uid}.slice TasksMax=infinity
  user-${uid}.slice AllowedCPUs=${cpu_range}
  kernel.sched_autogroup_enabled=0
  user@.service Delegate=cpu cpuset io memory pids

Note:
  The autogroup change applies immediately.
  The user slice weight changes apply immediately.
  The user@.service delegation change is persistent but needs a fresh login,
  a user-manager restart, or a reboot before user scopes gain cpuset/io
  controller delegation.
  Current user: ${username} (uid ${uid})
EOF
