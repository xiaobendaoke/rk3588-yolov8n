#!/usr/bin/env bash
set -euo pipefail

BOARD_IP="${BOARD_IP:-192.168.88.2}"
BOARD_USER="${BOARD_USER:-root}"
BOARD_PASS="${BOARD_PASS:-root}"
SSH_OPTS='-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null'

if ! command -v sshpass >/dev/null 2>&1; then
  echo "[ERR] sshpass not found"
  exit 1
fi

run_remote() {
  sshpass -p "$BOARD_PASS" ssh $SSH_OPTS "$BOARD_USER@$BOARD_IP" "$@"
}

echo "[INFO] checking board connectivity..."
ping -c 2 "$BOARD_IP" >/dev/null

mkdir -p ./logs
OUT="./logs/baseline_$(date +%Y%m%d_%H%M%S).txt"
{
  echo "===== host time ====="
  date -Is
  echo "===== remote basic ====="
  run_remote 'hostname; cat /etc/os-release | head -n 5; python3 -V; ip -4 -br addr'
  echo "===== imports ====="
  run_remote "python3 - <<'PY'
import cv2, numpy, sqlite3
import rknnlite.api
print('imports ok')
PY"
  echo "===== npu ====="
  run_remote 'cat /sys/class/devfreq/fdab0000.npu/cur_freq; ls /sys/class/devfreq/fdab0000.npu/'
  echo "===== camera ====="
  run_remote 'v4l2-ctl --device=/dev/video21 --info'
} | tee "$OUT"

echo "[OK] baseline report saved: $OUT"
