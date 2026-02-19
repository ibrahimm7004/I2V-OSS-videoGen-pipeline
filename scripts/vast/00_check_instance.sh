#!/usr/bin/env bash
set -euo pipefail

MIN_FREE_GB="${MIN_FREE_GB:-60}"

usage() {
  cat <<'EOF'
Usage: bash scripts/vast/00_check_instance.sh [--min-free-gb N]

Checks host basics (disk/GPU) before setup.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --min-free-gb)
      MIN_FREE_GB="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if ! [[ "${MIN_FREE_GB}" =~ ^[0-9]+$ ]]; then
  echo "ERROR: --min-free-gb must be an integer. Got: ${MIN_FREE_GB}" >&2
  exit 2
fi

echo "== Instance Check =="
echo "Hostname: $(hostname)"
echo "Uptime:"
uptime || true
echo
echo "Disk usage:"
df -h || true
echo

echo "GPU summary:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,driver_version --format=csv,noheader || nvidia-smi || true
else
  echo "WARNING: nvidia-smi not found on PATH"
fi
echo

if [[ ! -d /workspace ]]; then
  echo "ERROR: /workspace not found. This script expects Vast-style workspace storage." >&2
  exit 1
fi

FREE_KB="$(df -Pk /workspace | awk 'NR==2 {print $4}')"
REQUIRED_KB="$((MIN_FREE_GB * 1024 * 1024))"
FREE_GB="$(awk "BEGIN { printf \"%.1f\", ${FREE_KB} / 1024 / 1024 }")"

echo "/workspace free space: ${FREE_GB} GB (required: ${MIN_FREE_GB} GB)"
if (( FREE_KB < REQUIRED_KB )); then
  echo "ERROR: Not enough free space under /workspace. Need >= ${MIN_FREE_GB} GB." >&2
  exit 1
fi

echo "PASS: Instance checks completed."
