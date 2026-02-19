#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CHECK_ARGS=()
SETUP_ARGS=()
SMOKE_ARGS=()
RUN_ARGS=()

usage() {
  cat <<'EOF'
Usage: bash scripts/vast/run_all_steps.sh [options]

Runs:
  00_check_instance.sh
  01_setup_env.sh
  02_smoke_and_preflight.sh
  03_run_wan.sh

Forwarded options:
  --min-free-gb N
  --repo-url URL
  --prefetch-wan
  --job PATH
  --run-smoke
  --smoke-num-clips N
  --smoke-duration-sec S
  --smoke-steps N
  --smoke-model-id ID
  --skip-verify
  --run-id ID
  --session NAME
  --num-clips N
  --clip-duration-sec S
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --min-free-gb)
      CHECK_ARGS+=(--min-free-gb "${2:-}")
      shift 2
      ;;
    --repo-url)
      SETUP_ARGS+=(--repo-url "${2:-}")
      shift 2
      ;;
    --prefetch-wan)
      SETUP_ARGS+=(--prefetch-wan)
      shift
      ;;
    --job)
      SMOKE_ARGS+=(--job "${2:-}")
      RUN_ARGS+=(--job "${2:-}")
      shift 2
      ;;
    --run-smoke)
      SMOKE_ARGS+=(--run-smoke)
      shift
      ;;
    --smoke-num-clips)
      SMOKE_ARGS+=(--smoke-num-clips "${2:-}")
      shift 2
      ;;
    --smoke-duration-sec)
      SMOKE_ARGS+=(--smoke-duration-sec "${2:-}")
      shift 2
      ;;
    --smoke-steps)
      SMOKE_ARGS+=(--smoke-steps "${2:-}")
      shift 2
      ;;
    --smoke-model-id)
      SMOKE_ARGS+=(--smoke-model-id "${2:-}")
      shift 2
      ;;
    --skip-verify)
      SMOKE_ARGS+=(--skip-verify)
      shift
      ;;
    --run-id)
      RUN_ARGS+=(--run-id "${2:-}")
      shift 2
      ;;
    --session)
      RUN_ARGS+=(--session "${2:-}")
      shift 2
      ;;
    --num-clips)
      RUN_ARGS+=(--num-clips "${2:-}")
      shift 2
      ;;
    --clip-duration-sec)
      RUN_ARGS+=(--clip-duration-sec "${2:-}")
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

echo "== Vast automation: step 00 =="
bash "${SCRIPT_DIR}/00_check_instance.sh" "${CHECK_ARGS[@]}"

echo "== Vast automation: step 01 =="
bash "${SCRIPT_DIR}/01_setup_env.sh" "${SETUP_ARGS[@]}"

echo "== Vast automation: step 02 =="
bash "${SCRIPT_DIR}/02_smoke_and_preflight.sh" "${SMOKE_ARGS[@]}"

echo "== Vast automation: step 03 =="
bash "${SCRIPT_DIR}/03_run_wan.sh" "${RUN_ARGS[@]}"

echo "PASS: All Vast steps completed."
