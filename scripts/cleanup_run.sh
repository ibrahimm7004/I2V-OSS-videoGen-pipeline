#!/usr/bin/env bash
set -euo pipefail

RUN_ID=""
OUTPUT_ROOT="outputs"
DELETE_CACHE=0
CACHE_PATH="${HF_HUB_CACHE:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --delete-cache)
      DELETE_CACHE=1
      shift
      ;;
    --cache-path)
      CACHE_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "${RUN_ID}" ]]; then
  echo "Usage: bash scripts/cleanup_run.sh --run-id <id> [--output-root outputs] [--delete-cache] [--cache-path <path>]"
  exit 1
fi

RUN_DIR="${OUTPUT_ROOT}/${RUN_ID}"
if [[ -d "${RUN_DIR}" ]]; then
  echo "Deleting run directory: ${RUN_DIR}"
  rm -rf "${RUN_DIR}"
else
  echo "Run directory not found: ${RUN_DIR}"
fi

if [[ ${DELETE_CACHE} -eq 1 ]]; then
  if [[ -z "${CACHE_PATH}" && -n "${HF_HOME:-}" ]]; then
    CACHE_PATH="${HF_HOME}/hub"
  fi
  if [[ -z "${CACHE_PATH}" ]]; then
    echo "Cache path is unknown. Use --cache-path or set HF_HUB_CACHE/HF_HOME."
    exit 1
  fi
  if [[ "${CACHE_PATH}" == "/" ]]; then
    echo "Refusing to delete '/'."
    exit 1
  fi
  echo "Deleting cache directory: ${CACHE_PATH}"
  rm -rf "${CACHE_PATH}"
fi

echo "Cleanup done."
