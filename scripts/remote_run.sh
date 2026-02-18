#!/usr/bin/env bash
set -euo pipefail

JOB_PATH="${JOB_PATH:-jobs/example_wan.yaml}"
OUT_ROOT="${OUT_ROOT:-outputs}"
RUN_ID="${RUN_ID:-remote-$(date -u +%Y%m%dT%H%M%SZ)}"

STATUS_DIR="${OUT_ROOT}/${RUN_ID}/status"
mkdir -p "${STATUS_DIR}"

echo "Starting remote run"
echo "  job: ${JOB_PATH}"
echo "  out: ${OUT_ROOT}"
echo "  run_id: ${RUN_ID}"
echo "  stdout log: ${STATUS_DIR}/stdout.log"

python -m scripts.run_job --job "${JOB_PATH}" --out "${OUT_ROOT}" --run-id "${RUN_ID}" "$@" 2>&1 | tee "${STATUS_DIR}/stdout.log"
exit ${PIPESTATUS[0]}

