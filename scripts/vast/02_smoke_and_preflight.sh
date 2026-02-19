#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_DIR}"

if [[ -f /workspace/I2V_ENV.sh ]]; then
  # shellcheck disable=SC1091
  source /workspace/I2V_ENV.sh
fi

DEFAULT_JOB="${WAN_JOB:-jobs/wan/idea01.yaml}"
if [[ ! -f "${DEFAULT_JOB}" && -f jobs/idea01_wan.yaml ]]; then
  DEFAULT_JOB="jobs/idea01_wan.yaml"
fi

JOB_PATH="${JOB_PATH:-${DEFAULT_JOB}}"
RUN_SMOKE=0
SKIP_VERIFY=0
SMOKE_NUM_CLIPS="${SMOKE_NUM_CLIPS:-1}"
SMOKE_DURATION_SEC="${SMOKE_DURATION_SEC:-2}"
SMOKE_STEPS="${SMOKE_STEPS:-12}"
SMOKE_MODEL_ID="${SMOKE_MODEL_ID:-}"

usage() {
  cat <<'EOF'
Usage: bash scripts/vast/02_smoke_and_preflight.sh [options]

Options:
  --job <path>               WAN job yaml path
  --skip-verify              Skip verify_jobpacks stage
  --run-smoke                Run a lightweight smoke job
  --smoke-num-clips <N>      Smoke clip count override (default: 1)
  --smoke-duration-sec <S>   Smoke duration per clip (default: 2)
  --smoke-steps <N>          Smoke steps per clip (default: 12)
  --smoke-model-id <id>      Optional model override for smoke run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job)
      JOB_PATH="${2:-}"
      shift 2
      ;;
    --skip-verify)
      SKIP_VERIFY=1
      shift
      ;;
    --run-smoke)
      RUN_SMOKE=1
      shift
      ;;
    --smoke-num-clips)
      SMOKE_NUM_CLIPS="${2:-}"
      shift 2
      ;;
    --smoke-duration-sec)
      SMOKE_DURATION_SEC="${2:-}"
      shift 2
      ;;
    --smoke-steps)
      SMOKE_STEPS="${2:-}"
      shift 2
      ;;
    --smoke-model-id)
      SMOKE_MODEL_ID="${2:-}"
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

if [[ ! -f "${JOB_PATH}" ]]; then
  echo "ERROR: Job file not found: ${JOB_PATH}" >&2
  exit 1
fi

echo "== Smoke and Preflight =="
echo "Job: ${JOB_PATH}"

if [[ "${SKIP_VERIFY}" -eq 0 ]]; then
  echo "Running verify_jobpacks with relaxed POST_CLIP_MIN_FRAME_DIFF..."
  POST_CLIP_MIN_FRAME_DIFF=0.0001 python scripts/verify_jobpacks.py --jobs "${JOB_PATH}"
fi

echo "Running WAN encoder preflight..."
python - <<'PY'
import importlib.util
import json

from pipeline.ffmpeg_utils import assert_ffmpeg_working

imageio_ffmpeg_ok = importlib.util.find_spec("imageio_ffmpeg") is not None
if not imageio_ffmpeg_ok:
    raise SystemExit("ERROR: imageio_ffmpeg import failed. Install with: pip install imageio-ffmpeg")

info = assert_ffmpeg_working()
report = {
    "imageio_ffmpeg_available": imageio_ffmpeg_ok,
    "ffmpeg_path": info.get("ffmpeg_path"),
    "ffprobe_path": info.get("ffprobe_path"),
    "ffmpeg_version": info.get("ffmpeg_version"),
    "ffprobe_version": info.get("ffprobe_version"),
}
print(json.dumps(report, indent=2, sort_keys=True))
PY

if [[ "${RUN_SMOKE}" -eq 1 ]]; then
  echo "Running lightweight smoke job..."
  mkdir -p outputs/_vast_smoke_runtime

  SMOKE_RUNTIME_JOB="$(
    JOB_PATH="${JOB_PATH}" \
    SMOKE_NUM_CLIPS="${SMOKE_NUM_CLIPS}" \
    SMOKE_DURATION_SEC="${SMOKE_DURATION_SEC}" \
    SMOKE_STEPS="${SMOKE_STEPS}" \
    SMOKE_MODEL_ID="${SMOKE_MODEL_ID}" \
    python - <<'PY'
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import yaml

from scripts._job_loading import load_runtime_job_payload

job_path = Path(os.environ["JOB_PATH"]).resolve()
num_clips = int(os.environ["SMOKE_NUM_CLIPS"])
duration = float(os.environ["SMOKE_DURATION_SEC"])
steps = int(os.environ["SMOKE_STEPS"])
model_override = os.environ.get("SMOKE_MODEL_ID") or None
run_id = "wan-smoke-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

runtime, _, _ = load_runtime_job_payload(
    job_path,
    run_id_override=run_id,
    output_root_override="outputs",
    model_id_override=model_override,
    dry_run_override=False,
    fast_mode=False,
)

shots = list(runtime.get("shots", []))[: max(1, num_clips)]
for shot in shots:
    shot["duration_seconds"] = duration
    shot["steps"] = min(int(shot.get("steps", steps)), steps)
    fps = int(shot.get("fps", 8))
    shot["frames"] = max(1, int(round(duration * fps)))
runtime["shots"] = shots
runtime["run_id"] = run_id
runtime["job_name"] = run_id

out_path = Path("outputs/_vast_smoke_runtime") / f"{run_id}.yaml"
out_path.write_text(yaml.safe_dump(runtime, sort_keys=False), encoding="utf-8")
print(out_path.as_posix())
PY
  )"

  echo "Smoke runtime job: ${SMOKE_RUNTIME_JOB}"
  python scripts/run_job.py --job "${SMOKE_RUNTIME_JOB}" --out outputs
fi

echo "PASS: Preflight checks complete."
