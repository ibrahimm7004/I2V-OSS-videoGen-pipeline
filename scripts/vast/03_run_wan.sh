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

WAN_JOB_PATH="${DEFAULT_JOB}"
RUN_ID="${RUN_ID:-wan-$(date -u +%Y%m%dT%H%M%SZ)}"
TMUX_SESSION_DEFAULT="wan_run_${RUN_ID}"
TMUX_SESSION="${TMUX_SESSION:-${TMUX_SESSION_DEFAULT}}"
SESSION_EXPLICIT=0
if [[ "${TMUX_SESSION}" != "${TMUX_SESSION_DEFAULT}" ]]; then
  SESSION_EXPLICIT=1
fi
NUM_CLIPS_OVERRIDE=""
CLIP_DURATION_OVERRIDE=""

export WAN22_REPO_ID="${WAN22_REPO_ID:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
export WAN22_EXPORT_QUALITY="${WAN22_EXPORT_QUALITY:-9}"
export POST_CLIP_VALIDATION_ENABLED="${POST_CLIP_VALIDATION_ENABLED:-true}"
export POST_CLIP_MIN_FRAME_DIFF="${POST_CLIP_MIN_FRAME_DIFF:-0.003}"

usage() {
  cat <<'EOF'
Usage: bash scripts/vast/03_run_wan.sh [options]

Options:
  --job <path>               WAN job yaml path
  --run-id <id>              Explicit run id (default: wan-<utc timestamp>)
  --session <name>           tmux session name
  --num-clips <N>            Optional runtime override
  --clip-duration-sec <S>    Optional runtime override
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job)
      WAN_JOB_PATH="${2:-}"
      shift 2
      ;;
    --run-id)
      RUN_ID="${2:-}"
      if [[ "${SESSION_EXPLICIT}" -eq 0 ]]; then
        TMUX_SESSION="wan_run_${RUN_ID}"
      fi
      shift 2
      ;;
    --session)
      TMUX_SESSION="${2:-}"
      SESSION_EXPLICIT=1
      shift 2
      ;;
    --num-clips)
      NUM_CLIPS_OVERRIDE="${2:-}"
      shift 2
      ;;
    --clip-duration-sec)
      CLIP_DURATION_OVERRIDE="${2:-}"
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

if [[ ! -f "${WAN_JOB_PATH}" ]]; then
  echo "ERROR: WAN job not found: ${WAN_JOB_PATH}" >&2
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux is required. Install with apt-get install tmux." >&2
  exit 1
fi

if tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  echo "ERROR: tmux session already exists: ${TMUX_SESSION}" >&2
  echo "Use a different --session or stop it with: tmux kill-session -t ${TMUX_SESSION}" >&2
  exit 1
fi

mkdir -p outputs/_vast_runtime_jobs
RUNTIME_JOB_PATH="$(
  WAN_JOB_PATH="${WAN_JOB_PATH}" \
  RUN_ID="${RUN_ID}" \
  NUM_CLIPS_OVERRIDE="${NUM_CLIPS_OVERRIDE}" \
  CLIP_DURATION_OVERRIDE="${CLIP_DURATION_OVERRIDE}" \
  python - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

import yaml

from scripts._job_loading import load_runtime_job_payload

job_path = Path(os.environ["WAN_JOB_PATH"]).resolve()
run_id = os.environ["RUN_ID"]
num_clips_raw = os.environ.get("NUM_CLIPS_OVERRIDE", "").strip()
clip_duration_raw = os.environ.get("CLIP_DURATION_OVERRIDE", "").strip()

runtime, _, _ = load_runtime_job_payload(
    job_path,
    run_id_override=run_id,
    output_root_override="outputs",
    model_id_override=None,
    dry_run_override=False,
    fast_mode=False,
)

shots = list(runtime.get("shots", []))
if num_clips_raw:
    num_clips = max(1, int(num_clips_raw))
    shots = shots[:num_clips]
if clip_duration_raw:
    clip_duration = float(clip_duration_raw)
    for shot in shots:
        shot["duration_seconds"] = clip_duration
        fps = int(shot.get("fps", 8))
        shot["frames"] = max(1, int(round(clip_duration * fps)))

runtime["shots"] = shots
runtime["run_id"] = run_id
runtime["job_name"] = run_id

out_path = Path("outputs/_vast_runtime_jobs") / f"{run_id}.yaml"
out_path.write_text(yaml.safe_dump(runtime, sort_keys=False), encoding="utf-8")
print(out_path.as_posix())
PY
)"

LAUNCH_SCRIPT="outputs/_vast_runtime_jobs/${RUN_ID}_launch.sh"
cat > "${LAUNCH_SCRIPT}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_DIR}"
if [[ -f /workspace/I2V_ENV.sh ]]; then
  # shellcheck disable=SC1091
  source /workspace/I2V_ENV.sh
elif [[ -f /workspace/.venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source /workspace/.venv/bin/activate
fi
export WAN22_REPO_ID="${WAN22_REPO_ID}"
export WAN22_EXPORT_QUALITY="${WAN22_EXPORT_QUALITY}"
export POST_CLIP_VALIDATION_ENABLED="${POST_CLIP_VALIDATION_ENABLED}"
export POST_CLIP_MIN_FRAME_DIFF="${POST_CLIP_MIN_FRAME_DIFF}"
python scripts/run_job.py --job "${RUNTIME_JOB_PATH}" --out outputs --run-id "${RUN_ID}" 2>&1 | tee -a outputs/wanrun_live.log
EOF
chmod +x "${LAUNCH_SCRIPT}"

echo "Starting WAN run in tmux session: ${TMUX_SESSION}"
tmux new-session -d -s "${TMUX_SESSION}" "bash \"${LAUNCH_SCRIPT}\""

echo
echo "Run started."
echo "  run_id: ${RUN_ID}"
echo "  runtime job: ${RUNTIME_JOB_PATH}"
echo "  live log: ${REPO_DIR}/outputs/wanrun_live.log"
echo "  attach: tmux attach -t ${TMUX_SESSION}"
echo "  tail log: tail -f outputs/wanrun_live.log"
echo
echo "Artifacts will be under: ${REPO_DIR}/outputs/${RUN_ID}/"
echo "Download template from local PC:"
echo "  scp -i <KEY_PATH> root@<INSTANCE_IP>:/workspace/I2V-OSS-videoGen-pipeline/outputs/${RUN_ID}/${RUN_ID}_bundle.zip <LOCAL_DIR>/"
