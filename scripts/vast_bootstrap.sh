#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
REPO_ROOT="$(pwd)"
OUTPUT_DIR="${REPO_ROOT}/outputs"
BOOTSTRAP_INFO="${OUTPUT_DIR}/bootstrap_info.json"

command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "Missing ${PYTHON_BIN}"; exit 1; }

"${PYTHON_BIN}" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("Python >= 3.10 is required.")
print(f"Python version OK: {sys.version.split()[0]}")
PY

"${PYTHON_BIN}" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

FFMPEG_LOCAL="${REPO_ROOT}/ffmpeg/bin/ffmpeg"
FFPROBE_LOCAL="${REPO_ROOT}/ffmpeg/bin/ffprobe"

if [[ -x "${FFMPEG_LOCAL}" && -x "${FFPROBE_LOCAL}" ]]; then
  export FFMPEG_BIN="${FFMPEG_LOCAL}"
  export FFPROBE_BIN="${FFPROBE_LOCAL}"
  echo "Using repo-local ffmpeg: ${FFMPEG_BIN}"
elif command -v ffmpeg >/dev/null 2>&1 && command -v ffprobe >/dev/null 2>&1; then
  export FFMPEG_BIN="$(command -v ffmpeg)"
  export FFPROBE_BIN="$(command -v ffprobe)"
  echo "Using system ffmpeg: ${FFMPEG_BIN}"
  echo "Repo-local ffmpeg not found at ./ffmpeg/bin. If needed, place binaries there for pinned behavior."
else
  echo "FFmpeg not found."
  echo "Provide repo-local binaries at ./ffmpeg/bin/ffmpeg and ./ffmpeg/bin/ffprobe,"
  echo "or install system ffmpeg/ffprobe and retry."
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"
PY_VERSION="$(python -c 'import sys; print(sys.version.split()[0])')"
NVIDIA_SMI_TEXT="$(nvidia-smi 2>&1 || true)"
echo "${NVIDIA_SMI_TEXT}"

export BOOTSTRAP_INFO PY_VERSION NVIDIA_SMI_TEXT
python - <<'PY'
import json
import os
from datetime import datetime, timezone

payload = {
    "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    "python_version": os.environ.get("PY_VERSION"),
    "ffmpeg_bin": os.environ.get("FFMPEG_BIN"),
    "ffprobe_bin": os.environ.get("FFPROBE_BIN"),
    "nvidia_smi": os.environ.get("NVIDIA_SMI_TEXT"),
}
path = os.environ["BOOTSTRAP_INFO"]
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2, sort_keys=True)
print(f"Wrote {path}")
PY

echo "Bootstrap complete."
