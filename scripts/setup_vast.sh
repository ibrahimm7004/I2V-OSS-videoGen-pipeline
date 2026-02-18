#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "Checking runtime dependencies..."
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || { echo "Missing ${PYTHON_BIN}"; exit 1; }
command -v ffmpeg >/dev/null 2>&1 || { echo "Missing ffmpeg"; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "Missing nvidia-smi"; exit 1; }

echo "GPU summary:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "Creating venv at ${VENV_DIR}"
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "Setup complete."
echo "Activate with: source ${VENV_DIR}/bin/activate"

