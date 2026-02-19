#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/workspace/I2V-OSS-videoGen-pipeline}"
VENV_DIR="${VENV_DIR:-/workspace/.venv}"
ENV_FILE="${ENV_FILE:-/workspace/I2V_ENV.sh}"
REPO_URL="${REPO_URL:-}"
PREFETCH_WAN=0
PYTHON_BIN="${PYTHON_BIN:-python3}"
APT_UPDATED=0

usage() {
  cat <<'EOF'
Usage: bash scripts/vast/01_setup_env.sh [options]

Options:
  --repo-url <git-url>    Clone URL used when repo dir does not exist
  --repo-dir <path>       Repo path (default: /workspace/I2V-OSS-videoGen-pipeline)
  --venv-dir <path>       Venv path (default: /workspace/.venv)
  --prefetch-wan          Run WAN prefetch after setup
EOF
}

ensure_apt_package() {
  local pkg="$1"
  if dpkg -s "${pkg}" >/dev/null 2>&1; then
    echo "apt package present: ${pkg}"
    return 0
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    echo "ERROR: apt-get not available; cannot install missing package '${pkg}'." >&2
    exit 1
  fi
  if [[ "${APT_UPDATED}" -eq 0 ]]; then
    echo "Running apt-get update..."
    apt-get update -y
    APT_UPDATED=1
  fi
  echo "Installing apt package: ${pkg}"
  DEBIAN_FRONTEND=noninteractive apt-get install -y "${pkg}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-url)
      REPO_URL="${2:-}"
      shift 2
      ;;
    --repo-dir)
      REPO_DIR="${2:-}"
      shift 2
      ;;
    --venv-dir)
      VENV_DIR="${2:-}"
      shift 2
      ;;
    --prefetch-wan)
      PREFETCH_WAN=1
      shift
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

echo "== Setup Env =="
echo "Repo dir: ${REPO_DIR}"
echo "Venv dir: ${VENV_DIR}"

ensure_apt_package git
ensure_apt_package ffmpeg
ensure_apt_package tmux

mkdir -p /workspace
if [[ -d "${REPO_DIR}/.git" ]]; then
  echo "Reusing existing git repo at ${REPO_DIR}"
elif [[ -d "${REPO_DIR}" ]]; then
  echo "Reusing existing directory at ${REPO_DIR} (no .git detected)"
else
  if [[ -z "${REPO_URL}" ]]; then
    echo "ERROR: ${REPO_DIR} not found and --repo-url was not provided." >&2
    exit 1
  fi
  echo "Cloning repo from ${REPO_URL} -> ${REPO_DIR}"
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

cd "${REPO_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} not found." >&2
  exit 1
fi

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Creating venv at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "Reusing venv at ${VENV_DIR}"
fi

# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if ! python -m pip show ftfy >/dev/null 2>&1; then
  python -m pip install ftfy
fi
if ! python -m pip show imageio-ffmpeg >/dev/null 2>&1; then
  python -m pip install imageio-ffmpeg
fi

export HF_HOME="/workspace/hf_cache"
export HF_HUB_CACHE="/workspace/hf_cache"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}"

cat > "${ENV_FILE}" <<EOF
#!/usr/bin/env bash
export REPO_DIR="${REPO_DIR}"
export VENV_DIR="${VENV_DIR}"
export HF_HOME="/workspace/hf_cache"
export HF_HUB_CACHE="/workspace/hf_cache"
export PYTHONUNBUFFERED=1
if [[ -f "\${VENV_DIR}/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "\${VENV_DIR}/bin/activate"
fi
cd "\${REPO_DIR}"
EOF
chmod +x "${ENV_FILE}"

echo "Wrote ${ENV_FILE}"
echo "Use in future SSH sessions:"
echo "  source ${ENV_FILE}"

if [[ "${PREFETCH_WAN}" -eq 1 ]]; then
  echo "Running WAN prefetch..."
  python scripts/prefetch.py --models wan
fi

echo "PASS: Environment setup complete."
