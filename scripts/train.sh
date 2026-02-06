#!/usr/bin/env bash
set -euo pipefail

echo "--- GenRec-E Training Script Started ---"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GENREC_ROOT="${PROJECT_ROOT}/GenRec"

PYTHON_EXECUTABLE="${PYTHON_EXECUTABLE:-python3}"
CONFIG_FILE_PATH="${CONFIG_FILE_PATH:-${GENREC_ROOT}/config/genrec_e_movielens_config.json}"
PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-}"

if ! command -v "${PYTHON_EXECUTABLE}" >/dev/null 2>&1; then
    echo "!!! CRITICAL ERROR: Python executable not found: ${PYTHON_EXECUTABLE}"
    exit 1
fi

if [ ! -f "${CONFIG_FILE_PATH}" ]; then
    echo "!!! CRITICAL ERROR: Config file not found: ${CONFIG_FILE_PATH}"
    exit 1
fi

export PYTHONPATH="${GENREC_ROOT}:${PYTHONPATH:-}"
cd "${GENREC_ROOT}"

echo "Using Python interpreter: ${PYTHON_EXECUTABLE}"
echo "Set PYTHONPATH to: ${PYTHONPATH}"
echo "Current directory: $(pwd)"
echo "Starting training using config file: ${CONFIG_FILE_PATH}"

CMD=("${PYTHON_EXECUTABLE}" "genrec/train.py" "-c" "${CONFIG_FILE_PATH}")
if [ -n "${PRETRAINED_MODEL_PATH}" ]; then
    CMD+=("-pmp" "${PRETRAINED_MODEL_PATH}")
    echo "Using pretrained model: ${PRETRAINED_MODEL_PATH}"
fi

"${CMD[@]}"

echo "--- Training finished successfully. ---"
