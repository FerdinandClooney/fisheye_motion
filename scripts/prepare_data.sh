#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ZIP_PATH="${1:-${ROOT}/homework2.zip}"
DATA_ROOT="${ROOT}/data"
TARGET_DIR="${DATA_ROOT}/homework2"

if [[ -d "${TARGET_DIR}" ]]; then
  echo "Dataset already exists at ${TARGET_DIR}"
else
  if [[ ! -f "${ZIP_PATH}" ]]; then
    echo "Cannot find dataset zip: ${ZIP_PATH}" >&2
    echo "Place homework2.zip in the repository root or pass its path:" >&2
    echo "  bash scripts/prepare_data.sh /path/to/homework2.zip" >&2
    exit 1
  fi
  mkdir -p "${DATA_ROOT}"
  unzip -q "${ZIP_PATH}" -d "${DATA_ROOT}"
fi

python -m fisheye_ai.validate_data --config "${ROOT}/configs/default.yaml"
