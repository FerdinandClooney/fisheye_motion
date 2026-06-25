#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${CONDA_ENV:-fisheye_motion}"
if [[ "${USE_CURRENT_ENV:-0}" != "1" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
fi
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

python -m fisheye_ai.download_weights --config "${ROOT}/configs/default.yaml"
