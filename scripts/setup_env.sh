#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${CONDA_ENV:-fisheye_motion}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ "${USE_BASE:-0}" == "1" ]]; then
  conda activate base
else
  if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip git
  fi
  conda activate "${ENV_NAME}"
fi

python -m pip install --upgrade pip
python -m pip install --index-url "${TORCH_INDEX_URL}" torch torchvision
python -m pip install -r "${ROOT}/requirements.txt"
python -m pip install -e "${ROOT}"

python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required. Environment setup finished, but PyTorch cannot see the GPU.")
print("device", torch.cuda.get_device_name(0))
PY
