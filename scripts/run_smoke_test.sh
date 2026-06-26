#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${CONDA_ENV:-fisheye_motion}"
if [[ "${USE_CURRENT_ENV:-0}" != "1" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${ENV_NAME}"
fi
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

python -m fisheye_ai.validate_data --config "${ROOT}/configs/default.yaml"
python -m fisheye_ai.build_feature_cache --config "${ROOT}/configs/default.yaml" --domain fisheye --split train --limit 4
python -m fisheye_ai.build_feature_cache --config "${ROOT}/configs/default.yaml" --domain fisheye --split test --limit 8
python -m fisheye_ai.train --config "${ROOT}/configs/default.yaml" --domain fisheye --epochs 2 --limit 4 --overfit
python -m fisheye_ai.run_experiments --config "${ROOT}/configs/default.yaml" --domain fisheye --checkpoint "${ROOT}/checkpoints/best_f1.pth" --limit 8
python -m fisheye_ai.build_feature_cache --config "${ROOT}/configs/default.yaml" --domain rectified --split train --limit 4
python -m fisheye_ai.build_feature_cache --config "${ROOT}/configs/default.yaml" --domain rectified --split test --limit 8
python -m fisheye_ai.train --config "${ROOT}/configs/default.yaml" --domain rectified --epochs 2 --limit 4 --overfit
python -m fisheye_ai.run_experiments --config "${ROOT}/configs/default.yaml" --domain rectified --checkpoint "${ROOT}/checkpoints_rectified/best_f1.pth" --limit 8
python -m fisheye_ai.report_assets --config "${ROOT}/configs/default.yaml"
python -m fisheye_ai.compare_routes --config "${ROOT}/configs/default.yaml"
