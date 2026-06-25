# Fisheye Motion

GPU deep-learning pipeline for PRJ4: moving-area abstraction and target tracking in fisheye video.

The main system uses RAFT optical flow, Ultralytics YOLO segmentation priors, DINOv2 semantic motion priors, boundary-aware cues, a supervised two-stream `FisheyeMotionNet`, uncertainty-aware test-time augmentation, SAM2.1 mask refinement, fisheye geometry maps, region-wise evaluation, and short-range target tracking. Traditional FrameDiff and Farneback methods are kept only as baselines.

## What Is Included

- Full source code under `src/fisheye_ai/`.
- Reproducible scripts under `scripts/`.
- Default experiment config at `configs/default.yaml`.
- Final Chinese report at `PRJ4_report_fisheye_motion_ai.md`.
- Packaging files: `pyproject.toml`, `requirements.txt`, and `environment.yml`.

Large files are intentionally not committed:

- Dataset: `homework2.zip` and extracted `data/homework2/`.
- Trained checkpoints: `checkpoints/*.pth`.
- Downloaded weights: `yolo26n-seg.pt`, RAFT cache, SAM2.1 checkpoint.
- Torch hub weights: DINOv2 semantic encoder cache.
- Generated experiment outputs under `outputs/`.
- Cloned SAM2 repository under `third_party/sam2/`.

## Hardware And Software Requirements

- Linux workstation.
- NVIDIA GPU. The project is designed for GPU execution and refuses to train or infer if `torch.cuda.is_available()` is false.
- NVIDIA driver new enough for a CUDA PyTorch wheel. CUDA 12.8 wheels are used by default.
- Conda or Miniconda.
- `git`, `wget`, and `unzip`.
- Network access for Python packages, RAFT weights, YOLO weights, and SAM2.1 weights.

The original run used an RTX 4060 Laptop GPU with 8 GB VRAM. The default resolution and batch size are selected to fit that class of GPU.

## Quick Start From A Fresh Clone

```bash
git clone https://github.com/<your-github-name>/fisheye_motion.git
cd fisheye_motion
bash scripts/setup_env.sh
conda activate fisheye_motion
```

`scripts/setup_env.sh` creates a conda environment named `fisheye_motion`, installs CUDA-enabled PyTorch/torchvision, installs all project dependencies, installs this project in editable mode, and checks that CUDA is visible.

Advanced environment options:

```bash
# Use a custom conda environment name.
CONDA_ENV=prj4_fisheye_ai bash scripts/setup_env.sh

# Use an already prepared base environment.
USE_BASE=1 bash scripts/setup_env.sh

# Use another official PyTorch CUDA wheel index.
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 bash scripts/setup_env.sh
```

## Prepare The Dataset

Place `homework2.zip` in the repository root, or keep it anywhere and pass its path:

```bash
bash scripts/prepare_data.sh homework2.zip
```

Expected extracted layout:

```text
data/homework2/
  ...
```

The script runs:

```bash
python -m fisheye_ai.validate_data --config configs/default.yaml
```

This verifies that current frame, previous frame, ground-truth mask, and calibration JSON files are correctly paired by sample ID.

## Download Deep Model Weights

```bash
bash scripts/download_weights.sh
```

This prepares all required deep modules:

- Torchvision RAFT large weights are downloaded through `torchvision.models.optical_flow.raft_large(weights=DEFAULT)`.
- Ultralytics downloads `yolo26n-seg.pt`.
- Torch hub downloads DINOv2 `dinov2_vits14_reg`.
- `facebookresearch/sam2` is cloned to `third_party/sam2`.
- `sam2.1_hiera_base_plus.pt` is downloaded to `third_party/sam2/checkpoints/`.

If a download fails, rerun the script after checking network access.

## Smoke Test

```bash
bash scripts/run_smoke_test.sh
```

The smoke test validates the data, caches RAFT/YOLO features for a small subset, runs a short overfit training test, and writes quick visual/metric outputs. It is the fastest way to confirm that the GPU, dataset, weights, and package imports are all working.

## Full Training And Evaluation

```bash
bash scripts/run_full_pipeline.sh
```

This runs the complete experiment sequence:

1. Dataset validation.
2. RAFT and YOLO feature extraction.
3. DINOv2 semantic saliency/change prior extraction and boundary prior extraction.
4. `FisheyeMotionNet` training with boundary-weighted supervision.
5. Baseline evaluation: FrameDiff, Farneback, RAFT-only, YOLO-only, and SAM2 prompted by FrameDiff.
6. Main and ablation evaluation: FisheyeMotionNet variants and full RAFT+YOLO+DINO+FisheyeMotionNet+SAM2 model.
7. Short-range tracking, overlays, region metrics, and report figures.

Key generated files:

```text
checkpoints/best_f1.pth
outputs/deep_features/*.npz
outputs/metrics.csv
outputs/metrics_summary.csv
outputs/visualizations/*_summary.png
outputs/predictions/*_full.png
outputs/figures/training_curve.png
outputs/figures/method_f1_bar.png
PRJ4_report_fisheye_motion_ai.md
```

## Reproduce The Submitted Result

From a clean clone:

```bash
bash scripts/setup_env.sh
conda activate fisheye_motion
bash scripts/prepare_data.sh /path/to/homework2.zip
bash scripts/download_weights.sh
bash scripts/run_smoke_test.sh
bash scripts/run_full_pipeline.sh
```

The submitted full model reached the following held-out summary in the original run:

```text
Full-RAFT-YOLO-FMN-SAM2
IoU       0.3076
Precision 0.3541
Recall    0.5843
F1        0.3911
Center F1 0.4011
Middle F1 0.3413
Edge F1   0.2897
```

See `PRJ4_report_fisheye_motion_ai.md` for the full experimental discussion, ablations, failure cases, and region-wise analysis.

The newer research-oriented branch adds DINOv2 semantic priors, an edge prior, TTA uncertainty, and an uncertainty-weighted fusion stage. Its final method name in `metrics_summary.csv` is:

```text
Full-RAFT-YOLO-DINO-FMN-SAM2
```

Latest optimized GPU run:

```text
IoU       0.3131
Precision 0.3613
Recall    0.5830
F1        0.3932
Center F1 0.4012
Middle F1 0.3471
Edge F1   0.2915
```

## Configuration

Edit `configs/default.yaml` for common changes:

- `image_size`: default model input size.
- `batch_size`: lower this to `1` if GPU memory is tight.
- `epochs`: training length.
- `data_root`: extracted dataset directory.
- `output_root`: generated result directory.
- `checkpoint_dir`: model checkpoint directory.
- `dino.enabled` and `dino.model`: DINOv2 semantic prior extraction.
- `fusion`: weights for network, RAFT, YOLO, DINO, edge, and uncertainty penalty.
- `sam2.repo_dir` and `sam2.checkpoint`: SAM2 code and checkpoint locations.

Relative paths are resolved from the repository root. Absolute paths are also supported.

## Troubleshooting

`CUDA is required`:

Check `nvidia-smi`, driver installation, and that the installed PyTorch build is CUDA-enabled. The project deliberately does not fall back to CPU.

Out of memory:

Set `batch_size: 1` in `configs/default.yaml`. If necessary, reduce `image_size`, then rerun feature extraction and training.

YOLO or SAM2 download failure:

Rerun `bash scripts/download_weights.sh` after network access is restored. Existing files are reused.

Dataset validation failure:

Confirm that `homework2.zip` was extracted to `data/homework2/` and that the original folder structure was not modified.

SAM2 import failure:

Run `bash scripts/download_weights.sh` again. It installs the local SAM2 clone in editable mode after cloning.

## Repository Hygiene

The `.gitignore` file excludes datasets, checkpoints, generated outputs, downloaded weights, and third-party clones. This keeps the GitHub repository lightweight while preserving all code needed to recreate the full experiment.
