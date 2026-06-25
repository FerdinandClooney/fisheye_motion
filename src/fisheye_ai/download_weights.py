from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from .config import load_config
from .deep_modules import RaftFlow, Sam2Refiner, YoloObjectness
from .utils import require_cuda


def run(cmd: list[str], cwd: str | Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def run_with_env(cmd: list[str], env: dict[str, str], cwd: str | Path | None = None) -> None:
    print("+", " ".join(cmd))
    merged_env = os.environ.copy()
    merged_env.update(env)
    subprocess.run(cmd, cwd=cwd, env=merged_env, check=True)


def ensure_sam2(cfg: dict) -> None:
    repo = Path(cfg["sam2"]["repo_dir"])
    if not repo.exists():
        run(["git", "clone", "https://github.com/facebookresearch/sam2.git", str(repo)])
    run(
        [
            "python",
            "-m",
            "pip",
            "install",
            "hydra-core==1.3.2",
            "iopath",
            "portalocker",
        ]
    )
    run_with_env(
        ["python", "-m", "pip", "install", "--no-build-isolation", "--no-deps", "-e", str(repo)],
        {"SAM2_BUILD_CUDA": "0"},
    )
    ckpt = Path(cfg["sam2"]["checkpoint"])
    if not ckpt.exists():
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
        run(["wget", "-c", url, "-O", ckpt.name], cwd=ckpt.parent)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    require_cuda()
    ensure_sam2(cfg)
    print("Downloading/loading RAFT weights...")
    _ = RaftFlow(cfg["raft"].get("weights", "DEFAULT"))
    print("Downloading/loading YOLO weights...")
    _ = YoloObjectness(cfg["yolo"]["model"], cfg["yolo"]["imgsz"], cfg["yolo"]["conf"])
    print("Loading SAM2 checkpoint...")
    _ = Sam2Refiner(cfg["sam2"]["repo_dir"], cfg["sam2"]["checkpoint"], cfg["sam2"]["config"])
    print("All required deep weights are available.")


if __name__ == "__main__":
    main()
