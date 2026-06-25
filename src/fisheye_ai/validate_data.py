from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .dataset import discover_samples, load_calibration
from .utils import read_mask, read_rgb


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    samples = discover_samples(cfg["data_root"])
    first = samples[0]
    curr = read_rgb(first.current)
    prev = read_rgb(first.previous)
    gt = read_mask(first.gt)
    calib = load_calibration(first.calib)
    assert curr.shape == prev.shape, "current/previous frame shape mismatch"
    assert curr.shape[:2] == gt.shape[:2], "image/mask shape mismatch"
    assert calib["intrinsic"]["model"] == "radial_poly", "expected radial_poly fisheye model"
    print(f"valid samples: {len(samples)}")
    print(f"image shape: {curr.shape[1]}x{curr.shape[0]}")
    print(f"first sample: {first.sid}")
    print(f"data root: {Path(cfg['data_root']).resolve()}")


if __name__ == "__main__":
    main()
