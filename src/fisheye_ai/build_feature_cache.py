from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .config import ensure_dirs, image_size, load_config
from .dataset import discover_samples, split_samples
from .deep_modules import RaftFlow, YoloObjectness
from .utils import atomic_npz, read_rgb, require_cuda, resize_img, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    require_cuda()
    set_seed(int(cfg["seed"]))
    size = image_size(cfg)
    samples = discover_samples(cfg["data_root"])
    splits = split_samples(samples, int(cfg["seed"]), cfg["splits"])
    chosen = samples if args.split == "all" else splits[args.split]
    if args.limit:
        chosen = chosen[: args.limit]
    cache = Path(cfg["feature_cache"])
    raft = RaftFlow(cfg["raft"].get("weights", "DEFAULT"))
    yolo = YoloObjectness(cfg["yolo"]["model"], int(cfg["yolo"]["imgsz"]), float(cfg["yolo"]["conf"]))
    for sample in tqdm(chosen, desc=f"cache {args.split}"):
        out = cache / f"{sample.sid}.npz"
        if out.exists():
            continue
        prev = resize_img(read_rgb(sample.previous), size, cv2.INTER_AREA)
        curr = resize_img(read_rgb(sample.current), size, cv2.INTER_AREA)
        flow = raft(prev, curr)
        objectness, boxes = yolo(curr)
        atomic_npz(out, flow=flow, yolo_objectness=objectness.astype(np.float32), boxes=np.asarray(boxes, dtype=np.float32))
    print(f"feature cache written to {cache}")


if __name__ == "__main__":
    main()
