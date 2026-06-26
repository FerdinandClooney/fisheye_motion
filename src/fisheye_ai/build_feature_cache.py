from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .config import apply_processing_domain, ensure_dirs, image_size, load_config
from .dataset import discover_samples, load_calibration, split_samples
from .deep_modules import DinoSemanticPrior, RaftFlow, YoloObjectness
from .geometry import rectify_image
from .utils import atomic_npz, normalize01, read_rgb, require_cuda, resize_img, set_seed


def boundary_prior(curr: np.ndarray, prev: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
    canny = cv2.Canny(gray, 60, 160).astype(np.float32) / 255.0
    diff = cv2.absdiff(gray, prev_gray).astype(np.float32) / 255.0
    sobel_x = cv2.Sobel(diff, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(diff, cv2.CV_32F, 0, 1, ksize=3)
    motion_edge = normalize01(np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y))
    return np.maximum(canny * 0.5, motion_edge).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--domain", default="fisheye", choices=["fisheye", "rectified"])
    parser.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    cfg = apply_processing_domain(load_config(args.config), args.domain)
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
    dino = DinoSemanticPrior(cfg.get("dino", {}).get("model", "dinov2_vits14_reg")) if cfg.get("dino", {}).get("enabled", True) else None
    for sample in tqdm(chosen, desc=f"cache {args.split}"):
        out = cache / f"{sample.sid}.npz"
        existing = dict(np.load(out, allow_pickle=True)) if out.exists() else {}
        prev = resize_img(read_rgb(sample.previous), size, cv2.INTER_AREA)
        curr = resize_img(read_rgb(sample.current), size, cv2.INTER_AREA)
        if args.domain == "rectified":
            calib = load_calibration(sample.calib)
            prev = rectify_image(prev, calib, size, interpolation=cv2.INTER_LINEAR)
            curr = rectify_image(curr, calib, size, interpolation=cv2.INTER_LINEAR)
        flow = existing.get("flow")
        objectness = existing.get("yolo_objectness")
        boxes = existing.get("boxes")
        if flow is None:
            flow = raft(prev, curr)
        if objectness is None or boxes is None:
            objectness, boxes = yolo(curr)
        dino_prior = existing.get("dino_prior")
        if dino is not None and dino_prior is None:
            dino_prior = dino(prev, curr)
        edge_prior = existing.get("edge_prior")
        if edge_prior is None:
            edge_prior = boundary_prior(curr, prev)
        arrays = {
            "flow": flow.astype(np.float32),
            "yolo_objectness": objectness.astype(np.float32),
            "boxes": np.asarray(boxes, dtype=np.float32),
            "edge_prior": edge_prior.astype(np.float32),
        }
        if dino_prior is not None:
            arrays["dino_prior"] = dino_prior.astype(np.float32)
        atomic_npz(out, **arrays)
    print(f"feature cache written to {cache}")


if __name__ == "__main__":
    main()
