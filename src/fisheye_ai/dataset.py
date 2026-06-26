from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import geometry_maps, position_maps, rectify_image
from .utils import read_mask, read_rgb, resize_img


@dataclass(frozen=True)
class Sample:
    sid: str
    current: Path
    previous: Path
    gt: Path
    calib: Path


def discover_samples(data_root: str | Path) -> list[Sample]:
    data_root = Path(data_root)
    rgb_dir = data_root / "rgb_images"
    prev_dir = data_root / "previous_images"
    gt_dir = data_root / "motion_annotation" / "GroudTruth"
    calib_dir = data_root / "calibration_data"
    samples: list[Sample] = []
    for current in sorted(rgb_dir.glob("*_FV.png")):
        sid = current.name.replace("_FV.png", "")
        prev = prev_dir / f"{sid}_FV_prev.png"
        gt = gt_dir / f"{sid}_FV.png"
        calib = calib_dir / f"{sid}_FV.json"
        if prev.exists() and gt.exists() and calib.exists():
            samples.append(Sample(sid, current, prev, gt, calib))
    if not samples:
        raise RuntimeError(f"No complete samples found under {data_root}")
    return samples


def split_samples(samples: list[Sample], seed: int, ratios: dict[str, float]) -> dict[str, list[Sample]]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    n = len(samples)
    n_train = int(round(n * ratios["train"]))
    n_val = int(round(n * ratios["val"]))
    train = [samples[i] for i in idx[:n_train]]
    val = [samples[i] for i in idx[n_train : n_train + n_val]]
    test = [samples[i] for i in idx[n_train + n_val :]]
    return {"train": train, "val": val, "test": test}


def load_calibration(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class FisheyeMotionDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        image_size: tuple[int, int],
        feature_cache: str | Path,
        require_deep_features: bool = True,
        use_raft: bool = True,
        use_yolo: bool = True,
        use_dino: bool = True,
        use_edge: bool = True,
        use_geometry: bool = True,
        processing_domain: str = "fisheye",
        rectified_fov_deg: float = 120.0,
    ) -> None:
        self.samples = samples
        self.image_size = image_size
        self.feature_cache = Path(feature_cache)
        self.require_deep_features = require_deep_features
        self.use_raft = use_raft
        self.use_yolo = use_yolo
        self.use_dino = use_dino
        self.use_edge = use_edge
        self.use_geometry = use_geometry
        self.processing_domain = processing_domain
        self.rectified_fov_deg = float(rectified_fov_deg)

    def __len__(self) -> int:
        return len(self.samples)

    def _feature_path(self, sid: str) -> Path:
        return self.feature_cache / f"{sid}.npz"

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[idx]
        h, w = self.image_size
        calib = load_calibration(sample.calib)
        prev_rgb = resize_img(read_rgb(sample.previous), self.image_size).astype(np.uint8)
        curr_rgb = resize_img(read_rgb(sample.current), self.image_size).astype(np.uint8)
        gt_mask = resize_img(read_mask(sample.gt), self.image_size, cv2.INTER_NEAREST).astype(np.uint8)
        if self.processing_domain == "rectified":
            prev_rgb = rectify_image(prev_rgb, calib, self.image_size, interpolation=cv2.INTER_LINEAR)
            curr_rgb = rectify_image(curr_rgb, calib, self.image_size, interpolation=cv2.INTER_LINEAR)
            gt_mask = rectify_image(gt_mask, calib, self.image_size, interpolation=cv2.INTER_NEAREST)
        prev = prev_rgb.astype(np.float32) / 255.0
        curr = curr_rgb.astype(np.float32) / 255.0
        gt = gt_mask.astype(np.float32)
        diff = np.mean(np.abs(curr - prev), axis=2, keepdims=True)

        chans = [prev, curr, diff]
        feature_path = self._feature_path(sample.sid)
        if feature_path.exists():
            data = np.load(feature_path)
            if self.use_raft:
                flow = data["flow"].astype(np.float32)
                if flow.shape[:2] != (h, w):
                    flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
                chans.append(flow)
            if self.use_yolo:
                yolo = data["yolo_objectness"].astype(np.float32)
                if yolo.shape[:2] != (h, w):
                    yolo = cv2.resize(yolo, (w, h), interpolation=cv2.INTER_LINEAR)
                chans.append(yolo[..., None])
            if self.use_dino:
                dino = data["dino_prior"].astype(np.float32) if "dino_prior" in data else np.zeros((h, w, 2), dtype=np.float32)
                if dino.shape[:2] != (h, w):
                    dino = cv2.resize(dino, (w, h), interpolation=cv2.INTER_LINEAR)
                chans.append(dino)
            if self.use_edge:
                edge = data["edge_prior"].astype(np.float32) if "edge_prior" in data else np.zeros((h, w), dtype=np.float32)
                if edge.shape[:2] != (h, w):
                    edge = cv2.resize(edge, (w, h), interpolation=cv2.INTER_LINEAR)
                chans.append(edge[..., None])
        elif self.require_deep_features:
            raise FileNotFoundError(f"Missing RAFT/YOLO feature cache: {feature_path}")
        else:
            if self.use_raft:
                chans.append(np.zeros((h, w, 4), dtype=np.float32))
            if self.use_yolo:
                chans.append(np.zeros((h, w, 1), dtype=np.float32))
            if self.use_dino:
                chans.append(np.zeros((h, w, 2), dtype=np.float32))
            if self.use_edge:
                chans.append(np.zeros((h, w, 1), dtype=np.float32))

        if self.use_geometry:
            geom = geometry_maps(calib, self.image_size) if self.processing_domain == "fisheye" else position_maps(self.image_size)
            chans.append(geom)

        x = np.concatenate(chans, axis=2).transpose(2, 0, 1)
        return {
            "id": sample.sid,
            "image": torch.from_numpy(x).float(),
            "mask": torch.from_numpy(gt[None]).float(),
        }
