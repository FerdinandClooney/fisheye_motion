from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .utils import normalize01, require_cuda


def _to_tensor_rgb(img: np.ndarray, device: torch.device) -> torch.Tensor:
    x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
    return x.to(device)


def _pad8(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    h, w = x.shape[-2:]
    ph = (8 - h % 8) % 8
    pw = (8 - w % 8) % 8
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
    return x, (ph, pw)


class RaftFlow:
    def __init__(self, weights_name: str = "DEFAULT") -> None:
        require_cuda()
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        weights = Raft_Large_Weights.DEFAULT if weights_name == "DEFAULT" else Raft_Large_Weights[weights_name]
        self.device = torch.device("cuda")
        self.transforms = weights.transforms()
        self.model = raft_large(weights=weights, progress=True).to(self.device).eval()

    @torch.inference_mode()
    def __call__(self, prev_rgb: np.ndarray, curr_rgb: np.ndarray) -> np.ndarray:
        p = _to_tensor_rgb(prev_rgb, self.device)
        c = _to_tensor_rgb(curr_rgb, self.device)
        p, pad = _pad8(p)
        c, _ = _pad8(c)
        p, c = self.transforms(p, c)
        flows = self.model(p, c)
        flow = flows[-1]
        ph, pw = pad
        if ph:
            flow = flow[..., :-ph, :]
        if pw:
            flow = flow[..., :-pw]
        flow_np = flow[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
        mag = np.sqrt(np.sum(flow_np * flow_np, axis=2))
        ang = np.arctan2(flow_np[..., 1], flow_np[..., 0])
        return np.dstack([flow_np, normalize01(mag), (ang + np.pi) / (2 * np.pi)]).astype(np.float32)


class YoloObjectness:
    def __init__(self, model_name: str, imgsz: int = 640, conf: float = 0.2) -> None:
        require_cuda()
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.imgsz = imgsz
        self.conf = conf

    @torch.inference_mode()
    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, list[list[float]]]:
        h, w = rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        boxes_out: list[list[float]] = []
        results = self.model.predict(rgb, imgsz=self.imgsz, conf=self.conf, device=0, verbose=False)
        if not results:
            return mask, boxes_out
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.detach().cpu().numpy()
            confs = result.boxes.conf.detach().cpu().numpy()
            for box, score in zip(boxes, confs):
                x1, y1, x2, y2 = box
                boxes_out.append([float(x1), float(y1), float(x2), float(y2), float(score)])
                xi1, yi1 = max(0, int(x1)), max(0, int(y1))
                xi2, yi2 = min(w, int(x2)), min(h, int(y2))
                mask[yi1:yi2, xi1:xi2] = np.maximum(mask[yi1:yi2, xi1:xi2], float(score) * 0.35)
        if getattr(result, "masks", None) is not None and result.masks is not None:
            masks = result.masks.data.detach().cpu().numpy()
            confs = result.boxes.conf.detach().cpu().numpy() if result.boxes is not None else np.ones(len(masks), dtype=np.float32)
            for m, score in zip(masks, confs):
                m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
                mask = np.maximum(mask, m * float(score))
        return np.clip(mask, 0.0, 1.0), boxes_out


class Sam2Refiner:
    def __init__(self, repo_dir: str | Path, checkpoint: str | Path, config: str, device: str = "cuda") -> None:
        require_cuda()
        import sys

        repo_dir = Path(repo_dir)
        if str(repo_dir) not in sys.path:
            sys.path.insert(0, str(repo_dir))
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.device = torch.device(device)
        model = build_sam2(config, str(checkpoint), device=self.device)
        self.predictor = SAM2ImagePredictor(model)

    @torch.inference_mode()
    def refine(self, rgb: np.ndarray, prob: np.ndarray, boxes: list[list[float]], threshold: float = 0.5) -> np.ndarray:
        h, w = prob.shape[:2]
        bin_mask = (prob >= threshold).astype(np.uint8)
        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        prompt_boxes: list[list[float]] = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 25:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            prompt_boxes.append([x, y, x + bw, y + bh])
        for box in boxes:
            prompt_boxes.append(box[:4])
        if not prompt_boxes:
            return bin_mask
        self.predictor.set_image(rgb)
        refined = np.zeros((h, w), dtype=np.uint8)
        for box in prompt_boxes[:16]:
            box_np = np.array(box, dtype=np.float32)
            masks, scores, _ = self.predictor.predict(box=box_np, multimask_output=True)
            best = int(np.argmax(scores))
            refined = np.logical_or(refined, masks[best]).astype(np.uint8)
        return refined


def load_boxes(feature_cache: str | Path, sid: str) -> list[list[float]]:
    path = Path(feature_cache) / f"{sid}.npz"
    if not path.exists():
        return []
    data = np.load(path, allow_pickle=True)
    if "boxes" not in data:
        return []
    return data["boxes"].astype(np.float32).tolist()
