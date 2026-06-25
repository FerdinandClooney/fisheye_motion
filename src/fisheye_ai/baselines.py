from __future__ import annotations

import cv2
import numpy as np

from .utils import normalize01


def clean_mask(mask: np.ndarray, k: int = 5) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask.astype(np.uint8)


def frame_diff(prev_rgb: np.ndarray, curr_rgb: np.ndarray) -> np.ndarray:
    diff = cv2.absdiff(prev_rgb, curr_rgb)
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return clean_mask(mask)


def farneback(prev_rgb: np.ndarray, curr_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = cv2.cvtColor(prev_rgb, cv2.COLOR_RGB2GRAY)
    c = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(p, c, None, 0.5, 3, 25, 3, 5, 1.2, 0)
    mag = np.sqrt(np.sum(flow * flow, axis=2))
    score = normalize01(mag)
    mask = (score > max(0.15, float(np.percentile(score, 90)))).astype(np.uint8)
    return clean_mask(mask), flow


def raft_only(flow4: np.ndarray) -> np.ndarray:
    mag = flow4[..., 2]
    th = max(0.15, float(np.percentile(mag, 90)))
    return clean_mask((mag >= th).astype(np.uint8))


def yolo_only(objectness: np.ndarray) -> np.ndarray:
    return clean_mask((objectness > 0.15).astype(np.uint8))
