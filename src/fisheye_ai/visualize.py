from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def flow_to_rgb(flow4: np.ndarray) -> np.ndarray:
    flow = flow4[..., :2]
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX), 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (0, 255, 0), alpha: float = 0.45) -> np.ndarray:
    out = rgb.copy()
    color_img = np.zeros_like(out)
    color_img[:] = color
    m = mask.astype(bool)
    out[m] = (out[m] * (1 - alpha) + color_img[m] * alpha).astype(np.uint8)
    return out


def error_map(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    out = np.zeros((*pred.shape, 3), dtype=np.uint8)
    out[np.logical_and(pred, gt)] = (0, 255, 0)
    out[np.logical_and(pred, ~gt)] = (255, 0, 0)
    out[np.logical_and(~pred, gt)] = (0, 80, 255)
    return out


def save_grid(path: str | Path, panels: list[tuple[str, np.ndarray]], cell_w: int = 320) -> None:
    imgs = []
    for title, img in panels:
        if img.ndim == 2:
            img = cv2.cvtColor((img * 255 if img.max() <= 1 else img).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        scale = cell_w / img.shape[1]
        resized = cv2.resize(img, (cell_w, int(round(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        canvas = np.full((resized.shape[0] + 28, cell_w, 3), 255, dtype=np.uint8)
        canvas[28:] = resized
        cv2.putText(canvas, title, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        imgs.append(canvas)
    cols = 4
    rows = int(np.ceil(len(imgs) / cols))
    h = max(i.shape[0] for i in imgs)
    grid = np.full((rows * h, cols * cell_w, 3), 255, dtype=np.uint8)
    for i, img in enumerate(imgs):
        r, c = divmod(i, cols)
        grid[r * h : r * h + img.shape[0], c * cell_w : (c + 1) * cell_w] = img
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
