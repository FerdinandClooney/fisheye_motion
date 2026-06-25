from __future__ import annotations

import numpy as np


def binary_stats(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    iou = tp / (tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1}


def radial_region_masks(r_map: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "all": np.ones_like(r_map, dtype=bool),
        "center": r_map < 0.35,
        "middle": (r_map >= 0.35) & (r_map < 0.70),
        "edge": r_map >= 0.70,
    }


def region_stats(pred: np.ndarray, gt: np.ndarray, r_map: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, mask in radial_region_masks(r_map).items():
        stats = binary_stats(pred[mask], gt[mask])
        for k, v in stats.items():
            out[f"{name}_{k}"] = v
    return out
