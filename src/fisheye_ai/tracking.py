from __future__ import annotations

import cv2
import numpy as np


def boxes_from_mask(mask: np.ndarray, min_area: int = 40) -> list[tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, x + w, y + h))
    return boxes


def propagate_boxes(boxes: list[tuple[int, int, int, int]], flow4: np.ndarray) -> list[tuple[int, int, int, int]]:
    flow = flow4[..., :2]
    out: list[tuple[int, int, int, int]] = []
    h, w = flow.shape[:2]
    for x1, y1, x2, y2 in boxes:
        xi1, yi1 = max(0, x1), max(0, y1)
        xi2, yi2 = min(w, x2), min(h, y2)
        if xi2 <= xi1 or yi2 <= yi1:
            out.append((x1, y1, x2, y2))
            continue
        disp = np.median(flow[yi1:yi2, xi1:xi2].reshape(-1, 2), axis=0)
        dx, dy = int(round(float(disp[0]))), int(round(float(disp[1])))
        out.append((x1 + dx, y1 + dy, x2 + dx, y2 + dy))
    return out


def draw_boxes(rgb: np.ndarray, boxes: list[tuple[int, int, int, int]], color: tuple[int, int, int]) -> np.ndarray:
    img = rgb.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, f"ID {i}", (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img
