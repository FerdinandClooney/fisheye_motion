from __future__ import annotations

import cv2
import numpy as np


def principal_point(intrinsic: dict) -> tuple[float, float]:
    w = float(intrinsic["width"])
    h = float(intrinsic["height"])
    return w * 0.5 + float(intrinsic.get("cx_offset", 0.0)), h * 0.5 + float(intrinsic.get("cy_offset", 0.0))


def geometry_maps(calib: dict, size_hw: tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    intr = calib["intrinsic"]
    cx0, cy0 = principal_point(intr)
    sx = w / float(intr["width"])
    sy = h / float(intr["height"])
    cx = cx0 * sx
    cy = cy0 * sy
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    r = np.sqrt(dx * dx + dy * dy)
    r_norm = r / (np.sqrt(max(cx, w - cx) ** 2 + max(cy, h - cy) ** 2) + 1e-6)
    theta = np.arctan2(dy, dx)
    return np.stack([np.clip(r_norm, 0.0, 1.0), np.sin(theta), np.cos(theta)], axis=2).astype(np.float32)


def position_maps(size_hw: tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    cx = w * 0.5
    cy = h * 0.5
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = xx - cx
    dy = yy - cy
    r = np.sqrt(dx * dx + dy * dy)
    r_norm = r / (np.sqrt(cx * cx + cy * cy) + 1e-6)
    theta = np.arctan2(dy, dx)
    return np.stack([np.clip(r_norm, 0.0, 1.0), np.sin(theta), np.cos(theta)], axis=2).astype(np.float32)


def radial_poly_rho(theta: np.ndarray, intr: dict) -> np.ndarray:
    return (
        float(intr["k1"]) * theta
        + float(intr["k2"]) * theta**2
        + float(intr["k3"]) * theta**3
        + float(intr["k4"]) * theta**4
    )


def rectification_maps(calib: dict, out_size_hw: tuple[int, int], fov_deg: float = 120.0) -> tuple[np.ndarray, np.ndarray]:
    h, w = out_size_hw
    intr = calib["intrinsic"]
    cx, cy = principal_point(intr)
    f = (w * 0.5) / np.tan(np.deg2rad(fov_deg) * 0.5)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x = (xx - w * 0.5) / f
    y = (yy - h * 0.5) / f
    z = np.ones_like(x)
    norm_xy = np.sqrt(x * x + y * y)
    theta = np.arctan2(norm_xy, z)
    phi = np.arctan2(y, x)
    rho = radial_poly_rho(theta, intr)
    map_x = cx + rho * np.cos(phi)
    map_y = cy + rho * np.sin(phi)
    return map_x.astype(np.float32), map_y.astype(np.float32)


def rectify_image(img: np.ndarray, calib: dict, out_size_hw: tuple[int, int], interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
    map_x, map_y = rectification_maps(calib, out_size_hw)
    return cv2.remap(img, map_x, map_y, interpolation=interpolation, borderMode=cv2.BORDER_CONSTANT)


def project_rectified_array_to_fisheye(
    arr: np.ndarray,
    calib: dict,
    fisheye_size_hw: tuple[int, int],
    fov_deg: float = 120.0,
) -> np.ndarray:
    rect_h, rect_w = arr.shape[:2]
    fish_h, fish_w = fisheye_size_hw
    map_x, map_y = rectification_maps(calib, (rect_h, rect_w), fov_deg=fov_deg)
    xi = np.rint(map_x).astype(np.int32)
    yi = np.rint(map_y).astype(np.int32)
    valid = (xi >= 0) & (xi < fish_w) & (yi >= 0) & (yi < fish_h)
    flat_idx = yi[valid] * fish_w + xi[valid]

    if arr.ndim == 2:
        out = np.zeros((fish_h, fish_w), dtype=np.float32)
        np.maximum.at(out.reshape(-1), flat_idx, arr[valid].astype(np.float32))
        return out

    ch = arr.shape[2]
    out = np.zeros((fish_h, fish_w, ch), dtype=np.float32)
    flat_out = out.reshape(-1, ch)
    vals = arr[valid].astype(np.float32).reshape(-1, ch)
    for c in range(ch):
        np.maximum.at(flat_out[:, c], flat_idx, vals[:, c])
    return out


def project_rectified_mask_to_fisheye(
    mask: np.ndarray,
    calib: dict,
    fisheye_size_hw: tuple[int, int],
    fov_deg: float = 120.0,
) -> np.ndarray:
    score = project_rectified_array_to_fisheye(mask.astype(np.float32), calib, fisheye_size_hw, fov_deg=fov_deg)
    out = (score > 0.0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=2)
    return out.astype(np.uint8)
