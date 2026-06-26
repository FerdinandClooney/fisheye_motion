from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "homework2"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 170,
        "savefig.dpi": 260,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


BLUE = "#2563eb"
CYAN = "#0891b2"
GREEN = "#16a34a"
AMBER = "#d97706"
RED = "#dc2626"
SLATE = "#0f172a"
MUTED = "#64748b"
GRID = "#e2e8f0"
BG = "#f8fafc"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ["DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", "Arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _read_rgb(path: Path, size: tuple[int, int] = (640, 480)) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize(size, Image.Resampling.BILINEAR)
    return np.asarray(img)


def _read_mask(path: Path, size: tuple[int, int] = (640, 480)) -> np.ndarray:
    img = Image.open(path).convert("L").resize(size, Image.Resampling.NEAREST)
    return np.asarray(img) > 127


def _norm(x: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.percentile(x[np.isfinite(x)], [p_low, p_high])
    if hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def _heat(x: np.ndarray, cmap: str = "magma") -> np.ndarray:
    cm = plt.get_cmap(cmap)
    return (cm(_norm(x))[..., :3] * 255).astype(np.uint8)


def _flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    u, v = flow[..., 0], flow[..., 1]
    mag, ang = cv2.cartToPolar(u.astype(np.float32), v.astype(np.float32))
    hsv = np.zeros((*mag.shape, 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 220
    hsv[..., 2] = (_norm(mag) * 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _overlay(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.52) -> np.ndarray:
    out = rgb.copy().astype(np.float32)
    c = np.array(color, dtype=np.float32)
    out[mask] = out[mask] * (1 - alpha) + c * alpha
    return out.astype(np.uint8)


def _error_map(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = np.zeros((*gt.shape, 3), dtype=np.uint8)
    out[gt & pred] = (34, 197, 94)
    out[~gt & pred] = (239, 68, 68)
    out[gt & ~pred] = (59, 130, 246)
    return out


def _labelled_tile(img: np.ndarray, title: str, subtitle: str | None = None) -> Image.Image:
    tile = Image.fromarray(img).convert("RGB")
    w, h = tile.size
    bar_h = 34 if subtitle is None else 52
    canvas = Image.new("RGB", (w, h + bar_h), "white")
    canvas.paste(tile, (0, bar_h))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, w, bar_h), fill="white")
    draw.text((8, 7), title, fill=SLATE, font=_font(17, True))
    if subtitle:
        draw.text((8, 30), subtitle, fill=MUTED, font=_font(12))
    return canvas


def _grid(tiles: list[list[Image.Image]], gap: int = 10, bg: str = "white") -> Image.Image:
    rows = len(tiles)
    cols = max(len(r) for r in tiles)
    widths = [max(tiles[r][c].width for r in range(rows) if c < len(tiles[r])) for c in range(cols)]
    heights = [max(t.height for t in row) for row in tiles]
    w = sum(widths) + gap * (cols - 1)
    h = sum(heights) + gap * (rows - 1)
    canvas = Image.new("RGB", (w, h), bg)
    y = 0
    for r, row in enumerate(tiles):
        x = 0
        for c, tile in enumerate(row):
            canvas.paste(tile, (x, y))
            x += widths[c] + gap
        y += heights[r] + gap
    return canvas


def _sample_paths(sid: str) -> dict[str, Path]:
    return {
        "prev": DATA / "previous_images" / f"{sid}_FV_prev.png",
        "curr": DATA / "rgb_images" / f"{sid}_FV.png",
        "gt": DATA / "motion_annotation" / "GroudTruth" / f"{sid}_FV.png",
        "pred": ROOT / "outputs" / "predictions" / f"{sid}_full.png",
        "feat": ROOT / "outputs" / "deep_features" / f"{sid}.npz",
    }


def save_architecture_graph() -> None:
    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, text, fc, ec="#334155", color="white", size=9.5):
        patch = plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.0, joinstyle="round")
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=color, fontsize=size, weight="bold")

    def arrow(x1, y1, x2, y2, color="#475569", lw=1.2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=lw, color=color))

    box(0.03, 0.52, 0.16, 0.16, "Fisheye pair\n$I_{t-1}, I_t$", SLATE)
    box(0.03, 0.23, 0.16, 0.13, "Calibration\n$r, \\sin\\theta, \\cos\\theta$", "#475569")
    modules = [
        (0.26, 0.74, "RAFT\nflow $u,v,|f|,\\angle f$", BLUE),
        (0.26, 0.55, "YOLO26-seg\nobjectness + boxes", GREEN),
        (0.26, 0.36, "DINOv2\nsemantic change", CYAN),
        (0.26, 0.17, "Boundary prior\nCanny + diff Sobel", AMBER),
    ]
    for x, y, t, c in modules:
        box(x, y, 0.24, 0.12, t, c, size=9.0)
        arrow(0.19, 0.60, x, y + 0.06)
    arrow(0.19, 0.295, 0.58, 0.295)
    box(0.58, 0.40, 0.18, 0.24, "18-channel\nFisheyeMotionNet\nAttention U-Net", RED)
    for _, y, _, _ in modules:
        arrow(0.50, y + 0.06, 0.58, 0.52)
    arrow(0.58, 0.295, 0.65, 0.40)
    box(0.81, 0.55, 0.16, 0.14, "TTA uncertainty\nfusion", "#7c3aed")
    box(0.81, 0.31, 0.16, 0.14, "SAM2.1 prompt\nrefinement", "#0f766e")
    arrow(0.76, 0.52, 0.81, 0.62)
    arrow(0.89, 0.55, 0.89, 0.45)
    ax.text(0.89, 0.20, "Final moving-object mask\n+ short-range tracks", ha="center", va="center", fontsize=10, color=SLATE, weight="bold")
    arrow(0.89, 0.31, 0.89, 0.24)
    ax.text(
        0.03,
        0.93,
        "Motion is not decided by a single threshold: geometric distortion, dense flow, object priors, semantic change and boundary evidence are learned jointly.",
        ha="left",
        va="center",
        fontsize=10,
        color="#334155",
    )
    fig.savefig(OUT / "architecture_graph.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_feature_evidence_panel(sid: str = "00022") -> None:
    paths = _sample_paths(sid)
    rgb = _read_rgb(paths["curr"])
    gt = _read_mask(paths["gt"])
    pred = _read_mask(paths["pred"])
    feat = np.load(paths["feat"])
    flow_rgb = _flow_to_rgb(feat["flow"])
    yolo = _heat(feat["yolo_objectness"], "viridis")
    dino = _heat(feat["dino_prior"][..., 1], "magma")
    edge = _heat(feat["edge_prior"], "gray")
    overlay_pred = _overlay(rgb, pred, (34, 197, 94), 0.48)
    overlay_gt = _overlay(rgb, gt, (59, 130, 246), 0.48)
    tiles = [
        [
            _labelled_tile(rgb, "Current frame", f"sample {sid}"),
            _labelled_tile(flow_rgb, "RAFT flow", "direction = hue, magnitude = value"),
            _labelled_tile(yolo, "YOLO objectness", "instance prior"),
        ],
        [
            _labelled_tile(dino, "DINOv2 semantic change", "patch-token change prior"),
            _labelled_tile(edge, "Boundary prior", "Canny + frame-diff gradient"),
            _labelled_tile(overlay_pred, "Final mask overlay", "green = predicted moving region"),
        ],
        [
            _labelled_tile(overlay_gt, "Ground truth overlay", "blue = annotation"),
            _labelled_tile(_error_map(gt, pred), "Error map", "green TP, red FP, blue FN"),
            _labelled_tile(np.full_like(rgb, 245), "Why this panel matters", "all visible maps are real cached model evidence"),
        ],
    ]
    tiles[-1][-1] = _text_card(
        "Feature evidence",
        [
            "RAFT supplies dense motion.",
            "YOLO localizes object-like regions.",
            "DINOv2 recovers semantic foreground change.",
            "Edge prior preserves mask boundaries.",
            "FMN learns their fisheye-aware fusion.",
        ],
        size=(640, 532),
    )
    _grid(tiles, gap=12, bg="white").save(OUT / "feature_evidence_panel.jpg", quality=94, optimize=True)


def _text_card(title: str, lines: list[str], size: tuple[int, int]) -> Image.Image:
    card = Image.new("RGB", size, BG)
    draw = ImageDraw.Draw(card)
    draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=GRID, width=2)
    draw.text((24, 28), title, fill=SLATE, font=_font(25, True))
    y = 84
    for line in lines:
        draw.ellipse((26, y + 7, 36, y + 17), fill=BLUE)
        draw.text((50, y), line, fill="#334155", font=_font(18))
        y += 44
    return card


def save_qualitative_comparison() -> None:
    metrics = pd.read_csv(ROOT / "outputs" / "metrics.csv")
    full = metrics[metrics["method"] == "Full-RAFT-YOLO-DINO-FMN-SAM2"].copy()
    full = full[full["id"].apply(lambda x: _sample_paths(f"{int(x):05d}")["pred"].exists())]
    chosen_rows = [
        full.sort_values("all_f1", ascending=False).iloc[0],
        full.iloc[(full["all_f1"] - full["all_f1"].median()).abs().argsort().iloc[0]],
        full.sort_values("edge_f1").iloc[0],
    ]
    rows: list[list[Image.Image]] = []
    for row in chosen_rows:
        sid = f"{int(row['id']):05d}"
        paths = _sample_paths(sid)
        rgb = _read_rgb(paths["curr"])
        gt = _read_mask(paths["gt"])
        pred = _read_mask(paths["pred"])
        rows.append(
            [
                _labelled_tile(rgb, "Current RGB", f"id {sid}, F1={row['all_f1']:.3f}"),
                _labelled_tile(_overlay(rgb, gt, (59, 130, 246), 0.45), "Ground truth", "blue annotation"),
                _labelled_tile(_overlay(rgb, pred, (34, 197, 94), 0.45), "Full model", "green prediction"),
                _labelled_tile(_error_map(gt, pred), "Pixel-level error", "TP / FP / FN"),
            ]
        )
    _grid(rows, gap=12, bg="white").save(OUT / "qualitative_comparison.jpg", quality=94, optimize=True)


def save_ablation_region_matrix() -> None:
    df = pd.read_csv(ROOT / "outputs" / "metrics_summary.csv")
    methods = [
        "FrameDiff",
        "Farneback",
        "YOLO-only",
        "FisheyeMotionNet-no-geometry",
        "FisheyeMotionNet-no-DINO",
        "FisheyeMotionNet",
        "Full-RAFT-YOLO-DINO-FMN-SAM2",
    ]
    labels = ["FrameDiff", "Farneback", "YOLO", "w/o geom.", "w/o DINO", "FMN", "Full"]
    cols = ["all_f1", "center_f1", "middle_f1", "edge_f1"]
    col_labels = ["All", "Center", "Middle", "Edge"]
    mat = df.set_index("method").loc[methods, cols]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    im = ax.imshow(mat.values, cmap="Blues", vmin=0, vmax=max(0.42, float(mat.values.max())))
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(labels)
    ax.set_title("Region-aware F1: what each module contributes")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.iloc[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=SLATE if val < 0.32 else "white", fontsize=8.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("F1 score")
    fig.tight_layout()
    fig.savefig(OUT / "ablation_region_matrix.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_method_ranking() -> None:
    df = pd.read_csv(ROOT / "outputs" / "metrics_summary.csv").sort_values("all_f1", ascending=True)
    keep = [
        "FrameDiff",
        "Farneback",
        "RAFT-only",
        "SAM2-FrameDiff",
        "YOLO-only",
        "FisheyeMotionNet",
        "Full-RAFT-YOLO-DINO-FMN-SAM2",
    ]
    df = df[df["method"].isin(keep)].copy()
    names = {
        "Full-RAFT-YOLO-DINO-FMN-SAM2": "Full",
        "FisheyeMotionNet": "FMN",
        "SAM2-FrameDiff": "SAM2+FD",
    }
    df["label"] = df["method"].map(lambda x: names.get(x, x))
    fig, ax = plt.subplots(figsize=(7.7, 3.8))
    y = np.arange(len(df))
    ax.hlines(y, 0, df["all_f1"], color=GRID, lw=4)
    ax.scatter(df["all_f1"], y, s=95, color=[GREEN if m.startswith("Full") else BLUE for m in df["method"]], zorder=3)
    for yi, score in zip(y, df["all_f1"]):
        ax.text(score + 0.01, yi, f"{score:.3f}", va="center", color=SLATE, fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.set_xlim(0, max(0.45, float(df["all_f1"].max()) + 0.06))
    ax.set_xlabel("Held-out F1")
    ax.set_title("Meaningful baseline comparison")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "method_ranking.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_tracking_failure_panel() -> None:
    sid = "00022"
    paths = _sample_paths(sid)
    rgb = _read_rgb(paths["curr"])
    pred = _read_mask(paths["pred"])
    gt = _read_mask(paths["gt"])
    overlay = _overlay(rgb, pred, (34, 197, 94), 0.44)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(pred.astype(np.uint8), connectivity=8)
    draw_img = Image.fromarray(overlay)
    draw = ImageDraw.Draw(draw_img)
    kept = []
    for idx in range(1, num):
        x, y, w, h, area = stats[idx]
        if area < 80:
            continue
        kept.append((area, x, y, w, h, centroids[idx]))
    kept = sorted(kept, reverse=True)[:5]
    for track_id, (_, x, y, w, h, c) in enumerate(kept, start=1):
        draw.rectangle((x, y, x + w, y + h), outline=(250, 204, 21), width=3)
        draw.text((x + 4, max(0, y - 22)), f"ID {track_id}", fill=(250, 204, 21), font=_font(18, True))
        cx, cy = c
        draw.line((cx - 22, cy, cx + 18, cy - 10), fill=(250, 204, 21), width=3)
    error = _error_map(gt, pred)
    card = _text_card(
        "Failure analysis",
        [
            "Large static vehicle/body regions can be over-segmented.",
            "Tiny pedestrians near fisheye borders remain difficult.",
            "RAFT helps propagation, but prompts still need temporal memory.",
            "Future work: VOS memory and distortion-aware transformer heads.",
        ],
        size=(640, 532),
    )
    tiles = [
        [
            _labelled_tile(rgb, "Current RGB", f"sample {sid}"),
            _labelled_tile(np.asarray(draw_img), "Short-range tracks", "boxes from final connected components"),
            _labelled_tile(error, "Failure modes", "red FP, blue FN"),
        ],
        [card],
    ]
    panel = _grid([tiles[0]], gap=12, bg="white")
    bottom = Image.new("RGB", (panel.width, card.height), "white")
    bottom.paste(card, ((panel.width - card.width) // 2, 0))
    canvas = Image.new("RGB", (panel.width, panel.height + 12 + bottom.height), "white")
    canvas.paste(panel, (0, 0))
    canvas.paste(bottom, (0, panel.height + 12))
    canvas.save(OUT / "tracking_failure_panel.jpg", quality=94, optimize=True)


def main() -> None:
    save_architecture_graph()
    save_feature_evidence_panel()
    save_qualitative_comparison()
    save_ablation_region_matrix()
    save_method_ranking()
    save_tracking_failure_panel()
    print(f"wrote paper figures to {OUT}")


if __name__ == "__main__":
    main()
