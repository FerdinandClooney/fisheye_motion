from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 160,
        "savefig.dpi": 220,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


COLORS = {
    "blue": "#2563eb",
    "cyan": "#0891b2",
    "green": "#16a34a",
    "amber": "#d97706",
    "rose": "#e11d48",
    "slate": "#334155",
    "gray": "#94a3b8",
}


def load_summary() -> pd.DataFrame:
    path = ROOT / "outputs" / "metrics_summary.csv"
    df = pd.read_csv(path)
    if "method" not in df.columns:
        df = df.rename(columns={df.columns[0]: "method"})
    return df


def save_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.set_axis_off()
    blocks = [
        ("Fisheye Pair\nprev/current", 0.02, COLORS["slate"]),
        ("RAFT\nflow field", 0.18, COLORS["blue"]),
        ("YOLO26-seg\nobject prior", 0.32, COLORS["green"]),
        ("DINOv2\nsemantic change", 0.46, COLORS["cyan"]),
        ("Geometry + Edge\nr, theta, boundary", 0.60, COLORS["amber"]),
        ("FisheyeMotionNet\nAttention U-Net", 0.75, COLORS["rose"]),
        ("Uncertainty Fusion\n+ SAM2.1 refine", 0.90, COLORS["slate"]),
    ]
    y = 0.55
    for i, (text, x, color) in enumerate(blocks):
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.55,rounding_size=0.08", fc=color, ec="none"),
            transform=ax.transAxes,
        )
        if i < len(blocks) - 1:
            ax.annotate(
                "",
                xy=(blocks[i + 1][1] - 0.065, y),
                xytext=(x + 0.065, y),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#475569"),
                xycoords=ax.transAxes,
            )
    ax.text(
        0.5,
        0.12,
        "Deep motion evidence, semantic priors and fisheye geometry are fused before promptable SAM2 boundary densification.",
        ha="center",
        va="center",
        color="#475569",
        fontsize=10,
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(OUT / "pipeline_overview.png", bbox_inches="tight")
    plt.close(fig)


def save_metrics(df: pd.DataFrame) -> None:
    chosen = [
        "Full-RAFT-YOLO-DINO-FMN-SAM2",
        "FisheyeMotionNet",
        "FisheyeMotionNet-no-DINO",
        "FisheyeMotionNet-no-edge",
        "YOLO-only",
        "FrameDiff",
    ]
    plot_df = df[df["method"].isin(chosen)].copy()
    plot_df["method"] = pd.Categorical(plot_df["method"], chosen, ordered=True)
    plot_df = plot_df.sort_values("method")
    labels = ["Full", "FMN", "no DINO", "no Edge", "YOLO", "FrameDiff"]
    x = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(x - 0.18, plot_df["all_f1"], width=0.36, label="F1", color=COLORS["blue"])
    ax.bar(x + 0.18, plot_df["all_iou"], width=0.36, label="IoU", color=COLORS["cyan"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(0.45, float(plot_df["all_f1"].max()) + 0.06))
    ax.set_ylabel("Score")
    ax.set_title("Main Results on the Held-out Fisheye Test Set")
    ax.grid(axis="y", alpha=0.22)
    for xi, value in zip(x - 0.18, plot_df["all_f1"]):
        ax.text(xi, value + 0.008, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.legend(frameon=False, ncols=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "main_metrics.png", bbox_inches="tight")
    plt.close(fig)


def save_region_heatmap(df: pd.DataFrame) -> None:
    methods = [
        "Full-RAFT-YOLO-DINO-FMN-SAM2",
        "FisheyeMotionNet",
        "FisheyeMotionNet-no-DINO",
        "YOLO-only",
        "FrameDiff",
    ]
    cols = ["center_f1", "middle_f1", "edge_f1"]
    heat = df.set_index("method").loc[methods, cols]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    im = ax.imshow(heat.values, cmap="YlGnBu", vmin=0, vmax=max(0.42, float(heat.values.max())))
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["Center", "Middle", "Edge"])
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(["Full", "FMN", "no DINO", "YOLO", "FrameDiff"])
    ax.set_title("Radial Region F1 under Fisheye Distortion")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat.iloc[i, j]:.3f}", ha="center", va="center", color="#0f172a", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("F1")
    fig.tight_layout()
    fig.savefig(OUT / "region_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def save_ablation(df: pd.DataFrame) -> None:
    base = float(df.loc[df["method"] == "FisheyeMotionNet", "all_f1"].iloc[0])
    values = {
        "TTA+Fusion": float(df.loc[df["method"] == "UncertaintyFusion", "all_f1"].iloc[0]) - base,
        "SAM2 refine": float(df.loc[df["method"] == "Full-RAFT-YOLO-DINO-FMN-SAM2", "all_f1"].iloc[0])
        - float(df.loc[df["method"] == "UncertaintyFusion", "all_f1"].iloc[0]),
        "DINO removal": float(df.loc[df["method"] == "FisheyeMotionNet-no-DINO", "all_f1"].iloc[0]) - base,
        "Edge removal": float(df.loc[df["method"] == "FisheyeMotionNet-no-edge", "all_f1"].iloc[0]) - base,
    }
    names = list(values)
    vals = np.array([values[k] for k in names])
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    colors = [COLORS["green"] if v >= 0 else COLORS["rose"] for v in vals]
    ax.axhline(0, color="#475569", lw=1)
    ax.bar(names, vals, color=colors)
    ax.set_ylabel("Delta F1 relative to FisheyeMotionNet")
    ax.set_title("Ablation Effects of Innovation Modules")
    ax.grid(axis="y", alpha=0.22)
    for i, value in enumerate(vals):
        va = "bottom" if value >= 0 else "top"
        offset = 0.004 if value >= 0 else -0.004
        ax.text(i, value + offset, f"{value:+.3f}", ha="center", va=va, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "ablation_delta.png", bbox_inches="tight")
    plt.close(fig)


def save_training() -> None:
    path = ROOT / "outputs" / "training_history.csv"
    hist = pd.read_csv(path)
    fig, ax1 = plt.subplots(figsize=(8.4, 4.5))
    ax1.plot(hist["epoch"], hist["loss"], marker="o", color=COLORS["slate"], label="Train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.22)
    ax2 = ax1.twinx()
    ax2.plot(hist["epoch"], hist["val_f1"], marker="s", color=COLORS["blue"], label="Val F1")
    ax2.plot(hist["epoch"], hist["val_iou"], marker="^", color=COLORS["cyan"], label="Val IoU")
    ax2.set_ylabel("Validation score")
    best_idx = int(hist["val_f1"].idxmax())
    ax2.scatter(hist.loc[best_idx, "epoch"], hist.loc[best_idx, "val_f1"], s=90, color=COLORS["rose"], zorder=5)
    ax2.text(
        hist.loc[best_idx, "epoch"],
        hist.loc[best_idx, "val_f1"] + 0.018,
        f"best F1={hist.loc[best_idx, 'val_f1']:.3f}",
        ha="center",
        color=COLORS["rose"],
        fontsize=9,
    )
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, frameon=False, loc="center right")
    ax1.set_title("Optimization Dynamics of the 18-channel Model")
    fig.tight_layout()
    fig.savefig(OUT / "training_dynamics.png", bbox_inches="tight")
    plt.close(fig)


def save_qualitative() -> None:
    candidates = ["00011", "00022", "00068", "00318"]
    imgs: list[Image.Image] = []
    for sid in candidates:
        path = ROOT / "outputs" / "visualizations" / f"{sid}_summary.png"
        if path.exists():
            img = Image.open(path).convert("RGB")
            img.thumbnail((1120, 520))
            canvas = Image.new("RGB", (1120, 520), "white")
            canvas.paste(img, ((1120 - img.width) // 2, (520 - img.height) // 2))
            imgs.append(canvas)
    if not imgs:
        return
    w = max(img.width for img in imgs)
    h = sum(img.height for img in imgs)
    montage = Image.new("RGB", (w, h), "white")
    y = 0
    for img in imgs:
        montage.paste(img, (0, y))
        y += img.height
    montage.save(OUT / "qualitative_montage.jpg", quality=92, optimize=True)


def main() -> None:
    df = load_summary()
    save_pipeline()
    save_metrics(df)
    save_region_heatmap(df)
    save_ablation(df)
    save_training()
    save_qualitative()
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
