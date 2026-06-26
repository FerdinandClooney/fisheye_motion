from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import apply_processing_domain, load_config


def load_summary(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "route" not in df.columns:
        df["route"] = "fisheye" if "rectified" not in path.parent.name else "rectified"
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    fish_cfg = apply_processing_domain(base_cfg, "fisheye")
    rect_cfg = apply_processing_domain(base_cfg, "rectified")
    fish_summary = load_summary(Path(fish_cfg["output_root"]) / "metrics_summary.csv")
    rect_summary = load_summary(Path(rect_cfg["output_root"]) / "metrics_summary.csv")
    merged = pd.concat([fish_summary, rect_summary], ignore_index=True)

    out_csv = Path(fish_cfg["output_root"]) / "route_comparison.csv"
    merged.to_csv(out_csv, index=False)

    focus = merged[merged["method"] == "Full-RAFT-YOLO-DINO-FMN-SAM2"].copy()
    metrics = ["all_f1", "center_f1", "middle_f1", "edge_f1"]
    labels = ["All", "Center", "Middle", "Edge"]
    fig_dir = Path(fish_cfg["output_root"]) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.2, 4.1))
    x = range(len(metrics))
    width = 0.34
    fish_vals = focus.loc[focus["route"] == "fisheye", metrics].iloc[0].tolist()
    rect_vals = focus.loc[focus["route"] == "rectified", metrics].iloc[0].tolist()
    plt.bar([i - width / 2 for i in x], fish_vals, width=width, label="Fisheye domain", color="#2563eb")
    plt.bar([i + width / 2 for i in x], rect_vals, width=width, label="Rectified domain", color="#d97706")
    plt.xticks(list(x), labels)
    plt.ylabel("F1")
    plt.ylim(0, max(fish_vals + rect_vals) + 0.08)
    plt.title("Route Comparison on the Full Model")
    plt.grid(axis="y", alpha=0.22)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fig_dir / "route_comparison.png", dpi=220)
    print(merged[["route", "method", "all_f1", "center_f1", "middle_f1", "edge_f1"]].sort_values(["method", "route"]))
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
