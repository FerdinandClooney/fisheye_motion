from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(cfg["output_root"]) / "figures"
    out.mkdir(parents=True, exist_ok=True)
    hist_path = Path(cfg["output_root"]) / "training_history.csv"
    if hist_path.exists():
        hist = pd.read_csv(hist_path)
        plt.figure(figsize=(7, 4))
        plt.plot(hist["epoch"], hist["loss"], label="train loss")
        if "val_f1" in hist:
            plt.plot(hist["epoch"], hist["val_f1"], label="val F1")
        plt.xlabel("epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "training_curve.png", dpi=180)
    metrics_path = Path(cfg["output_root"]) / "metrics_summary.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        method_col = "method" if "method" in metrics.columns else metrics.columns[0]
        plt.figure(figsize=(10, 4))
        plt.bar(metrics[method_col], metrics["all_f1"])
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("F1")
        plt.tight_layout()
        plt.savefig(out / "method_f1_bar.png", dpi=180)


if __name__ == "__main__":
    main()
