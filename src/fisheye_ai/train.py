from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import apply_processing_domain, ensure_dirs, image_size, load_config
from .dataset import FisheyeMotionDataset, discover_samples, split_samples
from .losses import MotionLoss
from .metrics import binary_stats
from .model import FisheyeMotionNet
from .utils import require_cuda, set_seed


def evaluate(model: torch.nn.Module, loader: DataLoader, threshold: float) -> dict[str, float]:
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch in loader:
            x = batch["image"].cuda(non_blocking=True)
            y = batch["mask"].cuda(non_blocking=True)
            logits = model(x)
            pred = (torch.sigmoid(logits) >= threshold).detach().cpu().numpy()
            gt = y.detach().cpu().numpy()
            for p, g in zip(pred, gt):
                rows.append(binary_stats(p[0], g[0]))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--domain", default="fisheye", choices=["fisheye", "rectified"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overfit", action="store_true")
    args = parser.parse_args()
    cfg = apply_processing_domain(load_config(args.config), args.domain)
    ensure_dirs(cfg)
    require_cuda()
    set_seed(int(cfg["seed"]))
    samples = discover_samples(cfg["data_root"])
    splits = split_samples(samples, int(cfg["seed"]), cfg["splits"])
    if args.limit:
        splits = {k: v[: args.limit] for k, v in splits.items()}
    if args.overfit:
        splits["val"] = splits["train"]
    size = image_size(cfg)
    train_ds = FisheyeMotionDataset(
        splits["train"],
        size,
        cfg["feature_cache"],
        require_deep_features=True,
        processing_domain=cfg["processing_domain"],
        rectified_fov_deg=float(cfg.get("rectified", {}).get("fov_deg", 120.0)),
    )
    val_ds = FisheyeMotionDataset(
        splits["val"],
        size,
        cfg["feature_cache"],
        require_deep_features=True,
        processing_domain=cfg["processing_domain"],
        rectified_fov_deg=float(cfg.get("rectified", {}).get("fov_deg", 120.0)),
    )
    train_loader = DataLoader(train_ds, batch_size=int(cfg["batch_size"]), shuffle=True, num_workers=int(cfg["num_workers"]), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=int(cfg["batch_size"]), shuffle=False, num_workers=int(cfg["num_workers"]), pin_memory=True)
    in_ch = int(train_ds[0]["image"].shape[0])
    model = FisheyeMotionNet(in_channels=in_ch).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))
    loss_fn = MotionLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.get("amp", True)))
    epochs = int(args.epochs or cfg["epochs"])
    best_f1 = -1.0
    patience = int(cfg["early_stop_patience"])
    stale = 0
    history = []
    ckpt_dir = Path(cfg["checkpoint_dir"])
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}")
        for batch in pbar:
            x = batch["image"].cuda(non_blocking=True)
            y = batch["mask"].cuda(non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(cfg.get("amp", True))):
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            pbar.set_postfix(loss=np.mean(losses))
        val = evaluate(model, val_loader, float(cfg["threshold"]))
        row = {"epoch": epoch, "loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        pd.DataFrame(history).to_csv(Path(cfg["output_root"]) / "training_history.csv", index=False)
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            stale = 0
            torch.save({"model": model.state_dict(), "in_channels": in_ch, "config": cfg, "epoch": epoch, "val": val}, ckpt_dir / "best_f1.pth")
        else:
            stale += 1
        print(row)
        if stale >= patience:
            print("early stop")
            break
    print(f"best val f1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
