from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .baselines import farneback, frame_diff, raft_only, yolo_only
from .config import ensure_dirs, image_size, load_config
from .dataset import Sample, discover_samples, load_calibration, split_samples
from .deep_modules import Sam2Refiner, load_boxes
from .geometry import geometry_maps, rectify_image
from .metrics import region_stats
from .model import FisheyeMotionNet
from .tracking import boxes_from_mask, draw_boxes, propagate_boxes
from .utils import read_mask, read_rgb, require_cuda, resize_img, set_seed
from .visualize import error_map, flow_to_rgb, overlay_mask, save_grid


def make_input(
    sample: Sample,
    cfg: dict,
    zero_raft: bool = False,
    zero_yolo: bool = False,
    zero_geom: bool = False,
) -> tuple[torch.Tensor, dict[str, np.ndarray]]:
    size = image_size(cfg)
    prev = resize_img(read_rgb(sample.previous), size, cv2.INTER_AREA)
    curr = resize_img(read_rgb(sample.current), size, cv2.INTER_AREA)
    diff = np.mean(np.abs(curr.astype(np.float32) / 255.0 - prev.astype(np.float32) / 255.0), axis=2, keepdims=True)
    feat = np.load(Path(cfg["feature_cache"]) / f"{sample.sid}.npz")
    flow = feat["flow"].astype(np.float32)
    yolo = feat["yolo_objectness"].astype(np.float32)
    if flow.shape[:2] != size:
        flow = cv2.resize(flow, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    if yolo.shape[:2] != size:
        yolo = cv2.resize(yolo, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)
    geom = geometry_maps(load_calibration(sample.calib), size)
    if zero_raft:
        flow = np.zeros_like(flow)
    if zero_yolo:
        yolo = np.zeros_like(yolo)
    if zero_geom:
        geom = np.zeros_like(geom)
    x = np.concatenate(
        [
            prev.astype(np.float32) / 255.0,
            curr.astype(np.float32) / 255.0,
            diff,
            flow,
            yolo[..., None],
            geom,
        ],
        axis=2,
    ).transpose(2, 0, 1)
    return torch.from_numpy(x[None]).float(), {"prev": prev, "curr": curr, "flow": flow, "yolo": yolo, "geom": geom}


@torch.inference_mode()
def predict_net(model: torch.nn.Module, x: torch.Tensor) -> np.ndarray:
    prob = torch.sigmoid(model(x.cuda(non_blocking=True)))[0, 0].detach().cpu().numpy()
    return prob.astype(np.float32)


def add_metric(rows: list[dict], method: str, sid: str, pred: np.ndarray, gt: np.ndarray, r_map: np.ndarray) -> None:
    row = {"id": sid, "method": method}
    row.update(region_stats(pred.astype(bool), gt.astype(bool), r_map))
    rows.append(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-sam", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    require_cuda()
    set_seed(int(cfg["seed"]))
    size = image_size(cfg)
    samples = discover_samples(cfg["data_root"])
    test = split_samples(samples, int(cfg["seed"]), cfg["splits"])["test"]
    if args.limit:
        test = test[: args.limit]
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = FisheyeMotionNet(in_channels=int(ckpt["in_channels"])).cuda().eval()
    model.load_state_dict(ckpt["model"])
    sam = None
    if cfg["sam2"]["enabled"] and not args.no_sam:
        sam = Sam2Refiner(cfg["sam2"]["repo_dir"], cfg["sam2"]["checkpoint"], cfg["sam2"]["config"])
    rows: list[dict] = []
    vis_dir = Path(cfg["output_root"]) / "visualizations"
    pred_dir = Path(cfg["output_root"]) / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for i, sample in enumerate(tqdm(test, desc="experiments")):
        x, comp = make_input(sample, cfg)
        curr = comp["curr"]
        prev = comp["prev"]
        gt = resize_img(read_mask(sample.gt), size, cv2.INTER_NEAREST).astype(np.uint8)
        r_map = comp["geom"][..., 0]
        fd = frame_diff(prev, curr)
        fb, _ = farneback(prev, curr)
        raft = raft_only(comp["flow"])
        yo = yolo_only(comp["yolo"])
        boxes = load_boxes(cfg["feature_cache"], sample.sid)
        sam_fd = sam.refine(curr, fd.astype(np.float32), []) if sam is not None else fd
        variants = {
            "FrameDiff": fd,
            "Farneback": fb,
            "RAFT-only": raft,
            "YOLO-only": yo,
            "SAM2-FrameDiff": sam_fd,
        }
        for name, pred in variants.items():
            add_metric(rows, name, sample.sid, pred, gt, r_map)
        prob_full = predict_net(model, x)
        prob_no_raft = predict_net(model, make_input(sample, cfg, zero_raft=True)[0])
        prob_no_yolo = predict_net(model, make_input(sample, cfg, zero_yolo=True)[0])
        prob_no_geom = predict_net(model, make_input(sample, cfg, zero_geom=True)[0])
        net_masks = {
            "FisheyeMotionNet-no-RAFT": prob_no_raft >= float(cfg["threshold"]),
            "FisheyeMotionNet-no-YOLO": prob_no_yolo >= float(cfg["threshold"]),
            "FisheyeMotionNet-no-geometry": prob_no_geom >= float(cfg["threshold"]),
            "FisheyeMotionNet": prob_full >= float(cfg["threshold"]),
        }
        if sam is not None:
            full = sam.refine(curr, prob_full, boxes, threshold=float(cfg["threshold"]))
        else:
            full = (prob_full >= float(cfg["threshold"])).astype(np.uint8)
        net_masks["Full-RAFT-YOLO-FMN-SAM2"] = full
        for name, pred in net_masks.items():
            add_metric(rows, name, sample.sid, pred.astype(np.uint8), gt, r_map)
        cv2.imwrite(str(pred_dir / f"{sample.sid}_full.png"), (full * 255).astype(np.uint8))
        if i < 8:
            prev_boxes = boxes_from_mask(raft)
            prop_boxes = propagate_boxes(prev_boxes, comp["flow"])
            boxed = draw_boxes(overlay_mask(curr, full), prop_boxes, (255, 255, 0))
            calib = load_calibration(sample.calib)
            rectified = rectify_image(curr, calib, size)
            save_grid(
                vis_dir / f"{sample.sid}_summary.png",
                [
                    ("previous", prev),
                    ("current", curr),
                    ("rectified", rectified),
                    ("RAFT flow", flow_to_rgb(comp["flow"])),
                    ("YOLO objectness", comp["yolo"]),
                    ("FMN prob", prob_full),
                    ("full mask", overlay_mask(curr, full)),
                    ("ground truth", gt),
                    ("error TP/FP/FN", error_map(full, gt)),
                    ("tracking", boxed),
                ],
            )
    df = pd.DataFrame(rows)
    out_csv = Path(cfg["output_root"]) / "metrics.csv"
    df.to_csv(out_csv, index=False)
    summary = df.groupby("method").mean(numeric_only=True).sort_values("all_f1", ascending=False)
    summary.to_csv(Path(cfg["output_root"]) / "metrics_summary.csv")
    print(summary[["all_iou", "all_precision", "all_recall", "all_f1", "center_f1", "middle_f1", "edge_f1"]])
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
