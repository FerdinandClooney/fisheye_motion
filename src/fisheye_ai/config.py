from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _resolve_path(value: str | Path, project_root: Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return str(path.resolve())


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    repo_root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    project_root_value = cfg.get("project_root", repo_root)
    project_root = Path(project_root_value).expanduser()
    if not project_root.is_absolute():
        project_root = repo_root / project_root
    project_root = project_root.resolve()
    cfg["project_root"] = str(project_root)

    for key in ("data_root", "output_root", "checkpoint_dir", "feature_cache"):
        if key in cfg:
            cfg[key] = _resolve_path(cfg[key], project_root)

    sam2_cfg = cfg.get("sam2", {})
    for key in ("repo_dir", "checkpoint"):
        if key in sam2_cfg:
            sam2_cfg[key] = _resolve_path(sam2_cfg[key], project_root)

    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    for key in ("output_root", "checkpoint_dir", "feature_cache"):
        Path(cfg[key]).mkdir(parents=True, exist_ok=True)


def image_size(cfg: dict[str, Any]) -> tuple[int, int]:
    h, w = cfg["image_size"]
    return int(h), int(w)


def apply_processing_domain(cfg: dict[str, Any], domain: str) -> dict[str, Any]:
    if domain not in {"fisheye", "rectified"}:
        raise ValueError(f"Unsupported domain: {domain}")
    out = copy.deepcopy(cfg)
    out["processing_domain"] = domain
    rectified_cfg = out.setdefault("rectified", {})
    rectified_cfg.setdefault("fov_deg", 120.0)
    if domain == "rectified":
        for key in ("output_root", "checkpoint_dir", "feature_cache"):
            path = Path(out[key])
            out[key] = str((path.parent / f"{path.name}_rectified").resolve())
    return out
