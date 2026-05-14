from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass(slots=True)
class Settings:
    app_name: str
    log_level: str
    camera_device: str
    camera_width: int
    camera_height: int
    camera_crop_left: bool
    input_size: int
    infer_interval_ms: int
    model_path: str
    class_names: List[str]
    conf_threshold: float
    nms_threshold: float
    risk_distance_px: int
    risk_hold_frames: int
    event_cooldown_sec: int
    danger_roi: List[int]
    dense_count_threshold: int
    dense_iou_sum_threshold: float
    snapshot_root: str
    db_path: str
    web_host: str
    web_port: int


def load_settings(config_path: str) -> Settings:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return Settings(
        app_name=raw.get("app_name", "desk-safety"),
        log_level=raw.get("log_level", "INFO"),
        camera_device=raw["camera_device"],
        camera_width=int(raw.get("camera_width", 1280)),
        camera_height=int(raw.get("camera_height", 720)),
        camera_crop_left=bool(raw.get("camera_crop_left", False)),
        input_size=int(raw["input_size"]),
        infer_interval_ms=int(raw["infer_interval_ms"]),
        model_path=raw["model_path"],
        class_names=list(raw["class_names"]),
        conf_threshold=float(raw["conf_threshold"]),
        nms_threshold=float(raw["nms_threshold"]),
        risk_distance_px=int(raw["risk_distance_px"]),
        risk_hold_frames=int(raw["risk_hold_frames"]),
        event_cooldown_sec=int(raw["event_cooldown_sec"]),
        danger_roi=list(raw["danger_roi"]),
        dense_count_threshold=int(raw["dense_count_threshold"]),
        dense_iou_sum_threshold=float(raw["dense_iou_sum_threshold"]),
        snapshot_root=raw["snapshot_root"],
        db_path=raw["db_path"],
        web_host=raw.get("web_host", "0.0.0.0"),
        web_port=int(raw.get("web_port", 8080)),
    )
