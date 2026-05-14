from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import yaml


@dataclass(slots=True)
class Settings:
    """从 YAML 文件加载的应用配置。

    Attributes:
        app_name: 应用名称。
        log_level: 日志级别（如 "INFO", "DEBUG"）。
        camera_device: 摄像头设备路径（如 "/dev/video0"）。
        camera_width: 摄像头捕获宽度（像素）。
        camera_height: 摄像头捕获高度（像素）。
        camera_crop_left: 是否裁剪左半帧（用于双目摄像头）。
        input_size: 模型输入尺寸（宽高相同，正方形）。
        infer_interval_ms: 推理间隔（毫秒）。
        model_path: RKNN 模型文件路径。
        class_names: 模型可检测的类别名称列表。
        conf_threshold: 检测结果置信度阈值。
        nms_threshold: NMS 去重的 IoU 阈值。
        risk_distance_px: 液体-设备接近风险的像素距离阈值。
        risk_hold_frames: 确认风险所需的连续帧数。
        event_cooldown_sec: 同类型风险事件的最小间隔秒数。
        danger_roi: 尖锐工具检测的危险区域 [x1, y1, x2, y2]。
        dense_count_threshold: 桌面拥挤风险的目标数量阈值。
        dense_iou_sum_threshold: 桌面拥挤风险的 IoU 总和阈值。
        snapshot_root: 事件快照保存的根目录。
        db_path: SQLite 数据库文件路径。
        web_host: Web 服务器绑定地址。
        web_port: Web 服务器绑定端口。
    """

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
    npu_threads: int
    use_native: bool


def load_settings(config_path: str) -> Settings:
    """从 YAML 配置文件加载应用设置。

    Args:
        config_path: YAML 配置文件路径。

    Returns:
        填充完整的 Settings 实例，缺失的可选字段使用合理默认值。
    """
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
        npu_threads=int(raw.get("npu_threads", 1)),
        use_native=bool(raw.get("use_native", False)),
    )