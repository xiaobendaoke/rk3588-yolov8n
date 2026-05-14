from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

BBox = Tuple[int, int, int, int]
"""边界框类型别名，(x1, y1, x2, y2) 格式的元组。"""


@dataclass(slots=True)
class Detection:
    """推理引擎产生的单个目标检测结果。

    Attributes:
        ts_ms: 检测结果的毫秒级时间戳。
        class_id: 类别数字编号。
        class_name: 可读的类别名称（如 "cup", "phone"）。
        conf: 置信度分数，范围 [0, 1]。
        bbox_xyxy: 边界框的像素坐标 (x1, y1, x2, y2)。
    """

    ts_ms: int
    class_id: int
    class_name: str
    conf: float
    bbox_xyxy: BBox


@dataclass(slots=True)
class RiskEventCandidate:
    """规则引擎产生的风险事件候选项。

    Attributes:
        ts_ms: 检测到风险的毫秒级时间戳。
        risk_type: 风险类型（如 "liquid_near_electronics"）。
        severity: 严重级别（"high", "medium", "low"）。
        confidence: 风险评估的整体置信度。
        reason: 触发风险的可读原因说明。
        objects: 与该风险相关的 Detection 对象列表。
    """

    ts_ms: int
    risk_type: str
    severity: str
    confidence: float
    reason: str
    objects: List[Detection] = field(default_factory=list)


@dataclass(slots=True)
class RuntimeStatus:
    """监控系统的运行时状态快照。

    Attributes:
        fps: 当前推理帧率（每秒帧数）。
        queue_size: 处理队列中等待的帧数。
        last_event_time: 最近一次风险事件的 ISO 格式时间戳。
        cpu_percent: CPU 使用率百分比。
        gpu_load: GPU 负载百分比。
        npu_load: NPU 负载百分比。
        mem_percent: 内存使用率百分比。
        cpu_temp: CPU 温度（摄氏度）。
        gpu_temp: GPU 温度（摄氏度）。
        detection_count: 当前帧中检测到的目标数量。
    """

    fps: float = 0.0
    queue_size: int = 0
    last_event_time: str = ""
    cpu_percent: float = 0.0
    gpu_load: float = 0.0
    npu_load: float = 0.0
    mem_percent: float = 0.0
    cpu_temp: float = 0.0
    gpu_temp: float = 0.0
    detection_count: int = 0


def detection_to_dict(det: Detection) -> Dict[str, Any]:
    """将 Detection 对象转换为可序列化的字典。

    Args:
        det: 要转换的 Detection 对象。

    Returns:
        包含 ts_ms, class_id, class_name, conf, bbox_xyxy 键的字典。
    """
    return {
        "ts_ms": det.ts_ms,
        "class_id": det.class_id,
        "class_name": det.class_name,
        "conf": round(det.conf, 4),
        "bbox_xyxy": list(det.bbox_xyxy),
    }