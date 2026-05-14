from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

BBox = Tuple[int, int, int, int]


@dataclass(slots=True)
class Detection:
    ts_ms: int
    class_id: int
    class_name: str
    conf: float
    bbox_xyxy: BBox


@dataclass(slots=True)
class RiskEventCandidate:
    ts_ms: int
    risk_type: str
    severity: str
    confidence: float
    reason: str
    objects: List[Detection] = field(default_factory=list)


@dataclass(slots=True)
class RuntimeStatus:
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
    return {
        "ts_ms": det.ts_ms,
        "class_id": det.class_id,
        "class_name": det.class_name,
        "conf": round(det.conf, 4),
        "bbox_xyxy": list(det.bbox_xyxy),
    }
