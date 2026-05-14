from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from app.types import Detection, RiskEventCandidate


@dataclass(slots=True)
class RuleConfig:
    """规则引擎的配置参数。

    Attributes:
        risk_distance_px: 液体-设备接近检测的最大像素距离。
        risk_hold_frames: 风险触发前需要持续的连续帧数。
        event_cooldown_sec: 同类型风险事件之间的最小间隔秒数。
        danger_roi: 尖锐工具误放检测的危险区域 (x1, y1, x2, y2)。
        dense_count_threshold: 判定桌面拥挤的目标数量阈值。
        dense_iou_sum_threshold: 判定桌面拥挤的 IoU 总和阈值。
    """

    risk_distance_px: int
    risk_hold_frames: int
    event_cooldown_sec: int
    danger_roi: Tuple[int, int, int, int]
    dense_count_threshold: int
    dense_iou_sum_threshold: float


class RuleEngine:
    """在检测到的目标上评估安全规则并产生风险事件。

    实现三条安全规则：
    1. 液体靠近电子设备 - 杯子靠近手机/键盘时触发。
    2. 尖锐工具误放 - 剪刀在配置的危险区域内时触发。
    3. 桌面物品拥挤 - 目标过多或 IoU 总和过大时触发。

    使用持续帧计数和事件冷却机制防止重复事件。

    Attributes:
        cfg: 规则配置参数。
    """

    def __init__(self, cfg: RuleConfig) -> None:
        """初始化规则引擎。

        Args:
            cfg: 所有规则的配置参数。
        """
        self.cfg = cfg
        self._hold_counter: Dict[str, int] = defaultdict(int)
        self._last_emit_ts: Dict[str, float] = defaultdict(lambda: 0.0)

    def evaluate(self, detections: List[Detection]) -> List[RiskEventCandidate]:
        """对当前检测结果评估所有安全规则。

        在发出事件前应用持续帧计数和冷却逻辑。

        Args:
            detections: 当前帧中检测到的目标列表。

        Returns:
            通过持续帧和冷却检查的风险事件列表。
        """
        now = time.time()
        candidates: List[RiskEventCandidate] = []
        matched_rules: set[str] = set()

        liquid = self._check_liquid_near_electronics(detections)
        sharp = self._check_sharp_tool_misplaced(detections)
        dense = self._check_desk_overcrowded(detections)

        for c in [liquid, sharp, dense]:
            if c is None:
                continue
            rule = c.risk_type
            matched_rules.add(rule)
            self._hold_counter[rule] += 1
            if self._hold_counter[rule] < self.cfg.risk_hold_frames:
                continue
            if now - self._last_emit_ts[rule] < self.cfg.event_cooldown_sec:
                continue
            self._last_emit_ts[rule] = now
            candidates.append(c)

        for rule_name in [
            "liquid_near_electronics",
            "sharp_tool_misplaced",
            "desk_overcrowded",
        ]:
            if rule_name not in matched_rules:
                self._hold_counter[rule_name] = 0

        return candidates

    def _check_liquid_near_electronics(self, ds: List[Detection]) -> RiskEventCandidate | None:
        """检查是否有杯子离电子设备（手机、键盘）太近。

        Args:
            ds: 当前帧中的检测目标列表。

        Returns:
            如果杯子在 risk_distance_px 距离内靠近设备，返回 RiskEventCandidate，
            否则返回 None。
        """
        cups = [d for d in ds if d.class_name == "cup"]
        devices = [d for d in ds if d.class_name in {"cell phone", "keyboard"}]
        if not cups or not devices:
            return None

        pairs: List[tuple[Detection, Detection, float]] = []
        for cup in cups:
            for dev in devices:
                dist = _center_distance(cup.bbox_xyxy, dev.bbox_xyxy)
                if dist <= self.cfg.risk_distance_px:
                    pairs.append((cup, dev, dist))

        if not pairs:
            return None
        pairs.sort(key=lambda x: x[2])
        cup, dev, dist = pairs[0]

        conf = min(cup.conf, dev.conf)
        severity = "high" if dist < self.cfg.risk_distance_px * 0.6 else "medium"
        return RiskEventCandidate(
            ts_ms=cup.ts_ms,
            risk_type="liquid_near_electronics",
            severity=severity,
            confidence=float(conf),
            reason=f"cup near {dev.class_name}, dist={dist:.1f}px",
            objects=[cup, dev],
        )

    def _check_sharp_tool_misplaced(self, ds: List[Detection]) -> RiskEventCandidate | None:
        """检查是否有剪刀在配置的危险区域内。

        Args:
            ds: 当前帧中的检测目标列表。

        Returns:
            如果在危险区域发现剪刀，返回 RiskEventCandidate，否则返回 None。
        """
        scissors = [d for d in ds if d.class_name == "scissors"]
        if not scissors:
            return None

        x1, y1, x2, y2 = self.cfg.danger_roi
        for s in scissors:
            cx, cy = _center(s.bbox_xyxy)
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return RiskEventCandidate(
                    ts_ms=s.ts_ms,
                    risk_type="sharp_tool_misplaced",
                    severity="high",
                    confidence=float(s.conf),
                    reason="scissors detected in configured danger ROI",
                    objects=[s],
                )
        return None

    def _check_desk_overcrowded(self, ds: List[Detection]) -> RiskEventCandidate | None:
        """根据目标数量或 IoU 总和检查桌面是否拥挤。

        Args:
            ds: 当前帧中的检测目标列表。

        Returns:
            如果目标数量超过阈值或 IoU 总和超过阈值，返回 RiskEventCandidate，
            否则返回 None。
        """
        if not ds:
            return None

        iou_sum = 0.0
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                iou_sum += _iou(ds[i].bbox_xyxy, ds[j].bbox_xyxy)

        if len(ds) >= self.cfg.dense_count_threshold or iou_sum >= self.cfg.dense_iou_sum_threshold:
            conf = sum(d.conf for d in ds) / max(len(ds), 1)
            return RiskEventCandidate(
                ts_ms=ds[0].ts_ms,
                risk_type="desk_overcrowded",
                severity="medium",
                confidence=float(conf),
                reason=f"objects={len(ds)}, iou_sum={iou_sum:.2f}",
                objects=ds,
            )
        return None


def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    """计算边界框的中心点。

    Args:
        box: (x1, y1, x2, y2) 格式的边界框。

    Returns:
        中心点坐标 (cx, cy)，浮点数。
    """
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _center_distance(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """计算两个边界框中心点之间的欧氏距离。

    Args:
        a: 第一个边界框 (x1, y1, x2, y2)。
        b: 第二个边界框 (x1, y1, x2, y2)。

    Returns:
        两个框中心点之间的欧氏距离。
    """
    ax, ay = _center(a)
    bx, by = _center(b)
    return math.hypot(ax - bx, ay - by)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """计算两个边界框的交并比（IoU）。

    Args:
        a: 第一个边界框 (x1, y1, x2, y2)。
        b: 第二个边界框 (x1, y1, x2, y2)。

    Returns:
        [0, 1] 范围内的 IoU 值，框不重叠时返回 0.0。
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(ix2 - ix1, 0), max(iy2 - iy1, 0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(ax2 - ax1, 0) * max(ay2 - ay1, 0)
    area_b = max(bx2 - bx1, 0) * max(by2 - by1, 0)
    denom = area_a + area_b - inter
    if denom <= 0:
        return 0.0
    return inter / denom