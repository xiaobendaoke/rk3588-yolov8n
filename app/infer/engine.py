from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np

from app.types import Detection


class InferenceEngine:
    """RKNN 模型推理引擎，用于目标检测。

    加载 RKNN 模型，在摄像头帧上执行推理，并将模型输出解码为 Detection 对象。
    支持多种输出格式，包括 nx6、YOLO（带/不带目标置信度）和分离的框/类别输出。
    当模型文件不可用时，回退到存根模式以便开发测试。

    Attributes:
        model_path: RKNN 模型文件路径。
        class_names: 模型可检测的类别名称列表。
        conf_threshold: 检测结果的置信度阈值。
        nms_threshold: NMS 去重的 IoU 阈值。
        input_size: 模型输入尺寸（正方形，宽高相同）。
        rknn: 底层的 RKNNLite 运行时实例，未加载时为 None。
    """

    def __init__(
        self,
        model_path: str,
        class_names: Sequence[str],
        conf_threshold: float,
        nms_threshold: float,
        input_size: int,
    ) -> None:
        """初始化推理引擎。

        Args:
            model_path: RKNN 模型文件路径。
            class_names: 用于解码的类别名称列表。
            conf_threshold: 保留检测结果的最小置信度。
            nms_threshold: 非极大值抑制的 IoU 阈值。
            input_size: 模型输入尺寸（正方形）。
        """
        self.model_path = model_path
        self.class_names = list(class_names)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.rknn = None
        self._stub_mode = False
        self._log = logging.getLogger("desk-safety.infer")
        self._output_shape_logged = False
        self._decode_warn_count = 0

    def open(self) -> None:
        """加载 RKNN 模型并初始化运行时。

        如果模型无法加载，回退到存根模式，使管道的其他部分可以在无硬件的情况下测试。

        Raises:
            RuntimeError: RKNN 模型加载或运行时初始化失败时抛出
                （仅在模型文件存在但加载失败时抛出）。
        """
        try:
            from rknnlite.api import RKNNLite  # type: ignore

            self.rknn = RKNNLite()
            ret = self.rknn.load_rknn(self.model_path)
            if ret != 0:
                raise RuntimeError(f"load_rknn failed: {ret}")
            ret = self.rknn.init_runtime()
            if ret != 0:
                raise RuntimeError(f"init_runtime failed: {ret}")
        except Exception:
            # 开发回退：仍可验证管道、规则和 Web 功能。
            self._stub_mode = True
            self.rknn = None
            if not Path(self.model_path).exists():
                Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        """释放 RKNN 运行时资源。"""
        if self.rknn is not None:
            self.rknn.release()

    def infer(self, frame: np.ndarray) -> tuple[list[Detection], np.ndarray]:
        """对单帧图像执行推理。

        将帧缩放到模型输入尺寸，运行 RKNN 模型，并将输出解码为 Detection 对象。
        在存根模式下，返回合成的检测结果用于测试。

        Args:
            frame: 输入的摄像头帧 (H x W x 3)。

        Returns:
            包含 (检测结果列表, 缩放后的帧) 的元组。
        """
        ts_ms = int(time.time() * 1000)
        resized = cv2.resize(frame, (self.input_size, self.input_size))

        if self._stub_mode:
            h, w = resized.shape[:2]
            # 无模型模式下，生成稳定的多目标场景，使完整的
            # 风险/事件/Web 管道可以进行端到端集成测试。
            dets = [
                Detection(
                    ts_ms=ts_ms,
                    class_id=0,
                    class_name="cup",
                    conf=0.71,
                    bbox_xyxy=(int(0.18 * w), int(0.28 * h), int(0.32 * w), int(0.55 * h)),
                ),
                Detection(
                    ts_ms=ts_ms,
                    class_id=1,
                    class_name="phone",
                    conf=0.76,
                    bbox_xyxy=(int(0.31 * w), int(0.32 * h), int(0.43 * w), int(0.56 * h)),
                ),
                Detection(
                    ts_ms=ts_ms,
                    class_id=3,
                    class_name="scissors",
                    conf=0.69,
                    bbox_xyxy=(int(0.74 * w), int(0.55 * h), int(0.88 * w), int(0.82 * h)),
                ),
            ]
            return dets, resized

        input_tensor = np.expand_dims(resized, axis=0)
        outputs = self.rknn.inference(inputs=[input_tensor], data_format="nhwc")
        detections = self._decode(outputs, ts_ms)
        return detections, resized

    def _decode(self, outputs: list[np.ndarray], ts_ms: int) -> List[Detection]:
        """将 RKNN 模型的原始输出解码为 Detection 对象。

        尝试合并分离的输出（独立的框张量和类别张量），然后根据列数
        使用合适的解码器处理每个输出矩阵。

        Args:
            outputs: RKNN 模型输出的张量列表。
            ts_ms: 检测结果的毫秒级时间戳。

        Returns:
            NMS 处理后的 Detection 对象列表。
        """
        if not outputs:
            return []
        self._log_output_summary_once(outputs)

        decoded: list[tuple[int, float, tuple[int, int, int, int]]] = []
        merged = self._try_merge_split_outputs(outputs)
        if merged is not None:
            decoded.extend(self._decode_matrix(merged))
        for out in outputs:
            mat = self._to_prediction_matrix(out)
            if mat is None:
                continue
            decoded.extend(self._decode_matrix(mat))

        if not decoded:
            return []

        return self._apply_nms(decoded, ts_ms)

    def _to_prediction_matrix(self, out: np.ndarray) -> np.ndarray | None:
        """将原始输出张量转换为标准化的预测矩阵。

        处理各种张量形状（2D, 3D, 4D+），通过压缩和重塑为二维矩阵，
        其中行是预测结果，列是特征。必要时进行转置以匹配期望的特征维度。

        Args:
            out: 模型的原始输出张量。

        Returns:
            形状为 (N, F) 的 float32 二维矩阵，F 为期望的特征维度之一。
            如果形状无法识别则返回 None。
        """
        arr = np.asarray(out, dtype=np.float32)
        if arr.size == 0:
            return None
        arr = np.squeeze(arr)
        if arr.ndim < 2:
            return None

        if arr.ndim == 2:
            mat = arr
        elif arr.ndim == 3:
            if arr.shape[-1] <= 64:
                mat = arr.reshape(-1, arr.shape[-1])
            elif arr.shape[0] <= 64:
                mat = np.moveaxis(arr, 0, -1).reshape(-1, arr.shape[0])
            else:
                return None
        else:
            dims = list(arr.shape)
            feature_axes = [i for i, d in enumerate(dims) if d <= 64]
            if not feature_axes:
                return None
            feat_axis = min(feature_axes, key=lambda i: dims[i])
            feat_dim = dims[feat_axis]
            mat = np.moveaxis(arr, feat_axis, -1).reshape(-1, feat_dim)

        feature_dims = {6, 4 + len(self.class_names), 5 + len(self.class_names)}
        if mat.shape[0] in feature_dims and mat.shape[1] > mat.shape[0]:
            mat = mat.T
        elif mat.shape[1] not in feature_dims and mat.shape[0] <= 16 < mat.shape[1]:
            mat = mat.T

        if mat.shape[1] not in feature_dims:
            return None
        return np.ascontiguousarray(mat, dtype=np.float32)

    def _try_merge_split_outputs(self, outputs: list[np.ndarray]) -> np.ndarray | None:
        """尝试合并分离的框和类别输出张量。

        某些模型将框和类别分数输出为独立的张量。
        此方法尝试将它们拼接成单个矩阵。

        Args:
            outputs: 输出张量列表。

        Returns:
            包含框+类别列的合并矩阵，如果无法合并（少于2个矩阵、
            形状不匹配等）则返回 None。
        """
        matrices: list[np.ndarray] = []
        for out in outputs:
            arr = np.asarray(out, dtype=np.float32)
            if arr.size == 0:
                continue
            arr = np.squeeze(arr)
            if arr.ndim == 2:
                matrices.append(arr)
            elif arr.ndim == 3 and arr.shape[-1] <= 64:
                matrices.append(arr.reshape(-1, arr.shape[-1]))

        if len(matrices) < 2:
            return None

        box_mats = [m for m in matrices if m.shape[1] == 4]
        cls_mats = []
        for m in matrices:
            if m.shape[1] == len(self.class_names) + 1:
                cls_mats.append(m)
            elif m.shape[1] == len(self.class_names) and not self._looks_like_box_matrix(m):
                cls_mats.append(m)

        for b in box_mats:
            for c in cls_mats:
                if b is c:
                    continue
                if b.shape[0] != c.shape[0]:
                    continue
                if c.shape[1] == len(self.class_names):
                    return np.concatenate([b, c], axis=1)
                return np.concatenate([b, c[:, :1], c[:, 1:]], axis=1)
        return None

    def _looks_like_box_matrix(self, mat: np.ndarray) -> bool:
        """检查矩阵是否看起来像 4 列的归一化框张量。

        Args:
            mat: 二维 numpy 数组。

        Returns:
            如果矩阵是二维、4列、所有值有限且在 [-2, 2] 范围内
            （归一化坐标）则返回 True。
        """
        if mat.ndim != 2 or mat.shape[1] != 4 or mat.size == 0:
            return False
        if not np.all(np.isfinite(mat)):
            return False
        vmax = float(np.max(np.abs(mat)))
        return vmax <= 2.0

    def _decode_matrix(self, mat: np.ndarray) -> list[tuple[int, float, tuple[int, int, int, int]]]:
        """根据列数将预测矩阵路由到相应的解码器。

        Args:
            mat: 二维预测矩阵。

        Returns:
            (class_id, confidence, bbox_xyxy) 元组的列表。
        """
        cols = mat.shape[1]
        if cols == 6 and np.mean(np.abs(mat[:, 5] - np.round(mat[:, 5])) < 1e-3) > 0.8:
            return self._decode_nx6(mat)
        if cols == 4 + len(self.class_names):
            return self._decode_yolo(mat, with_obj=False)
        if cols == 5 + len(self.class_names):
            return self._decode_yolo(mat, with_obj=True)
        return []

    def _decode_nx6(self, mat: np.ndarray) -> list[tuple[int, float, tuple[int, int, int, int]]]:
        """解码 Nx6 格式的矩阵：[x1, y1, x2, y2, conf, class_id]。

        Args:
            mat: Nx6 预测矩阵。

        Returns:
            (class_id, confidence, bbox_xyxy) 元组的列表。
        """
        out: list[tuple[int, float, tuple[int, int, int, int]]] = []
        for row in mat:
            cls = int(round(float(row[5])))
            if cls < 0 or cls >= len(self.class_names):
                continue
            conf = self._sanitize_conf(float(row[4]))
            if conf < self.conf_threshold:
                continue
            bbox = self._sanitize_bbox(row[:4], prefer_xyxy=True)
            if bbox is None:
                continue
            out.append((cls, conf, bbox))
        return out

    def _decode_yolo(self, mat: np.ndarray, with_obj: bool) -> list[tuple[int, float, tuple[int, int, int, int]]]:
        """解码 YOLO 格式的矩阵：[cx, cy, w, h, (obj), cls_scores...]。

        Args:
            mat: YOLO 格式的预测矩阵。
            with_obj: 矩阵是否包含目标置信度分数列。

        Returns:
            (class_id, confidence, bbox_xyxy) 元组的列表。
        """
        out: list[tuple[int, float, tuple[int, int, int, int]]] = []
        obj_idx = 4 if with_obj else None
        cls_start = 5 if with_obj else 4

        for row in mat:
            cls_scores = row[cls_start:]
            if cls_scores.size != len(self.class_names):
                continue
            cls = int(np.argmax(cls_scores))
            cls_prob = self._sanitize_conf(float(cls_scores[cls]))
            obj_prob = self._sanitize_conf(float(row[obj_idx])) if obj_idx is not None else 1.0
            conf = obj_prob * cls_prob
            if conf < self.conf_threshold:
                continue
            bbox = self._sanitize_bbox(row[:4], prefer_xyxy=False)
            if bbox is None:
                continue
            out.append((cls, conf, bbox))
        return out

    def _sanitize_conf(self, value: float) -> float:
        """将置信度值归一化到 [0, 1] 范围。

        如果值不在 [0, 1] 范围内，则将其视为 logit 并通过 sigmoid 函数转换。

        Args:
            value: 原始置信度或 logit 分数。

        Returns:
            归一化后的置信度，范围 [0, 1]。
        """
        if not np.isfinite(value):
            return 0.0
        val = float(value)
        if val < 0.0 or val > 1.0:
            # 某些导出的模型输出 logit；转换为概率。
            val = 1.0 / (1.0 + np.exp(-val))
        return float(min(max(val, 0.0), 1.0))

    def _sanitize_bbox(self, box: np.ndarray, prefer_xyxy: bool) -> tuple[int, int, int, int] | None:
        """验证并转换原始边界框为像素坐标。

        处理归一化坐标（值 <= 2.0），通过缩放到 input_size 实现。
        根据偏好尝试 xyxy 和 xywh 两种格式。

        Args:
            box: 包含4个值的原始边界框数组。
            prefer_xyxy: 是否优先尝试 xyxy 格式。

        Returns:
            (x1, y1, x2, y2) 像素坐标，无效时返回 None。
        """
        vals = np.asarray(box, dtype=np.float32).reshape(-1)
        if vals.size != 4 or not np.all(np.isfinite(vals)):
            return None
        vals = vals.copy()
        if float(np.max(np.abs(vals))) <= 2.0:
            vals *= float(self.input_size)

        if prefer_xyxy:
            candidates = [self._clip_xyxy(*vals), self._clip_xywh(*vals)]
        else:
            candidates = [self._clip_xywh(*vals), self._clip_xyxy(*vals)]

        for c in candidates:
            if c is not None:
                return c

        self._warn_decode(f"drop invalid bbox raw={vals.tolist()}")
        return None

    def _clip_xyxy(self, x1: float, y1: float, x2: float, y2: float) -> tuple[int, int, int, int] | None:
        """裁剪 xyxy 格式的框并转换为整数像素坐标。

        Args:
            x1: 左坐标。
            y1: 上坐标。
            x2: 右坐标。
            y2: 下坐标。

        Returns:
            裁剪到有效范围内的 (x1, y1, x2, y2)，无效时返回 None。
        """
        return self._clip_box(x1, y1, x2, y2)

    def _clip_xywh(self, cx: float, cy: float, w: float, h: float) -> tuple[int, int, int, int] | None:
        """将 xywh（中心）格式转换为 xyxy 并裁剪到有效范围。

        Args:
            cx: 中心 x 坐标。
            cy: 中心 y 坐标。
            w: 框宽度。
            h: 框高度。

        Returns:
            裁剪到有效范围内的 (x1, y1, x2, y2)，无效时返回 None。
        """
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return self._clip_box(x1, y1, x2, y2)

    def _clip_box(self, x1: float, y1: float, x2: float, y2: float) -> tuple[int, int, int, int] | None:
        """将框坐标裁剪到有效范围 [0, input_size-1]。

        Args:
            x1: 左坐标。
            y1: 上坐标。
            x2: 右坐标。
            y2: 下坐标。

        Returns:
            裁剪到有效范围内的整数 (x1, y1, x2, y2)，
            框面积为零或负值时返回 None。
        """
        limit = max(self.input_size - 1, 1)
        ix1 = int(round(min(max(x1, 0.0), limit)))
        iy1 = int(round(min(max(y1, 0.0), limit)))
        ix2 = int(round(min(max(x2, 0.0), limit)))
        iy2 = int(round(min(max(y2, 0.0), limit)))
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        return ix1, iy1, ix2, iy2

    def _apply_nms(self, rows: list[tuple[int, float, tuple[int, int, int, int]]], ts_ms: int) -> List[Detection]:
        """按类别应用非极大值抑制，并转换为 Detection 对象。

        按类别分组检测结果，对每组运行 OpenCV NMS，然后返回最终的 Detection 对象列表。

        Args:
            rows: (class_id, confidence, bbox_xyxy) 元组的列表。
            ts_ms: 检测结果的毫秒级时间戳。

        Returns:
            NMS 过滤后的 Detection 对象列表。
        """
        if not rows:
            return []

        kept: list[tuple[int, float, tuple[int, int, int, int]]] = []
        by_class: dict[int, list[tuple[float, tuple[int, int, int, int]]]] = {}
        for cls, conf, box in rows:
            by_class.setdefault(cls, []).append((conf, box))

        for cls, items in by_class.items():
            boxes = [[b[0], b[1], max(b[2] - b[0], 1), max(b[3] - b[1], 1)] for _, b in items]
            scores = [float(s) for s, _ in items]
            keep = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.nms_threshold)
            if keep is None or len(keep) == 0:
                continue
            for idx in keep.flatten().tolist():
                conf, box = items[idx]
                if 0.0 <= conf <= 1.0:
                    kept.append((cls, conf, box))

        return [
            Detection(
                ts_ms=ts_ms,
                class_id=cls,
                class_name=self.class_names[cls],
                conf=float(conf),
                bbox_xyxy=box,
            )
            for cls, conf, box in kept
        ]

    def _log_output_summary_once(self, outputs: list[np.ndarray]) -> None:
        """记录每个输出张量的形状、数据类型和值范围（仅一次）。

        Args:
            outputs: RKNN 模型的输出张量列表。
        """
        if self._output_shape_logged:
            return
        summary = []
        for i, out in enumerate(outputs):
            arr = np.asarray(out)
            if arr.size == 0:
                summary.append(f"o{i}:shape={tuple(arr.shape)} empty")
                continue
            summary.append(
                f"o{i}:shape={tuple(arr.shape)} dtype={arr.dtype} min={float(np.min(arr)):.4f} max={float(np.max(arr)):.4f}"
            )
        self._log.info("RKNN output summary: %s", " | ".join(summary))
        self._output_shape_logged = True

    def _warn_decode(self, msg: str) -> None:
        """记录解码警告，最多限制输出5条消息。

        Args:
            msg: 要记录的警告消息。
        """
        if self._decode_warn_count >= 5:
            return
        self._decode_warn_count += 1
        self._log.warning("decode warning: %s", msg)