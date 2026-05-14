from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np

from app.types import Detection


class InferenceEngine:
    def __init__(
        self,
        model_path: str,
        class_names: Sequence[str],
        conf_threshold: float,
        nms_threshold: float,
        input_size: int,
    ) -> None:
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
            # Development fallback: still lets pipeline, rules, and web be verified.
            self._stub_mode = True
            self.rknn = None
            if not Path(self.model_path).exists():
                Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        if self.rknn is not None:
            self.rknn.release()

    def infer(self, frame: np.ndarray) -> tuple[list[Detection], np.ndarray]:
        ts_ms = int(time.time() * 1000)
        resized = cv2.resize(frame, (self.input_size, self.input_size))

        if self._stub_mode:
            h, w = resized.shape[:2]
            # In no-model mode, emit a stable multi-object scene so the full
            # risk/event/web pipeline can be integration-tested end-to-end.
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
        if mat.ndim != 2 or mat.shape[1] != 4 or mat.size == 0:
            return False
        if not np.all(np.isfinite(mat)):
            return False
        vmax = float(np.max(np.abs(mat)))
        return vmax <= 2.0

    def _decode_matrix(self, mat: np.ndarray) -> list[tuple[int, float, tuple[int, int, int, int]]]:
        cols = mat.shape[1]
        if cols == 6 and np.mean(np.abs(mat[:, 5] - np.round(mat[:, 5])) < 1e-3) > 0.8:
            return self._decode_nx6(mat)
        if cols == 4 + len(self.class_names):
            return self._decode_yolo(mat, with_obj=False)
        if cols == 5 + len(self.class_names):
            return self._decode_yolo(mat, with_obj=True)
        return []

    def _decode_nx6(self, mat: np.ndarray) -> list[tuple[int, float, tuple[int, int, int, int]]]:
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
        if not np.isfinite(value):
            return 0.0
        val = float(value)
        if val < 0.0 or val > 1.0:
            # Some exported models output logits; convert to probability.
            val = 1.0 / (1.0 + np.exp(-val))
        return float(min(max(val, 0.0), 1.0))

    def _sanitize_bbox(self, box: np.ndarray, prefer_xyxy: bool) -> tuple[int, int, int, int] | None:
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
        return self._clip_box(x1, y1, x2, y2)

    def _clip_xywh(self, cx: float, cy: float, w: float, h: float) -> tuple[int, int, int, int] | None:
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return self._clip_box(x1, y1, x2, y2)

    def _clip_box(self, x1: float, y1: float, x2: float, y2: float) -> tuple[int, int, int, int] | None:
        limit = max(self.input_size - 1, 1)
        ix1 = int(round(min(max(x1, 0.0), limit)))
        iy1 = int(round(min(max(y1, 0.0), limit)))
        ix2 = int(round(min(max(x2, 0.0), limit)))
        iy2 = int(round(min(max(y2, 0.0), limit)))
        if ix2 <= ix1 or iy2 <= iy1:
            return None
        return ix1, iy1, ix2, iy2

    def _apply_nms(self, rows: list[tuple[int, float, tuple[int, int, int, int]]], ts_ms: int) -> List[Detection]:
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
        if self._decode_warn_count >= 5:
            return
        self._decode_warn_count += 1
        self._log.warning("decode warning: %s", msg)
