"""Python ctypes 绑定，调用 C++ RKNN 推理库。"""

from __future__ import annotations

import ctypes
import os
import threading
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future
from typing import List

import cv2
import numpy as np

from app.types import Detection


# C 结构体定义
class C_Detection(ctypes.Structure):
    _fields_ = [
        ("class_id", ctypes.c_int),
        ("confidence", ctypes.c_float),
        ("x1", ctypes.c_int),
        ("y1", ctypes.c_int),
        ("x2", ctypes.c_int),
        ("y2", ctypes.c_int),
    ]


class C_DetectionResult(ctypes.Structure):
    _fields_ = [
        ("dets", C_Detection * 128),
        ("count", ctypes.c_int),
    ]


class NativeInferenceEngine:
    """使用 C++ 推理库的高性能引擎，支持异步并行推理。"""

    def __init__(
        self,
        model_path: str,
        class_names: list[str],
        conf_threshold: float = 0.25,
        nms_threshold: float = 0.45,
        input_size: int = 640,
        n_workers: int = 3,
    ) -> None:
        self.model_path = model_path
        self.class_names = list(class_names)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.n_workers = n_workers
        self._engine = None
        self._lib = None
        self._stub_mode = False
        self._pool: ThreadPoolExecutor | None = None

    def open(self) -> None:
        """加载 C++ 推理库和模型。"""
        lib_path = self._find_library()
        if lib_path is None:
            print("WARN: librknn_infer.so not found, falling back to stub mode")
            self._stub_mode = True
            return

        self._lib = ctypes.CDLL(lib_path)

        self._lib.rknn_engine_create.restype = ctypes.c_void_p
        self._lib.rknn_engine_create.argtypes = [
            ctypes.c_char_p, ctypes.c_int, ctypes.c_float, ctypes.c_float
        ]

        self._lib.rknn_engine_infer.restype = ctypes.c_int
        self._lib.rknn_engine_infer.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(C_DetectionResult)
        ]

        self._lib.rknn_engine_destroy.restype = None
        self._lib.rknn_engine_destroy.argtypes = [ctypes.c_void_p]

        self._engine = self._lib.rknn_engine_create(
            self.model_path.encode("utf-8"),
            self.input_size,
            self.conf_threshold,
            self.nms_threshold,
        )

        if not self._engine:
            print("WARN: rknn_engine_create failed, falling back to stub mode")
            self._stub_mode = True
            return

        self._pool = ThreadPoolExecutor(max_workers=self.n_workers)

    def _find_library(self) -> str | None:
        """查找 librknn_infer.so。"""
        candidates = [
            Path("/opt/desk-safety/native/librknn_infer.so"),
            Path(__file__).parent.parent.parent / "native" / "librknn_infer.so",
            Path("native/librknn_infer.so"),
        ]
        for p in candidates:
            if p.exists():
                return str(p)
        return None

    def close(self) -> None:
        """释放引擎。"""
        if self._pool:
            self._pool.shutdown(wait=False)
            self._pool = None
        if self._engine and self._lib:
            self._lib.rknn_engine_destroy(self._engine)
            self._engine = None

    def infer(self, frame: np.ndarray) -> tuple[list[Detection], np.ndarray]:
        """同步推理单帧图像。

        Args:
            frame: BGR 格式的图像 (H x W x 3)。

        Returns:
            (检测结果列表, resize 后的图像)。
        """
        if self._stub_mode:
            return self._stub_infer(frame)

        resized = cv2.resize(frame, (self.input_size, self.input_size))
        dets = self._infer_frame(resized)
        return dets, resized

    def _infer_frame(self, resized: np.ndarray) -> list[Detection]:
        """对已 resize 的帧执行推理。"""
        ptr = resized.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        result = C_DetectionResult()

        ret = self._lib.rknn_engine_infer(
            self._engine, ptr, self.input_size, self.input_size,
            ctypes.byref(result)
        )

        ts_ms = int(time.time() * 1000)
        dets = []
        if ret == 0:
            for i in range(result.count):
                d = result.dets[i]
                cid = d.class_id
                if 0 <= cid < len(self.class_names):
                    dets.append(Detection(
                        ts_ms=ts_ms,
                        class_id=cid,
                        class_name=self.class_names[cid],
                        conf=float(d.confidence),
                        bbox_xyxy=(int(d.x1), int(d.y1), int(d.x2), int(d.y2)),
                    ))
        return dets

    def _stub_infer(self, frame: np.ndarray) -> tuple[list[Detection], np.ndarray]:
        """Stub 模式，返回空结果。"""
        resized = cv2.resize(frame, (self.input_size, self.input_size))
        return [], resized

    def infer_async(self, resized_frame: np.ndarray) -> Future:
        """异步推理已 resize 的帧。

        Args:
            resized_frame: 已 resize 到 input_size 的帧。

        Returns:
            Future 对象，result() 返回检测结果列表。
        """
        if self._pool is None:
            raise RuntimeError("engine not opened")
        return self._pool.submit(self._infer_frame, resized_frame)
