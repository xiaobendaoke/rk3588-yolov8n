"""多线程推理引擎，利用 RK3588 三个 NPU 核心并行推理。"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import List, Sequence

import cv2
import numpy as np

from app.infer.engine import InferenceEngine
from app.types import Detection


class MultiThreadEngine:
    """流水线多线程推理引擎。

    使用 N 个 RKNN 实例 + 1 个提交线程 + 1 个结果收集线程：
    - 提交线程：从帧队列取帧，轮询提交到 worker
    - Worker：各自独立执行 RKNN 推理
    - 收集线程：收集结果，更新最新检测

    Attributes:
        model_path: RKNN 模型文件路径。
        n_workers: worker 数量（建议设为 3，匹配 NPU 核心数）。
    """

    def __init__(
        self,
        model_path: str,
        class_names: Sequence[str],
        conf_threshold: float,
        nms_threshold: float,
        input_size: int,
        n_workers: int = 3,
    ) -> None:
        self.model_path = model_path
        self.class_names = list(class_names)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.n_workers = n_workers
        self._log = logging.getLogger("desk-safety.multi-infer")

        self._engines: list[InferenceEngine] = []
        self._stub_mode = False

        # 流水线队列
        self._frame_queue: deque[tuple[int, np.ndarray]] = deque(maxlen=2)
        self._result_queue: deque[tuple[list[Detection], np.ndarray]] = deque(maxlen=2)
        self._running = False
        self._submit_thread: threading.Thread | None = None
        self._collect_thread: threading.Thread | None = None
        self._worker_events: list[threading.Event] = []
        self._worker_inputs: list[tuple[int, np.ndarray] | None] = []
        self._worker_outputs: list[tuple[list[Detection], np.ndarray] | None] = []
        self._frame_counter = 0

    def open(self) -> None:
        """创建多个推理引擎实例并启动流水线线程。"""
        for i in range(self.n_workers):
            engine = InferenceEngine(
                model_path=self.model_path,
                class_names=self.class_names,
                conf_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold,
                input_size=self.input_size,
            )
            engine.open()
            self._engines.append(engine)
            if engine._stub_mode:
                self._stub_mode = True

        if not self._stub_mode:
            self._log.info("Created %d RKNN workers", len(self._engines))

        # 初始化 worker 事件和输入/输出缓冲
        self._worker_events = [threading.Event() for _ in range(self.n_workers)]
        self._worker_inputs = [None] * self.n_workers
        self._worker_outputs = [None] * self.n_workers

        self._running = True

        # 启动 worker 线程（每个 worker 有自己的循环）
        for i in range(self.n_workers):
            t = threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            t.start()

    def _worker_loop(self, worker_id: int) -> None:
        """Worker 线程主循环：等待输入，执行推理，输出结果。

        Args:
            worker_id: worker 编号。
        """
        engine = self._engines[worker_id]
        event = self._worker_events[worker_id]

        while self._running:
            # 等待有输入
            event.wait(timeout=0.1)
            if not event.is_set():
                continue

            inp = self._worker_inputs[worker_id]
            if inp is None:
                event.clear()
                continue

            frame_id, frame = inp
            try:
                dets, resized = engine.infer(frame)
                self._worker_outputs[worker_id] = (frame_id, dets, resized)
            except Exception as e:
                self._log.warning("Worker %d error: %s", worker_id, e)
                self._worker_outputs[worker_id] = (frame_id, [], frame)

            event.clear()

    def close(self) -> None:
        """释放所有推理引擎和线程。"""
        self._running = False
        for engine in self._engines:
            engine.close()
        self._engines.clear()

    def infer(self, frame: np.ndarray) -> tuple[list[Detection], np.ndarray]:
        """同步推理：提交帧到 worker，等待结果。

        Args:
            frame: 输入的摄像头帧 (H x W x 3)。

        Returns:
            包含 (检测结果列表, 缩放后的帧) 的元组。
        """
        if self._stub_mode:
            return self._engines[0].infer(frame)

        self._frame_counter += 1
        frame_id = self._frame_counter

        # 找空闲 worker
        best_idx = self._find_idle_worker()

        # 提交给 worker
        self._worker_inputs[best_idx] = (frame_id, frame)
        self._worker_events[best_idx].set()

        # 等待结果
        while True:
            out = self._worker_outputs[best_idx]
            if out is not None and out[0] == frame_id:
                self._worker_outputs[best_idx] = None
                return out[1], out[2]
            time.sleep(0.001)

    def _find_idle_worker(self) -> int:
        """找到第一个空闲的 worker。

        Returns:
            空闲 worker 的索引。
        """
        # 先找没有待处理输入的
        for i in range(self.n_workers):
            if not self._worker_events[i].is_set() and self._worker_outputs[i] is None:
                return i
        # 都忙，等最老的那个
        return 0
