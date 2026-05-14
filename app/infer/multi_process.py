"""多进程推理引擎，利用 RK3588 三个 NPU 核心并行推理。

使用 multiprocessing 绕过 Python GIL，每个进程有独立的 RKNNLite 实例。
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from typing import List, Sequence

import cv2
import numpy as np

from app.types import Detection


def _worker_process(
    model_path: str,
    class_names: list[str],
    conf_threshold: float,
    nms_threshold: float,
    input_size: int,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    ready_event: mp.Event,
) -> None:
    """Worker 进程：加载模型，循环等待输入，推理后输出结果。"""
    from app.infer.engine import InferenceEngine

    engine = InferenceEngine(
        model_path=model_path,
        class_names=class_names,
        conf_threshold=conf_threshold,
        nms_threshold=nms_threshold,
        input_size=input_size,
    )
    engine.open()
    ready_event.set()

    while True:
        item = input_queue.get()
        if item is None:
            break

        frame_id, frame = item
        try:
            dets, resized = engine.infer(frame)
            output_queue.put((frame_id, dets, resized))
        except Exception:
            output_queue.put((frame_id, [], frame))

    engine.close()


class MultiProcessEngine:
    """多进程推理引擎，创建 N 个独立进程绕过 Python GIL。"""

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
        self._log = logging.getLogger("desk-safety.multi-proc")

        self._input_queues: list[mp.Queue] = []
        self._output_queues: list[mp.Queue] = []
        self._processes: list[mp.Process] = []
        self._frame_counter = 0
        self._stub_mode = False

    def open(self) -> None:
        """启动 N 个 worker 进程。"""
        ctx = mp.get_context("spawn")

        for i in range(self.n_workers):
            in_q: mp.Queue = ctx.Queue(maxsize=2)
            out_q: mp.Queue = ctx.Queue(maxsize=2)
            ready = ctx.Event()

            p = ctx.Process(
                target=_worker_process,
                args=(
                    self.model_path,
                    self.class_names,
                    self.conf_threshold,
                    self.nms_threshold,
                    self.input_size,
                    in_q,
                    out_q,
                    ready,
                ),
                daemon=True,
            )
            p.start()
            ready.wait(timeout=30)
            self._input_queues.append(in_q)
            self._output_queues.append(out_q)
            self._processes.append(p)

        self._log.info("Started %d worker processes", self.n_workers)

    def close(self) -> None:
        """停止所有 worker 进程。"""
        for q in self._input_queues:
            try:
                q.put_nowait(None)
            except Exception:
                pass
        for p in self._processes:
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()
        self._processes.clear()
        self._input_queues.clear()
        self._output_queues.clear()

    def infer(self, frame: np.ndarray) -> tuple[list[Detection], np.ndarray]:
        """提交推理请求并等待结果。

        Args:
            frame: 输入的摄像头帧。

        Returns:
            (检测结果列表, 缩放后的帧)。
        """
        self._frame_counter += 1
        frame_id = self._frame_counter

        # 轮询选择 worker
        idx = frame_id % self.n_workers

        # 提交
        try:
            self._input_queues[idx].put_nowait((frame_id, frame))
        except Exception:
            # 队列满，跳过
            return [], frame

        # 等待结果
        while True:
            try:
                result = self._output_queues[idx].get(timeout=5)
                if result[0] == frame_id:
                    return result[1], result[2]
            except Exception:
                return [], frame
