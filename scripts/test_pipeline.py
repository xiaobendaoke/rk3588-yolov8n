#!/usr/bin/env python3
"""Test async pipeline inference with YOLO11m.

流水线模式：
- 线程1: 读摄像头，提交到 worker
- 3个 worker 进程: 并行推理
- 主线程: 收集结果
"""

import sys
import time
import json
import threading
from collections import deque

sys.path.insert(0, "/opt/desk-safety")
from app.config import load_settings
from app.capture.camera import CameraCapture
from app.infer.multi_process import MultiProcessEngine


def main():
    cfg = load_settings("./configs/config_yolo11m.yaml")
    cam = CameraCapture(cfg.camera_device, cfg.camera_width, cfg.camera_height, cfg.camera_crop_left)

    print(f"Creating MultiProcessEngine with {cfg.npu_threads} workers...")
    engine = MultiProcessEngine(
        model_path=cfg.model_path,
        class_names=cfg.class_names,
        conf_threshold=cfg.conf_threshold,
        nms_threshold=cfg.nms_threshold,
        input_size=cfg.input_size,
        n_workers=cfg.npu_threads,
    )

    cam.open()
    engine.open()

    # 流水线：连续提交多个帧，然后收集结果
    total_dets = 0
    class_hist = {}
    frames = 30
    pending = deque()  # 存储 (frame_id, worker_idx)
    results = {}

    start = time.time()

    # 预热：提交前 N 个帧
    for i in range(min(cfg.npu_threads, frames)):
        frame = cam.read()
        engine._frame_counter += 1
        fid = engine._frame_counter
        idx = fid % cfg.npu_threads
        engine._input_queues[idx].put_nowait((fid, frame))
        pending.append((fid, idx))

    # 主循环：提交新帧 + 收集旧结果
    submitted = len(pending)
    while submitted < frames or pending:
        # 提交新帧
        if submitted < frames:
            frame = cam.read()
            engine._frame_counter += 1
            fid = engine._frame_counter
            idx = fid % cfg.npu_threads
            engine._input_queues[idx].put_nowait((fid, frame))
            pending.append((fid, idx))
            submitted += 1

        # 收集已完成的结果
        if pending:
            fid, idx = pending[0]
            try:
                result = engine._output_queues[idx].get(timeout=0.01)
                if result[0] == fid:
                    pending.popleft()
                    dets = result[1]
                    total_dets += len(dets)
                    for d in dets:
                        class_hist[d.class_name] = class_hist.get(d.class_name, 0) + 1
            except Exception:
                pass

    elapsed = time.time() - start
    summary = {
        "model": "yolo11m",
        "npu_threads": cfg.npu_threads,
        "mode": "async_pipeline",
        "frames": frames,
        "elapsed_sec": round(elapsed, 3),
        "avg_fps": round(frames / elapsed, 2),
        "total_detections": total_dets,
        "avg_detections_per_frame": round(total_dets / frames, 3),
        "class_hist": class_hist,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    engine.close()
    cam.close()


if __name__ == "__main__":
    main()
