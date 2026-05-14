#!/usr/bin/env python3
"""Test multi-process inference with YOLO11m."""

import sys
import time
import json

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

    total_dets = 0
    class_hist = {}

    start = time.time()
    frames = 30
    for i in range(1, frames + 1):
        frame = cam.read()
        dets, _ = engine.infer(frame)
        total_dets += len(dets)
        for d in dets:
            class_hist[d.class_name] = class_hist.get(d.class_name, 0) + 1
        if i % 10 == 0:
            print(f"[progress] {i}/{frames} frames, dets={total_dets}")

    elapsed = time.time() - start
    summary = {
        "model": "yolo11m",
        "npu_threads": cfg.npu_threads,
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
