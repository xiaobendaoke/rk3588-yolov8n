#!/usr/bin/env python3
"""Test native C++ inference engine."""

import sys
import time
import json

sys.path.insert(0, "/opt/desk-safety")
from app.config import load_settings
from app.capture.camera import CameraCapture
from app.infer.native_engine import NativeInferenceEngine


def main():
    cfg = load_settings("./configs/config_yolo11m.yaml")
    cam = CameraCapture(cfg.camera_device, cfg.camera_width, cfg.camera_height, cfg.camera_crop_left)

    engine = NativeInferenceEngine(
        model_path=cfg.model_path,
        class_names=cfg.class_names,
        conf_threshold=cfg.conf_threshold,
        nms_threshold=cfg.nms_threshold,
        input_size=cfg.input_size,
    )

    cam.open()
    engine.open()

    if engine._stub_mode:
        print("ERROR: Engine in stub mode, C++ library not loaded!")
        return

    total_dets = 0
    class_hist = {}

    # Warmup
    frame = cam.read()
    engine.infer(frame)

    start = time.time()
    frames = 50
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
        "engine": "native_cpp",
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
