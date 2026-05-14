#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from collections import Counter

from app.capture.camera import CameraCapture
from app.config import load_settings
from app.infer.engine import InferenceEngine


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample inference for decode validation")
    p.add_argument("--config", default="./configs/config.yaml")
    p.add_argument("--frames", type=int, default=100)
    p.add_argument("--interval-ms", type=int, default=80)
    return p.parse_args()


def is_valid_box(box: tuple[int, int, int, int], limit: int) -> bool:
    x1, y1, x2, y2 = box
    return 0 <= x1 < x2 <= limit and 0 <= y1 < y2 <= limit


def main() -> int:
    args = parse_args()
    cfg = load_settings(args.config)
    cam = CameraCapture(cfg.camera_device, cfg.camera_width, cfg.camera_height)
    infer = InferenceEngine(
        cfg.model_path,
        cfg.class_names,
        cfg.conf_threshold,
        cfg.nms_threshold,
        cfg.input_size,
    )

    total_dets = 0
    invalid_conf = 0
    invalid_bbox = 0
    class_hist: Counter[str] = Counter()

    cam.open()
    infer.open()
    start = time.time()
    try:
        for i in range(1, args.frames + 1):
            frame = cam.read()
            dets, _ = infer.infer(frame)
            total_dets += len(dets)
            for d in dets:
                class_hist[d.class_name] += 1
                if not (0.0 <= d.conf <= 1.0):
                    invalid_conf += 1
                if not is_valid_box(d.bbox_xyxy, cfg.input_size - 1):
                    invalid_bbox += 1
            if i % 10 == 0:
                print(f"[progress] {i}/{args.frames} frames, dets={total_dets}")
            time.sleep(max(args.interval_ms, 1) / 1000.0)
    finally:
        infer.close()
        cam.close()

    elapsed = max(time.time() - start, 1e-6)
    summary = {
        "frames": args.frames,
        "elapsed_sec": round(elapsed, 3),
        "avg_fps": round(args.frames / elapsed, 2),
        "total_detections": total_dets,
        "avg_detections_per_frame": round(total_dets / max(args.frames, 1), 3),
        "invalid_conf_count": invalid_conf,
        "invalid_bbox_count": invalid_bbox,
        "class_hist": dict(class_hist),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if (invalid_conf > 0 or invalid_bbox > 0) else 0


if __name__ == "__main__":
    raise SystemExit(main())
