#!/usr/bin/env python3
"""Profile YOLO11 inference pipeline to find bottlenecks."""

import sys
import time
import cv2
import numpy as np

sys.path.insert(0, "/opt/desk-safety")
from app.config import load_settings
from app.capture.camera import CameraCapture
from rknnlite.api import RKNNLite


def main():
    cfg = load_settings("./configs/config_yolo11m.yaml")
    cam = CameraCapture(cfg.camera_device, cfg.camera_width, cfg.camera_height)
    cam.open()

    rknn = RKNNLite()
    rknn.load_rknn(cfg.model_path)
    rknn.init_runtime()

    # Warmup
    frame = cam.read()
    resized = cv2.resize(frame, (cfg.input_size, cfg.input_size))
    input_tensor = np.expand_dims(resized, axis=0)
    rknn.inference(inputs=[input_tensor], data_format="nhwc")

    n = 30
    t_cam, t_resize, t_infer, t_decode = 0, 0, 0, 0

    for i in range(n):
        t0 = time.perf_counter()
        frame = cam.read()
        t1 = time.perf_counter()
        resized = cv2.resize(frame, (cfg.input_size, cfg.input_size))
        t2 = time.perf_counter()
        input_tensor = np.expand_dims(resized, axis=0)
        outputs = rknn.inference(inputs=[input_tensor], data_format="nhwc")
        t3 = time.perf_counter()

        t_cam += t1 - t0
        t_resize += t2 - t1
        t_infer += t3 - t2

    print(f"Average over {n} frames:")
    print(f"  Camera read:  {t_cam/n*1000:.1f} ms")
    print(f"  Resize:       {t_resize/n*1000:.1f} ms")
    print(f"  RKNN infer:   {t_infer/n*1000:.1f} ms")
    print(f"  Total:        {(t_cam+t_resize+t_infer)/n*1000:.1f} ms")
    print(f"  FPS:          {n/(t_cam+t_resize+t_infer):.1f}")

    rknn.release()
    cam.close()


if __name__ == "__main__":
    main()
