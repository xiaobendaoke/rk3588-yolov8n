#!/usr/bin/env python3
"""Debug: dump RKNN model output shapes for YOLO11."""

import sys
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
    ret = rknn.load_rknn(cfg.model_path)
    print(f"load_rknn: {ret}")
    ret = rknn.init_runtime()
    print(f"init_runtime: {ret}")

    frame = cam.read()
    print(f"frame shape: {frame.shape}")

    import cv2
    resized = cv2.resize(frame, (cfg.input_size, cfg.input_size))
    input_tensor = np.expand_dims(resized, axis=0)
    print(f"input_tensor shape: {input_tensor.shape}")

    outputs = rknn.inference(inputs=[input_tensor], data_format="nhwc")
    print(f"number of outputs: {len(outputs)}")
    for i, out in enumerate(outputs):
        arr = np.asarray(out)
        print(f"  output[{i}]: shape={arr.shape}, dtype={arr.dtype}, min={float(arr.min()):.4f}, max={float(arr.max()):.4f}, mean={float(arr.mean()):.4f}")

    rknn.release()
    cam.close()


if __name__ == "__main__":
    main()
