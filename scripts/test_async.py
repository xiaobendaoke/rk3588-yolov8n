#!/usr/bin/env python3
"""Test async pipeline with native C++ engine.

流水线模式：同时提交多个帧到不同的 NPU 核心
"""

import sys
import time
import json
import ctypes
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/opt/desk-safety")
import cv2
import numpy as np
from app.config import load_settings
from app.capture.camera import CameraCapture


# C 结构体
class C_Detection(ctypes.Structure):
    _fields_ = [
        ("class_id", ctypes.c_int),
        ("confidence", ctypes.c_float),
        ("x1", ctypes.c_int), ("y1", ctypes.c_int),
        ("x2", ctypes.c_int), ("y2", ctypes.c_int),
    ]

class C_DetectionResult(ctypes.Structure):
    _fields_ = [("dets", C_Detection * 128), ("count", ctypes.c_int)]


def main():
    cfg = load_settings("./configs/config_yolo11m.yaml")
    cam = CameraCapture(cfg.camera_device, cfg.camera_width, cfg.camera_height, cfg.camera_crop_left)
    cam.open()

    # 加载 C++ 库
    lib = ctypes.CDLL("/opt/desk-safety/native/librknn_infer.so")
    lib.rknn_engine_create.restype = ctypes.c_void_p
    lib.rknn_engine_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_float, ctypes.c_float]
    lib.rknn_engine_infer.restype = ctypes.c_int
    lib.rknn_engine_infer.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int, ctypes.c_int, ctypes.POINTER(C_DetectionResult)
    ]
    lib.rknn_engine_destroy.restype = None
    lib.rknn_engine_destroy.argtypes = [ctypes.c_void_p]

    engine = lib.rknn_engine_create(cfg.model_path.encode(), 640, cfg.conf_threshold, cfg.nms_threshold)
    print(f"Engine created (3 workers)")

    class_names = cfg.class_names
    total_dets = 0
    class_hist = {}
    frames = 50

    def infer_one(resized_frame):
        """单帧推理"""
        ptr = resized_frame.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        result = C_DetectionResult()
        ret = lib.rknn_engine_infer(engine, ptr, 640, 640, ctypes.byref(result))
        dets = []
        if ret == 0:
            for i in range(result.count):
                d = result.dets[i]
                cid = d.class_id
                if 0 <= cid < len(class_names):
                    dets.append((cid, class_names[cid], float(d.confidence)))
        return dets

    # 预处理所有帧
    print("Pre-processing frames...")
    resized_frames = []
    for i in range(frames):
        frame = cam.read()
        resized = cv2.resize(frame, (640, 640))
        resized_frames.append(resized)

    # 测试1: 顺序推理 (baseline)
    print("\n--- Sequential inference ---")
    start = time.time()
    for i in range(frames):
        dets = infer_one(resized_frames[i])
        total_dets += len(dets)
        for cid, cname, conf in dets:
            class_hist[cname] = class_hist.get(cname, 0) + 1
    elapsed = time.time() - start
    print(f"  {frames} frames in {elapsed:.2f}s = {frames/elapsed:.2f} FPS, {total_dets} detections")

    # 测试2: 多线程并行推理
    print("\n--- Parallel inference (3 threads) ---")
    total_dets2 = 0
    class_hist2 = {}
    start = time.time()

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(infer_one, resized_frames[i]) for i in range(frames)]
        for f in as_completed(futures):
            dets = f.result()
            total_dets2 += len(dets)
            for cid, cname, conf in dets:
                class_hist2[cname] = class_hist2.get(cname, 0) + 1

    elapsed2 = time.time() - start
    print(f"  {frames} frames in {elapsed2:.2f}s = {frames/elapsed2:.2f} FPS, {total_dets2} detections")
    print(f"  Speedup: {elapsed/elapsed2:.2f}x")

    lib.rknn_engine_destroy(engine)
    cam.close()


if __name__ == "__main__":
    main()
