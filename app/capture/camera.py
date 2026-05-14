from __future__ import annotations

import cv2
import numpy as np


class CameraCapture:
    def __init__(self, device: str, width: int, height: int, crop_left: bool = False) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.crop_left = crop_left
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self.cap.isOpened():
            raise RuntimeError(f"failed to open camera device: {self.device}")

    def read(self) -> np.ndarray:
        if self.cap is None:
            raise RuntimeError("camera not opened")
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        if self.crop_left:
            h, w = frame.shape[:2]
            frame = frame[:, :w // 2]
            if not hasattr(self, '_crop_logged'):
                import logging
                logging.getLogger("desk-safety.camera").info("crop_left enabled: %dx%d -> %dx%d", w, h, frame.shape[1], frame.shape[0])
                self._crop_logged = True
        return frame

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
