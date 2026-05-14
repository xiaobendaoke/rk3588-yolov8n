from __future__ import annotations

import cv2
import numpy as np


class CameraCapture:
    """OpenCV VideoCapture 的封装，用于读取摄像头帧。

    支持左半帧裁剪（用于双目摄像头）。

    Attributes:
        device: 摄像头设备路径（如 "/dev/video0"）。
        width: 期望的帧宽度。
        height: 期望的帧高度。
        crop_left: 是否只保留帧的左半部分。
        cap: 底层的 OpenCV VideoCapture 实例，未打开时为 None。
    """

    def __init__(self, device: str, width: int, height: int, crop_left: bool = False) -> None:
        """初始化摄像头捕获封装。

        Args:
            device: 摄像头设备路径。
            width: 期望的帧宽度。
            height: 期望的帧高度。
            crop_left: 是否裁剪左半帧。
        """
        self.device = device
        self.width = width
        self.height = height
        self.crop_left = crop_left
        self.cap: cv2.VideoCapture | None = None

    def open(self) -> None:
        """打开摄像头设备并配置分辨率。

        Raises:
            RuntimeError: 摄像头设备无法打开时抛出。
        """
        self.cap = cv2.VideoCapture(self.device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self.cap.isOpened():
            raise RuntimeError(f"failed to open camera device: {self.device}")

    def read(self) -> np.ndarray:
        """从摄像头读取一帧图像。

        如果启用了 crop_left，只返回左半帧。

        Returns:
            捕获的帧，numpy 数组格式 (H x W x 3)。

        Raises:
            RuntimeError: 摄像头未打开或读取失败时抛出。
        """
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
        """释放摄像头设备并重置内部捕获句柄。"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None