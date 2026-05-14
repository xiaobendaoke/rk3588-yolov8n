from __future__ import annotations

import argparse
import logging
import threading
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import cv2
import uvicorn

from app.capture.camera import CameraCapture
from app.config import load_settings
from app.infer.engine import InferenceEngine
from app.infer.native_engine import NativeInferenceEngine
from app.rules.engine import RuleConfig, RuleEngine
from app.storage.events import EventStore
from app.web.server import AppState, annotate_frame, create_app


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    Returns:
        解析后的参数命名空间，包含配置文件路径。
    """
    p = argparse.ArgumentParser(description="Desk safety monitor")
    p.add_argument("--config", default="./configs/config.yaml")
    return p.parse_args()


def setup_logging(level: str, log_dir: str = "./logs") -> None:
    """配置日志系统，同时输出到控制台和轮转文件。

    Args:
        level: 日志级别字符串（如 "INFO", "DEBUG"）。
        log_dir: 日志文件目录路径。
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    rotating = RotatingFileHandler(
        Path(log_dir) / "desk-safety.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            rotating,
        ],
    )


def read_npu_load() -> float:
    """从 sysfs 读取 NPU 负载百分比。

    Returns:
        NPU 负载百分比（0-100），不可用时返回 0.0。
    """
    path = Path("/sys/class/devfreq/fdab0000.npu/load")
    if not path.exists():
        return 0.0
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if "@" in raw:
            return float(raw.split("@")[0])
        return 0.0
    except Exception:
        return 0.0


def read_gpu_load() -> float:
    """从 sysfs 读取 GPU 负载百分比。

    Returns:
        GPU 负载百分比（0-100），不可用时返回 0.0。
    """
    path = Path("/sys/class/devfreq/fb000000.gpu/load")
    if not path.exists():
        return 0.0
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if "@" in raw:
            return float(raw.split("@")[0])
        return 0.0
    except Exception:
        return 0.0


def read_cpu_percent() -> float:
    """从 /proc/stat 读取 CPU 使用率百分比。

    使用前一次采样值计算基于差值的利用率。

    Returns:
        CPU 使用率百分比（0-100），出错时返回 0.0。
    """
    try:
        with open("/proc/stat", "r") as f:
            line = f.readline()
        parts = line.split()[1:]
        total = sum(int(p) for p in parts)
        idle = int(parts[3])
        if not hasattr(read_cpu_percent, "_prev"):
            read_cpu_percent._prev = (total, idle)
            return 0.0
        prev_total, prev_idle = read_cpu_percent._prev
        read_cpu_percent._prev = (total, idle)
        d_total = total - prev_total
        d_idle = idle - prev_idle
        if d_total == 0:
            return 0.0
        return round((1.0 - d_idle / d_total) * 100, 1)
    except Exception:
        return 0.0


def read_mem_percent() -> float:
    """从 /proc/meminfo 读取内存使用率百分比。

    Returns:
        内存使用率百分比（0-100），出错时返回 0.0。
    """
    try:
        with open("/proc/meminfo", "r") as f:
            lines = f.readlines()
        info = {}
        for line in lines:
            parts = line.split()
            if len(parts) >= 2:
                info[parts[0].rstrip(":")] = int(parts[1])
        total = info.get("MemTotal", 1)
        available = info.get("MemAvailable", 0)
        return round((1.0 - available / total) * 100, 1)
    except Exception:
        return 0.0


def read_thermal(zone: str) -> float:
    """从 sysfs 的热区读取温度。

    Args:
        zone: 热区名称（如 "thermal_zone0"）。

    Returns:
        摄氏度温度值，不可用时返回 0.0。
    """
    path = Path(f"/sys/class/thermal/{zone}/temp")
    if not path.exists():
        return 0.0
    try:
        val = int(path.read_text(encoding="utf-8").strip())
        return val / 1000.0 if val > 1000 else float(val)
    except Exception:
        return 0.0


def save_snapshot(snapshot_root: str, frame) -> str:
    """将帧保存为按日期组织的 JPEG 快照文件。

    Args:
        snapshot_root: 快照存储的根目录。
        frame: 要保存的图像帧（numpy 数组）。

    Returns:
        保存的快照文件的完整路径。
    """
    now = datetime.now()
    day = now.strftime("%Y-%m-%d")
    stamp = now.strftime("%Y%m%d_%H%M%S_%f")
    target_dir = Path(snapshot_root) / day
    target_dir.mkdir(parents=True, exist_ok=True)
    out_path = target_dir / f"event_{stamp}.jpg"
    cv2.imwrite(str(out_path), frame)
    return str(out_path)


def infer_loop(
    camera: CameraCapture,
    infer,
    rules: RuleEngine,
    state: AppState,
    settings,
    store: EventStore,
    log: logging.Logger,
) -> None:
    """主推理循环：捕获、检测、评估风险并提供帧服务。

    持续运行，从摄像头读取帧，执行推理，评估安全规则，为风险事件保存
    快照，并用最新帧和状态更新共享的 Web 服务器状态。

    Args:
        camera: 摄像头捕获实例。
        infer: 目标检测推理引擎。
        rules: 风险评估规则引擎。
        state: Web 服务器的共享应用状态。
        settings: 应用设置（用于 snapshot_root 和 interval）。
        store: 用于持久化风险事件的事件存储。
        log: 日志记录器实例。
    """
    from app.infer.native_engine import NativeInferenceEngine

    if isinstance(infer, NativeInferenceEngine) and infer._pool is not None:
        _infer_loop_native(camera, infer, rules, state, settings, store, log)
    else:
        _infer_loop_standard(camera, infer, rules, state, settings, store, log)


def _infer_loop_native(
    camera: CameraCapture,
    infer,
    rules: RuleEngine,
    state: AppState,
    settings,
    store: EventStore,
    log: logging.Logger,
) -> None:
    """异步推理循环：利用多线程并行推理。"""
    import cv2
    from collections import deque
    from concurrent.futures import Future

    last = time.time()
    frames = 0
    frame_size_logged = False

    # 流水线：提交多个帧到线程池
    pending: deque[tuple[Future, np.ndarray]] = deque(maxlen=infer.n_workers + 1)

    while True:
        frame = camera.read()
        if not frame_size_logged:
            log.info("camera frame size: %dx%d", frame.shape[1], frame.shape[0])
            frame_size_logged = True

        # Resize 并提交异步推理
        resized = cv2.resize(frame, (infer.input_size, infer.input_size))
        fut = infer.infer_async(resized)
        pending.append((fut, resized))

        # 收集已完成的结果
        if len(pending) > infer.n_workers:
            fut, model_frame = pending.popleft()
            detections = fut.result(timeout=5)
            risks = rules.evaluate(detections)
            annotated = annotate_frame(model_frame.copy(), detections, risks)

            for risk in risks:
                snapshot_path = save_snapshot(settings.snapshot_root, annotated)
                event_id = store.insert_event(risk, snapshot_path)
                with state.lock:
                    state.status.last_event_time = datetime.now().isoformat(timespec="seconds")
                log.warning("event id=%s type=%s severity=%s", event_id, risk.risk_type, risk.severity)

            ok, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                with state.lock:
                    state.latest_jpeg = jpg.tobytes()
                    state.status.queue_size = 0
                    state.status.npu_load = read_npu_load()
                    state.status.gpu_load = read_gpu_load()
                    state.status.detection_count = len(detections)

            frames += 1
            now = time.time()
            if now - last >= 1.0:
                fps = frames / (now - last)
                with state.lock:
                    state.status.fps = fps
                    state.status.cpu_percent = read_cpu_percent()
                    state.status.mem_percent = read_mem_percent()
                    state.status.cpu_temp = read_thermal("thermal_zone0")
                    state.status.gpu_temp = read_thermal("thermal_zone1")
                log.info("fps=%.2f detections=%d risks=%d", fps, len(detections), len(risks))
                frames = 0
                last = now


def _infer_loop_standard(
    camera: CameraCapture,
    infer,
    rules: RuleEngine,
    state: AppState,
    settings,
    store: EventStore,
    log: logging.Logger,
) -> None:
    """标准推理循环（单线程）。"""
    last = time.time()
    frames = 0
    frame_size_logged = False

    while True:
        frame = camera.read()
        if not frame_size_logged:
            log.info("camera frame size: %dx%d", frame.shape[1], frame.shape[0])
            frame_size_logged = True

        detections, model_frame = infer.infer(frame)
        risks = rules.evaluate(detections)
        annotated = annotate_frame(model_frame.copy(), detections, risks)

        for risk in risks:
            snapshot_path = save_snapshot(settings.snapshot_root, annotated)
            event_id = store.insert_event(risk, snapshot_path)
            with state.lock:
                state.status.last_event_time = datetime.now().isoformat(timespec="seconds")
            log.warning("event id=%s type=%s severity=%s", event_id, risk.risk_type, risk.severity)

        ok, jpg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with state.lock:
                state.latest_jpeg = jpg.tobytes()
                state.status.queue_size = 0
                state.status.npu_load = read_npu_load()
                state.status.gpu_load = read_gpu_load()
                state.status.detection_count = len(detections)

        frames += 1
        now = time.time()
        if now - last >= 1.0:
            fps = frames / (now - last)
            with state.lock:
                state.status.fps = fps
                state.status.cpu_percent = read_cpu_percent()
                state.status.mem_percent = read_mem_percent()
                state.status.cpu_temp = read_thermal("thermal_zone0")
                state.status.gpu_temp = read_thermal("thermal_zone1")
            log.info("fps=%.2f detections=%d risks=%d", fps, len(detections), len(risks))
            frames = 0
            last = now

        interval = settings.infer_interval_ms
        if interval > 0:
            time.sleep(interval / 1000.0)


def run_pipeline(args: argparse.Namespace) -> None:
    """运行完整的桌面安全监控管道。

    初始化所有组件（摄像头、推理、规则、存储、Web 服务器），
    启动 Web 服务器和推理循环线程，然后阻塞直到中断。

    Args:
        args: 解析后的命令行参数。
    """
    settings = load_settings(args.config)
    setup_logging(settings.log_level)
    log = logging.getLogger("desk-safety")

    state = AppState()

    store = EventStore(settings.db_path)
    store.init_schema()

    camera = CameraCapture(settings.camera_device, settings.camera_width, settings.camera_height, settings.camera_crop_left)

    if settings.use_native:
        infer = NativeInferenceEngine(
            settings.model_path,
            settings.class_names,
            settings.conf_threshold,
            settings.nms_threshold,
            settings.input_size,
        )
    elif settings.npu_threads > 1:
        from app.infer.multi_process import MultiProcessEngine
        infer = MultiProcessEngine(
            settings.model_path,
            settings.class_names,
            settings.conf_threshold,
            settings.nms_threshold,
            settings.input_size,
            n_workers=settings.npu_threads,
        )
    else:
        infer = InferenceEngine(
            settings.model_path,
            settings.class_names,
            settings.conf_threshold,
            settings.nms_threshold,
            settings.input_size,
        )
    rules = RuleEngine(
        RuleConfig(
            risk_distance_px=settings.risk_distance_px,
            risk_hold_frames=settings.risk_hold_frames,
            event_cooldown_sec=settings.event_cooldown_sec,
            danger_roi=tuple(settings.danger_roi),
            dense_count_threshold=settings.dense_count_threshold,
            dense_iou_sum_threshold=settings.dense_iou_sum_threshold,
        )
    )

    camera.open()
    infer.open()

    app = create_app(
        state=state,
        list_events=store.list_events,
        get_event=store.get_event,
        get_event_stats=store.get_event_stats,
    )
    web_thread = threading.Thread(
        target=lambda: uvicorn.run(
            app,
            host=settings.web_host,
            port=settings.web_port,
            log_level="warning",
            log_config=None,
        ),
        daemon=True,
    )
    web_thread.start()
    log.info("web server started at http://%s:%s", settings.web_host, settings.web_port)

    infer_thread = threading.Thread(
        target=infer_loop,
        args=(camera, infer, rules, state, settings, store, log),
        daemon=True,
    )
    infer_thread.start()
    log.info("inference thread started")

    try:
        web_thread.join()
    except KeyboardInterrupt:
        log.info("stopped by keyboard interrupt")
    finally:
        infer.close()
        camera.close()


if __name__ == "__main__":
    run_pipeline(parse_args())