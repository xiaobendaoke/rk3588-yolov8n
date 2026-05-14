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
from app.rules.engine import RuleConfig, RuleEngine
from app.storage.events import EventStore
from app.web.server import AppState, annotate_frame, create_app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Desk safety monitor")
    p.add_argument("--config", default="./configs/config.yaml")
    return p.parse_args()


def setup_logging(level: str, log_dir: str = "./logs") -> None:
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
    path = Path(f"/sys/class/thermal/{zone}/temp")
    if not path.exists():
        return 0.0
    try:
        val = int(path.read_text(encoding="utf-8").strip())
        return val / 1000.0 if val > 1000 else float(val)
    except Exception:
        return 0.0


def save_snapshot(snapshot_root: str, frame) -> str:
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
    infer: InferenceEngine,
    rules: RuleEngine,
    state: AppState,
    settings,
    store: EventStore,
    log: logging.Logger,
) -> None:
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
    settings = load_settings(args.config)
    setup_logging(settings.log_level)
    log = logging.getLogger("desk-safety")

    state = AppState()

    store = EventStore(settings.db_path)
    store.init_schema()

    camera = CameraCapture(settings.camera_device, settings.camera_width, settings.camera_height, settings.camera_crop_left)
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
