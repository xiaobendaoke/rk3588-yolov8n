from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import cv2
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from app.types import RuntimeStatus


class AppState:
    """推理循环和 Web 服务器之间的共享状态。

    存储最新帧、其 JPEG 编码和运行时状态，通过锁保护实现线程安全访问。

    Attributes:
        lock: 用于同步帧和状态访问的线程锁。
        latest_frame: 最新的原始帧（numpy 数组或 None）。
        latest_jpeg: 最新帧编码为 JPEG 字节。
        status: 当前运行时状态快照。
    """

    def __init__(self) -> None:
        """初始化共享应用状态。"""
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_jpeg = b""
        self.status = RuntimeStatus()


def create_app(
    state: AppState,
    list_events: Callable[..., list],
    get_event: Callable[[int], dict | None] | None = None,
    get_event_stats: Callable[[], dict] | None = None,
) -> FastAPI:
    """创建并配置 FastAPI Web 应用。

    注册仪表板、实时视频流、帧服务、事件 API 和系统状态接口的路由。

    Args:
        state: 用于线程安全数据访问的共享应用状态。
        list_events: 列出事件的可调用对象（支持分页/过滤）。
        get_event: 根据 ID 获取单条事件的可调用对象。
        get_event_stats: 获取事件统计信息的可调用对象。

    Returns:
        配置好的 FastAPI 应用实例。
    """
    app = FastAPI(title="Desk Safety")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        """提供主监控仪表板 HTML 页面。"""
        return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Desk Safety Monitor</title>
  <style>
    :root {
      --bg: #1a1a2e;
      --panel: #16213e;
      --card: #0f3460;
      --accent: #e94560;
      --success: #00b894;
      --warning: #fdcb6e;
      --danger: #e94560;
      --text: #eee;
      --text-muted: #aaa;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
    }
    .header {
      background: linear-gradient(135deg, var(--panel), var(--card));
      padding: 12px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 2px solid var(--accent);
    }
    .header h1 { font-size: 1.2rem; font-weight: 600; }
    .header .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 12px;
      border-radius: 20px;
      font-size: 0.8rem;
      background: rgba(0,184,148,0.2);
      color: var(--success);
    }
    .header .status-badge::before {
      content: '';
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--success);
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.4; }
    }
    .main {
      display: grid;
      grid-template-columns: 1.5fr 1fr;
      gap: 16px;
      padding: 16px;
      max-width: 1400px;
      margin: 0 auto;
    }
    .card {
      background: var(--panel);
      border-radius: 12px;
      overflow: hidden;
      box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .card-header {
      background: var(--card);
      padding: 10px 16px;
      font-size: 0.9rem;
      font-weight: 600;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .card-body { padding: 16px; }
    .video-container {
      position: relative;
      background: #000;
      border-radius: 8px;
      overflow: hidden;
    }
    .video-container img {
      width: 100%;
      display: block;
    }
    .video-overlay {
      position: absolute;
      bottom: 8px;
      left: 8px;
      display: flex;
      gap: 8px;
    }
    .video-badge {
      background: rgba(0,0,0,0.7);
      padding: 4px 10px;
      border-radius: 4px;
      font-size: 0.75rem;
      backdrop-filter: blur(4px);
    }
    .stats-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 16px;
    }
    .stat-item {
      background: var(--card);
      border-radius: 8px;
      padding: 12px;
      text-align: center;
    }
    .stat-value {
      font-size: 1.5rem;
      font-weight: 700;
      color: var(--accent);
    }
    .stat-label {
      font-size: 0.75rem;
      color: var(--text-muted);
      margin-top: 4px;
    }
    .event-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
    }
    .event-table th {
      text-align: left;
      padding: 8px 12px;
      background: var(--card);
      color: var(--text-muted);
      font-weight: 500;
    }
    .event-table td {
      padding: 8px 12px;
      border-bottom: 1px solid rgba(255,255,255,0.05);
    }
    .event-table tr:hover td {
      background: rgba(255,255,255,0.03);
    }
    .severity-high { color: var(--danger); font-weight: 600; }
    .severity-medium { color: var(--warning); }
    .severity-low { color: var(--success); }
    .risk-tag {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      background: rgba(233,69,96,0.2);
      color: var(--accent);
    }
    .detections-list {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }
    .detection-tag {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 0.8rem;
      background: rgba(0,184,148,0.15);
      color: var(--success);
      border: 1px solid rgba(0,184,148,0.3);
    }
    .detection-tag .conf {
      color: var(--text-muted);
      font-size: 0.7rem;
    }
    .no-events {
      text-align: center;
      color: var(--text-muted);
      padding: 24px;
      font-size: 0.9rem;
    }
    @media (max-width: 900px) {
      .main { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="header">
    <h1>Desk Safety Monitor</h1>
    <div class="status-badge" id="statusBadge">运行中</div>
  </div>

  <div class="main">
    <div class="card">
      <div class="card-header">
        <span>实时监控画面</span>
        <span id="fpsBadge" style="font-size:0.8rem;color:var(--success)">-- FPS</span>
      </div>
      <div class="card-body">
        <div class="video-container">
          <img id="live" src="/frame.jpg" />
          <div class="video-overlay">
            <span class="video-badge" id="timeBadge">--:--:--</span>
            <span class="video-badge" id="detectionBadge">检测中...</span>
          </div>
        </div>
        <div class="detections-list" id="detectionsList"></div>
      </div>
    </div>

    <div style="display:flex;flex-direction:column;gap:16px;">
      <div class="card">
        <div class="card-header">系统状态</div>
        <div class="card-body">
          <div class="stats-grid">
            <div class="stat-item">
              <div class="stat-value" id="statFps">--</div>
              <div class="stat-label">FPS</div>
            </div>
            <div class="stat-item">
              <div class="stat-value" id="statCpu">--%</div>
              <div class="stat-label">CPU</div>
            </div>
            <div class="stat-item">
              <div class="stat-value" id="statGpu">--%</div>
              <div class="stat-label">GPU</div>
            </div>
            <div class="stat-item">
              <div class="stat-value" id="statNpu">--%</div>
              <div class="stat-label">NPU</div>
            </div>
            <div class="stat-item">
              <div class="stat-value" id="statMem">--%</div>
              <div class="stat-label">内存</div>
            </div>
            <div class="stat-item">
              <div class="stat-value" id="statTemp">--°C</div>
              <div class="stat-label">温度</div>
            </div>
          </div>
        </div>
      </div>

      <div class="card" style="flex:1;">
        <div class="card-header">
          <span>风险事件记录</span>
          <span id="eventCount" style="font-size:0.8rem;color:var(--text-muted)">--</span>
        </div>
        <div class="card-body" style="max-height:400px;overflow:auto;">
          <table class="event-table" id="eventTable">
            <thead>
              <tr>
                <th>时间</th>
                <th>风险类型</th>
                <th>严重程度</th>
              </tr>
            </thead>
            <tbody id="eventBody">
              <tr><td colspan="3" class="no-events">加载中...</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <script>
    const RISK_NAMES = {
      'liquid_near_electronics': '液体靠近电子设备',
      'sharp_tool_misplaced': '尖锐工具误放',
      'desk_overcrowded': '桌面物品拥挤'
    };
    const SEVERITY_NAMES = {
      'high': '高',
      'medium': '中',
      'low': '低'
    };

    function updateFrame() {
      var img = document.getElementById('live');
      img.src = '/frame.jpg?' + Date.now();
      document.getElementById('timeBadge').textContent = new Date().toLocaleTimeString('zh-CN');
    }
    setInterval(updateFrame, 100);

    async function refreshStatus() {
      try {
        const s = await fetch('/api/status').then(r => r.json());
        document.getElementById('statFps').textContent = s.fps.toFixed(1);
        document.getElementById('statCpu').textContent = s.cpu_percent.toFixed(0) + '%';
        document.getElementById('statGpu').textContent = s.gpu_load.toFixed(0) + '%';
        document.getElementById('statNpu').textContent = s.npu_load.toFixed(0) + '%';
        document.getElementById('statMem').textContent = s.mem_percent.toFixed(0) + '%';
        document.getElementById('statTemp').textContent = s.cpu_temp.toFixed(0) + '°C';
        document.getElementById('fpsBadge').textContent = s.fps.toFixed(1) + ' FPS';
        document.getElementById('detectionBadge').textContent = '检测: ' + s.detection_count + ' 个目标';
      } catch(e) {}
    }

    async function refreshEvents() {
      try {
        const e = await fetch('/api/events?page=1&size=20').then(r => r.json());
        const body = document.getElementById('eventBody');
        document.getElementById('eventCount').textContent = e.items.length + ' 条记录';

        if (e.items.length === 0) {
          body.innerHTML = '<tr><td colspan="3" class="no-events">暂无风险事件</td></tr>';
          return;
        }

        body.innerHTML = e.items.map(item => {
          const time = item.event_time ? item.event_time.split('T')[1] : '--:--';
          const riskName = RISK_NAMES[item.risk_type] || item.risk_type;
          const sevName = SEVERITY_NAMES[item.severity] || item.severity;
          const sevClass = 'severity-' + item.severity;
          return '<tr>' +
            '<td>' + time + '</td>' +
            '<td><span class="risk-tag">' + riskName + '</span></td>' +
            '<td class="' + sevClass + '">' + sevName + '</td>' +
            '</tr>';
        }).join('');
      } catch(e) {}
    }

    setInterval(refreshStatus, 1000);
    setInterval(refreshEvents, 3000);
    refreshStatus();
    refreshEvents();
  </script>
</body>
</html>
        """

    @app.get("/live.mjpg")
    def live_mjpg() -> StreamingResponse:
        """以 MJPEG 视频流方式推送最新帧。"""
        def stream():
            while True:
                with state.lock:
                    frame = state.latest_jpeg
                if frame:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                else:
                    time.sleep(0.05)

        return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/frame.jpg")
    def frame_jpeg() -> Response:
        """提供最新帧的静态 JPEG 图片。"""
        with state.lock:
            frame = state.latest_jpeg
        if frame:
            return Response(content=frame, media_type="image/jpeg")
        return Response(content=b"", media_type="image/jpeg")

    @app.get("/api/events")
    def api_events(
        page: int = Query(default=1, ge=1),
        size: int = Query(default=20, ge=1, le=200),
        risk_type: Optional[str] = Query(default=None),
    ) -> JSONResponse:
        """列出风险事件，支持分页和可选的风险类型过滤。"""
        items = list_events(page=page, size=size, risk_type=risk_type)
        return JSONResponse({"page": page, "size": size, "items": items})

    @app.get("/api/events/stats")
    def api_event_stats() -> JSONResponse:
        """获取事件聚合统计信息。"""
        if get_event_stats is None:
            return JSONResponse({"error": "not supported"}, status_code=501)
        return JSONResponse(get_event_stats())

    @app.get("/api/events/{event_id}")
    def api_event_detail(event_id: int) -> JSONResponse:
        """根据 ID 获取单条风险事件详情。"""
        if get_event is None:
            return JSONResponse({"error": "not supported"}, status_code=501)
        item = get_event(event_id)
        if item is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(item)

    @app.get("/api/status")
    def api_status() -> JSONResponse:
        """获取当前系统运行时状态。"""
        with state.lock:
            status = {
                "fps": round(state.status.fps, 2),
                "queue_size": state.status.queue_size,
                "last_event_time": state.status.last_event_time,
                "cpu_percent": state.status.cpu_percent,
                "gpu_load": round(state.status.gpu_load, 1),
                "npu_load": round(state.status.npu_load, 1),
                "mem_percent": state.status.mem_percent,
                "cpu_temp": round(state.status.cpu_temp, 1),
                "gpu_temp": round(state.status.gpu_temp, 1),
                "detection_count": state.status.detection_count,
            }
        return JSONResponse(status)

    return app


def annotate_frame(frame, detections, risks):
    """在帧上绘制检测框和风险标签。

    Args:
        frame: 要标注的图像帧（原地修改）。
        detections: 要绘制的 Detection 对象列表。
        risks: 要显示为文本的 RiskEventCandidate 对象列表。

    Returns:
        标注后的帧（与输入相同对象，原地修改）。
    """
    for d in detections:
        x1, y1, x2, y2 = d.bbox_xyxy
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 220, 80), 2)
        cv2.putText(
            frame,
            f"{d.class_name} {d.conf:.2f}",
            (x1, max(y1 - 6, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    for i, r in enumerate(risks):
        cv2.putText(
            frame,
            f"RISK:{r.risk_type}({r.severity})",
            (10, 24 + i * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 250),
            2,
            cv2.LINE_AA,
        )

    return frame