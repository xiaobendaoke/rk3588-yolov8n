# Desk Safety (RK3588)

## 配置

1. 复制示例配置文件：
```bash
cp configs/config.example.yaml configs/config.yaml
```

2. 编辑配置文件，根据您的设备修改：
   - `camera_device`: 摄像头设备路径（通过 `v4l2-ctl --list-devices` 查看）
   - `model_path`: 模型文件路径
   - 其他参数根据需要调整

## Quick Start

```bash
cd /home/nidie/rk3588/desk-safety
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m app.main --config ./configs/config.yaml
```

Open: `http://<board-ip>:8080`

## Deploy to board

```bash
sudo mkdir -p /opt/desk-safety
sudo rsync -a --delete ./ /opt/desk-safety/
sudo cp ./systemd/desk-safety.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now desk-safety.service
sudo systemctl status desk-safety.service
```

## Acceptance checks

```bash
python3 - <<'PY'
import cv2, numpy, sqlite3
print('imports ok')
PY

cat /sys/class/devfreq/fdab0000.npu/cur_freq
v4l2-ctl --device=/dev/video21 --info
curl -s http://127.0.0.1:8080/api/status
```
