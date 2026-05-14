# Deploy Guide

## 1. Copy project to board
```bash
cd /home/nidie/rk3588/desk-safety
rsync -a --delete ./ root@192.168.88.2:/opt/desk-safety/
```

## 2. Install runtime deps (board)
```bash
ssh root@192.168.88.2
cd /opt/desk-safety
python3 -m pip install -r requirements.txt
```

## 3. Configure
- Edit `configs/config.yaml` if camera/model path differs.
- Put RKNN file at `/opt/desk-safety/models/detector.rknn`.

## 4. Run manually
```bash
cd /opt/desk-safety
python3 -m app.main --config ./configs/config.yaml
```
Open `http://<board-ip>:8080`.

## 5. Enable service
```bash
cp /opt/desk-safety/systemd/desk-safety.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now desk-safety.service
systemctl status desk-safety.service
```
