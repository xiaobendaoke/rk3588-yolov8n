# Ops Runbook

## Service control
```bash
systemctl status desk-safety.service
systemctl restart desk-safety.service
journalctl -u desk-safety.service -f
```

## Logs and data
- App logs: `/opt/desk-safety/logs/desk-safety.log`
- DB: `/opt/desk-safety/data/events.db`
- Snapshots: `/opt/desk-safety/data/snapshots/YYYY-MM-DD/`

## Common checks
```bash
cat /sys/class/devfreq/fdab0000.npu/cur_freq
v4l2-ctl --device=/dev/video21 --info
curl -s http://127.0.0.1:8080/api/status
```

## Failure handling
1. Web unavailable: check service status and port 8080 occupancy.
2. Zero FPS: verify `/dev/video21` and camera permissions.
3. No events: inspect model output conf threshold and rule params in `configs/config.yaml`.
4. Time mismatch: enable NTP (`timedatectl set-ntp true`) to keep event timestamps reliable.

## Security hardening before production
1. Replace `root/root`.
2. Use SSH key auth and disable password auth.
3. Restrict board management network.
