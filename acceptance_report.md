# Acceptance Report Template

Date: ________
Board: ________
Operator: ________

## Gate A: Environment
- [ ] `python3/pip3` available
- [ ] import `cv2/numpy/sqlite3/rknnlite` pass
- [ ] NPU runtime init pass (`load_rknn=0`, `init_runtime=0`)
- [ ] `/dev/video21` capture pass

## Gate B: Functional
- [ ] live web stream accessible
- [ ] detections overlay shown
- [ ] 3 risk rules can be triggered in controlled scenario
- [ ] event saved into SQLite and queryable via `/api/events`
- [ ] snapshot file created on risk event

## Gate C: Stability/Performance
- [ ] E2E alert latency < 1s
- [ ] effective FPS >= 10 (single stream)
- [ ] 24h continuous run without crash

## Evidence
- baseline report path: ____________________
- service logs path: ____________________
- DB check output: ____________________
- web screenshots: ____________________

## Final Result
- [ ] PASS
- [ ] FAIL

Notes:
