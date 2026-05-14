#!/usr/bin/env bash
set -euo pipefail

WEB_URL="${WEB_URL:-http://127.0.0.1:8080}"

curl -fsS "$WEB_URL/api/status" | grep -E 'fps|npu_freq' >/dev/null
curl -fsS "$WEB_URL/api/events?page=1&size=1" | grep -E 'items' >/dev/null

DB_PATH="${DB_PATH:-./data/events.db}"
python3 - <<PY
import sqlite3
conn = sqlite3.connect('${DB_PATH}')
cur = conn.cursor()
cur.execute("select count(*) from events")
print('event_count', cur.fetchone()[0])
PY

echo "healthcheck ok"
