from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from app.types import RiskEventCandidate, detection_to_dict


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  event_time TEXT NOT NULL,
  risk_type TEXT NOT NULL,
  severity TEXT NOT NULL,
  confidence REAL NOT NULL,
  snapshot_path TEXT NOT NULL,
  meta_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_events_time ON events(event_time DESC);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(risk_type);
"""


class EventStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def init_schema(self) -> None:
        with self.conn:
            self.conn.executescript(SCHEMA_SQL)
            self.conn.execute("PRAGMA journal_mode=WAL;")

    def insert_event(self, event: RiskEventCandidate, snapshot_path: str) -> int:
        payload = {
            "reason": event.reason,
            "objects": [detection_to_dict(d) for d in event.objects],
        }
        with self.conn:
            cur = self.conn.execute(
                """
                INSERT INTO events (event_time, risk_type, severity, confidence, snapshot_path, meta_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    _to_iso(event.ts_ms),
                    event.risk_type,
                    event.severity,
                    round(event.confidence, 4),
                    snapshot_path,
                    json.dumps(payload, ensure_ascii=True),
                ),
            )
        return int(cur.lastrowid)

    def list_events(self, page: int = 1, size: int = 20, risk_type: str | None = None) -> List[Dict[str, Any]]:
        page = max(page, 1)
        size = max(min(size, 200), 1)
        offset = (page - 1) * size

        sql = """
        SELECT id, event_time, risk_type, severity, confidence, snapshot_path, meta_json, created_at
        FROM events
        """
        args: list[Any] = []
        if risk_type:
            sql += " WHERE risk_type = ?"
            args.append(risk_type)

        sql += " ORDER BY event_time DESC LIMIT ? OFFSET ?"
        args.extend([size, offset])

        rows = self.conn.execute(sql, args).fetchall()
        result: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            try:
                item["meta_json"] = json.loads(item["meta_json"])
            except Exception:
                pass
            result.append(item)
        return result

    def get_event(self, event_id: int) -> Dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM events WHERE id = ?", (event_id,)
        ).fetchone()
        if row is None:
            return None
        item = dict(row)
        try:
            item["meta_json"] = json.loads(item["meta_json"])
        except Exception:
            pass
        return item

    def get_event_stats(self) -> Dict[str, Any]:
        rows = self.conn.execute(
            "SELECT risk_type, COUNT(*) as cnt FROM events GROUP BY risk_type"
        ).fetchall()
        stats = {row["risk_type"]: row["cnt"] for row in rows}

        total = self.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        recent = self.conn.execute(
            "SELECT COUNT(*) FROM events WHERE event_time >= datetime('now', '-24 hours')"
        ).fetchone()[0]

        return {
            "total": total,
            "recent_24h": recent,
            "by_type": stats,
        }


def _to_iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0).isoformat(timespec="seconds")
