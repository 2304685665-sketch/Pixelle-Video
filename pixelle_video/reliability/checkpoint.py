"""
SQLite checkpoint store for tasks and per-shot state.

Upstream persists a storyboard only in ``finalize()``, after the whole video
succeeded. Here the plan is written before any paid call and every shot
transition is committed immediately, so a crash or a failed shot never loses
finished work. Writes are single-row UPSERTs inside a transaction; SQLite WAL
mode keeps them safe against a process kill.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    pipeline TEXT NOT NULL,
    plan_json TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS shots (
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    status TEXT NOT NULL,          -- pending | done | failed
    attempts INTEGER NOT NULL DEFAULT 0,
    error_class TEXT,
    last_error TEXT,
    segment_path TEXT,
    duration REAL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (task_id, idx)
);
"""


@dataclass
class ShotRecord:
    idx: int
    status: str
    attempts: int
    error_class: Optional[str]
    last_error: Optional[str]
    segment_path: Optional[str]
    duration: Optional[float]


class CheckpointStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=30)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ #

    def save_plan(self, task_id: str, pipeline: str, plan: dict, n_shots: int) -> None:
        now = time.time()
        with self._conn() as c:
            c.execute(
                "INSERT INTO tasks VALUES (?,?,?,?,?,?) ON CONFLICT(task_id) DO UPDATE SET "
                "plan_json=excluded.plan_json, updated_at=excluded.updated_at",
                (task_id, pipeline, json.dumps(plan, ensure_ascii=False, default=str), "running", now, now),
            )
            c.executemany(
                "INSERT OR IGNORE INTO shots (task_id, idx, status, updated_at) VALUES (?,?,?,?)",
                [(task_id, i, "pending", now) for i in range(n_shots)],
            )

    def load_plan(self, task_id: str) -> Optional[dict]:
        with self._conn() as c:
            row = c.execute("SELECT plan_json FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def set_task_status(self, task_id: str, status: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE tasks SET status=?, updated_at=? WHERE task_id=?", (status, time.time(), task_id))

    def task_status(self, task_id: str) -> Optional[str]:
        with self._conn() as c:
            row = c.execute("SELECT status FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return row[0] if row else None

    def record_attempt(self, task_id: str, idx: int, error_class: Optional[str], error: Optional[str]) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE shots SET attempts=attempts+1, error_class=?, last_error=?, updated_at=? "
                "WHERE task_id=? AND idx=?",
                (error_class, (error or "")[:500] or None, time.time(), task_id, idx),
            )

    def mark_done(self, task_id: str, idx: int, segment_path: str, duration: float) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE shots SET status='done', segment_path=?, duration=?, error_class=NULL, "
                "last_error=NULL, updated_at=? WHERE task_id=? AND idx=?",
                (segment_path, duration, time.time(), task_id, idx),
            )

    def mark_failed(self, task_id: str, idx: int) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE shots SET status='failed', updated_at=? WHERE task_id=? AND idx=?",
                (time.time(), task_id, idx),
            )

    def shots(self, task_id: str) -> Dict[int, ShotRecord]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT idx,status,attempts,error_class,last_error,segment_path,duration "
                "FROM shots WHERE task_id=? ORDER BY idx",
                (task_id,),
            ).fetchall()
        return {r[0]: ShotRecord(*r) for r in rows}

    def tasks(self, status: Optional[str] = None) -> List[str]:
        q, args = "SELECT task_id FROM tasks", ()
        if status:
            q, args = q + " WHERE status=?", (status,)
        with self._conn() as c:
            return [r[0] for r in c.execute(q + " ORDER BY created_at", args).fetchall()]
