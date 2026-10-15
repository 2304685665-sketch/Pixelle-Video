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
    input_hash TEXT,
    PRIMARY KEY (task_id, idx)
);
-- Content-addressed cache of generated segments, keyed by shot input fingerprint.
CREATE TABLE IF NOT EXISTS artifacts (
    task_id TEXT NOT NULL,
    input_hash TEXT NOT NULL,
    segment_path TEXT NOT NULL,
    duration REAL NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (task_id, input_hash)
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
    input_hash: Optional[str] = None


class CheckpointStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(SCHEMA)
            cols = {r[1] for r in c.execute("PRAGMA table_info(shots)")}
            if "input_hash" not in cols:  # databases created before the artifact cache
                c.execute("ALTER TABLE shots ADD COLUMN input_hash TEXT")

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

    def mark_done(
        self, task_id: str, idx: int, segment_path: str, duration: float, input_hash: Optional[str] = None
    ) -> None:
        """Mark a shot done; with ``input_hash`` the segment is also added to the artifact cache."""
        now = time.time()
        with self._conn() as c:
            c.execute(
                "UPDATE shots SET status='done', segment_path=?, duration=?, error_class=NULL, "
                "last_error=NULL, input_hash=?, updated_at=? WHERE task_id=? AND idx=?",
                (segment_path, duration, input_hash, now, task_id, idx),
            )
            if input_hash:
                c.execute(
                    "INSERT INTO artifacts VALUES (?,?,?,?,?) ON CONFLICT(task_id, input_hash) DO UPDATE "
                    "SET segment_path=excluded.segment_path, duration=excluded.duration",
                    (task_id, input_hash, segment_path, duration, now),
                )

    def get_artifact(self, task_id: str, input_hash: str) -> Optional[tuple[str, float]]:
        with self._conn() as c:
            row = c.execute(
                "SELECT segment_path, duration FROM artifacts WHERE task_id=? AND input_hash=?",
                (task_id, input_hash),
            ).fetchone()
        return (row[0], row[1]) if row else None

    def replace_plan(self, task_id: str, plan: dict, n_shots: int) -> None:
        """Swap in an edited plan: shot rows are reset, the artifact cache is kept."""
        now = time.time()
        with self._conn() as c:
            c.execute(
                "UPDATE tasks SET plan_json=?, status='running', updated_at=? WHERE task_id=?",
                (json.dumps(plan, ensure_ascii=False, default=str), now, task_id),
            )
            c.execute("DELETE FROM shots WHERE task_id=?", (task_id,))
            c.executemany(
                "INSERT INTO shots (task_id, idx, status, updated_at) VALUES (?,?,?,?)",
                [(task_id, i, "pending", now) for i in range(n_shots)],
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
                "SELECT idx,status,attempts,error_class,last_error,segment_path,duration,input_hash "
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
