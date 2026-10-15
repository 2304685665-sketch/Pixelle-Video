"""
Redis-backed job queue for video generation.

Replaces the in-memory ``api/tasks/manager.py`` (whose docstring notes it "can
be replaced with Redis later"): jobs survive API restarts, several workers can
share one queue, and a crashed worker's job is handed to another worker.

Keys (prefix ``pv:``)::

    idem:{key}      "<task_id>|<sha256 of request>"  idempotency record (TTL)
    task:{id}       hash: status, request, created/updated, requeues, error, result
    queue           list of task ids waiting
    processing      list of task ids taken by a worker
    lease:{id}      worker id, TTL-renewed while the worker is alive
    events:{id}     stream of progress events (see events.py)

Submission is one Lua script, so "check key, create task, enqueue" cannot be
interleaved by concurrent clients or half-applied by a crash.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

P = "pv:"
TERMINAL = {"completed", "failed"}

_SUBMIT_LUA = """
local existing = redis.call('GET', KEYS[1])
if existing then
  return {0, existing}
end
local task_id = ARGV[1]
redis.call('SET', KEYS[1], task_id .. '|' .. ARGV[2], 'EX', tonumber(ARGV[5]))
redis.call('HSET', ARGV[6] .. 'task:' .. task_id,
  'status', 'queued', 'request', ARGV[3], 'created', ARGV[4], 'updated', ARGV[4],
  'requeues', 0, 'idem_key', ARGV[7])
redis.call('RPUSH', KEYS[2], task_id)
return {1, task_id .. '|' .. ARGV[2]}
"""

# Move a job back to the queue only if it is still in `processing` and has no lease.
_REQUEUE_LUA = """
if redis.call('EXISTS', KEYS[3]) == 1 then return 0 end
local removed = redis.call('LREM', KEYS[1], 0, ARGV[1])
if removed == 0 then return 0 end
redis.call('RPUSH', KEYS[2], ARGV[1])
redis.call('HINCRBY', KEYS[4], 'requeues', 1)
redis.call('HSET', KEYS[4], 'status', 'queued', 'updated', ARGV[2])
return 1
"""


class IdempotencyConflict(Exception):
    """Same idempotency key reused with a different request body."""


@dataclass
class Submission:
    task_id: str
    deduplicated: bool


def request_digest(request: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class JobQueue:
    def __init__(self, redis, idempotency_ttl: int = 24 * 3600, lease_ttl: int = 30, prefix: str = P):
        """``redis`` is a ``redis.asyncio.Redis`` client (decode_responses=True)."""
        self.r, self.p = redis, prefix
        self.idempotency_ttl, self.lease_ttl = idempotency_ttl, lease_ttl
        self._submit = redis.register_script(_SUBMIT_LUA)
        self._requeue = redis.register_script(_REQUEUE_LUA)

    def k(self, *parts: str) -> str:
        return self.p + ":".join(parts)

    # ------------------------------------------------------------------ #
    # Client side
    # ------------------------------------------------------------------ #

    async def submit(self, request: Dict[str, Any], idempotency_key: str) -> Submission:
        digest = request_digest(request)
        created, value = await self._submit(
            keys=[self.k("idem", idempotency_key), self.k("queue")],
            args=[
                uuid.uuid4().hex[:16], digest, json.dumps(request, ensure_ascii=False),
                time.time(), self.idempotency_ttl, self.p, idempotency_key,
            ],
        )
        task_id, stored_digest = value.split("|", 1)
        if not created and stored_digest != digest:
            raise IdempotencyConflict(idempotency_key)
        return Submission(task_id=task_id, deduplicated=not created)

    async def get(self, task_id: str) -> Optional[Dict[str, str]]:
        data = await self.r.hgetall(self.k("task", task_id))
        return data or None

    async def requeue_partial(self, task_id: str) -> bool:
        """Put a partially finished task back in the queue (it will resume, not restart)."""
        status = await self.r.hget(self.k("task", task_id), "status")
        if status != "partial":
            return False
        await self.r.hset(self.k("task", task_id), mapping={"status": "queued", "updated": time.time()})
        await self.r.rpush(self.k("queue"), task_id)
        return True

    # ------------------------------------------------------------------ #
    # Worker side
    # ------------------------------------------------------------------ #

    async def claim(self, worker_id: str, timeout: float = 1.0) -> Optional[str]:
        """Atomically move one job from queue to processing and take its lease."""
        task_id = await self.r.blmove(self.k("queue"), self.k("processing"), timeout, "LEFT", "RIGHT")
        if task_id is None:
            return None
        await self.r.set(self.k("lease", task_id), worker_id, ex=self.lease_ttl)
        await self.r.hset(
            self.k("task", task_id), mapping={"status": "running", "worker": worker_id, "updated": time.time()}
        )
        return task_id

    async def heartbeat(self, task_id: str, worker_id: str) -> bool:
        """Renew the lease; False if another worker took the job over."""
        owner = await self.r.get(self.k("lease", task_id))
        if owner not in (None, worker_id):
            return False
        await self.r.set(self.k("lease", task_id), worker_id, ex=self.lease_ttl)
        return True

    async def finish(self, task_id: str, status: str, **fields: Any) -> None:
        mapping = {"status": status, "updated": time.time()}
        mapping.update({k: json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v for k, v in fields.items()})
        pipe = self.r.pipeline(transaction=True)
        pipe.hset(self.k("task", task_id), mapping=mapping)
        pipe.lrem(self.k("processing"), 0, task_id)
        pipe.delete(self.k("lease", task_id))
        await pipe.execute()

    async def reap(self) -> List[str]:
        """Requeue jobs whose worker lease has expired (worker crashed or hung)."""
        requeued = []
        for task_id in await self.r.lrange(self.k("processing"), 0, -1):
            ok = await self._requeue(
                keys=[self.k("processing"), self.k("queue"), self.k("lease", task_id), self.k("task", task_id)],
                args=[task_id, time.time()],
            )
            if ok:
                requeued.append(task_id)
        return requeued

    async def depth(self) -> Dict[str, int]:
        return {
            "queued": await self.r.llen(self.k("queue")),
            "processing": await self.r.llen(self.k("processing")),
        }
