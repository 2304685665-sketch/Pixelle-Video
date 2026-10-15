"""
Per-task progress events on a Redis Stream, served as Server-Sent Events.

The stream entry id doubles as the SSE ``id:``. A client that drops the
connection reconnects with ``Last-Event-ID`` and continues with ``XREAD`` after
that id, so no event is lost or repeated, and any API replica can serve the
reconnect (no sticky sessions).
"""

from __future__ import annotations

import json
from typing import AsyncIterator, Optional

TERMINAL_EVENTS = {"completed", "failed", "partial"}


class EventLog:
    def __init__(self, redis, prefix: str = "pv:", maxlen: int = 2000, ttl: int = 7 * 24 * 3600):
        self.r, self.p, self.maxlen, self.ttl = redis, prefix, maxlen, ttl

    def key(self, task_id: str) -> str:
        return f"{self.p}events:{task_id}"

    async def emit(self, task_id: str, event: dict) -> str:
        key = self.key(task_id)
        eid = await self.r.xadd(key, {"data": json.dumps(event, ensure_ascii=False)},
                                maxlen=self.maxlen, approximate=True)
        await self.r.expire(key, self.ttl)
        return eid

    async def read(
        self, task_id: str, last_id: Optional[str] = None, block_ms: int = 1000, stop_on_terminal: bool = True
    ) -> AsyncIterator[tuple[str, dict]]:
        """Yield (event_id, event) after ``last_id`` until a terminal event is seen."""
        cursor = last_id or "0-0"
        while True:
            resp = await self.r.xread({self.key(task_id): cursor}, block=block_ms, count=100)
            if not resp:
                continue
            for _, entries in resp:
                for eid, fields in entries:
                    cursor = eid
                    event = json.loads(fields["data"])
                    yield eid, event
                    if stop_on_terminal and event.get("type") == "task" and event.get("status") in TERMINAL_EVENTS:
                        return


def sse_format(event_id: str, event: dict) -> str:
    return f"id: {event_id}\nevent: {event.get('type', 'message')}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
