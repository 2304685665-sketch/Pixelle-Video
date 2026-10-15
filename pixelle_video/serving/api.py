"""
HTTP API v2 for queued video generation.

    POST /v2/videos                 Idempotency-Key header required -> 202 {task_id, deduplicated}
    GET  /v2/videos/{id}            task status
    GET  /v2/videos/{id}/events     SSE progress; honours Last-Event-ID
    POST /v2/videos/{id}/resume     requeue a partial task (resumes from checkpoint)
"""

from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from pixelle_video.serving.events import EventLog, sse_format
from pixelle_video.serving.queue import IdempotencyConflict, JobQueue


class VideoRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    params: Dict[str, Any] = Field(default_factory=dict)


def build_router(queue: JobQueue, events: EventLog) -> APIRouter:
    router = APIRouter(prefix="/v2/videos", tags=["videos-v2"])

    @router.post("", status_code=202)
    async def submit(body: VideoRequest, idempotency_key: str | None = Header(default=None)):
        if not idempotency_key or len(idempotency_key) > 200:
            raise HTTPException(400, "Idempotency-Key header (<=200 chars) is required")
        if "task_id" in body.params:
            raise HTTPException(400, "params.task_id is assigned by the server")
        try:
            sub = await queue.submit(body.model_dump(), idempotency_key)
        except IdempotencyConflict:
            raise HTTPException(409, "Idempotency-Key reused with a different request body")
        return {"task_id": sub.task_id, "deduplicated": sub.deduplicated}

    @router.get("/{task_id}")
    async def status(task_id: str):
        task = await queue.get(task_id)
        if task is None:
            raise HTTPException(404, "unknown task")
        task.pop("request", None)
        for k in ("result", "failed", "pending"):
            if k in task:
                task[k] = json.loads(task[k])
        return {"task_id": task_id, **task}

    @router.get("/{task_id}/events")
    async def stream(task_id: str, request: Request, last_event_id: str | None = Header(default=None)):
        if await queue.get(task_id) is None:
            raise HTTPException(404, "unknown task")

        async def gen():
            async for eid, event in events.read(task_id, last_id=last_event_id):
                if await request.is_disconnected():
                    return
                yield sse_format(eid, event)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @router.post("/{task_id}/resume", status_code=202)
    async def resume(task_id: str):
        if await queue.get(task_id) is None:
            raise HTTPException(404, "unknown task")
        if not await queue.requeue_partial(task_id):
            raise HTTPException(409, "only partial tasks can be resumed")
        return {"task_id": task_id, "status": "queued"}

    return router


def create_app(redis) -> FastAPI:
    app = FastAPI(title="Pixelle-Video queue API")
    app.include_router(build_router(JobQueue(redis), EventLog(redis)))
    return app
