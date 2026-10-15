"""
Queue worker: claims jobs, keeps the lease alive, runs the reliable pipeline and
publishes progress events in order.

Crash semantics: if a worker dies mid-job its lease expires, ``JobQueue.reap``
moves the job back to the queue, and the next worker calls the runner with the
same task id. The runner resumes from the SQLite checkpoint, so shots finished
before the crash are reused, not paid for again.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Awaitable, Callable, Optional

from loguru import logger

from pixelle_video.reliability import ReliablePipeline, ShotsIncompleteError
from pixelle_video.serving.events import EventLog
from pixelle_video.serving.queue import JobQueue
from pixelle_video.serving.ratelimit import RateLimitedProcessor, TokenBucket

Runner = Callable[[str, dict, Callable[[dict], None]], Awaitable[dict]]


class Worker:
    def __init__(
        self,
        queue: JobQueue,
        events: EventLog,
        runner: Runner,
        worker_id: Optional[str] = None,
        heartbeat_every: Optional[float] = None,
    ):
        self.queue, self.events, self.runner = queue, events, runner
        self.worker_id = worker_id or f"w-{uuid.uuid4().hex[:8]}"
        self.heartbeat_every = heartbeat_every or max(1.0, queue.lease_ttl / 3)

    async def run_one(self, timeout: float = 1.0) -> Optional[str]:
        task_id = await self.queue.claim(self.worker_id, timeout=timeout)
        if task_id is None:
            return None
        task = await self.queue.get(task_id)
        request = json.loads(task["request"])

        outbox: asyncio.Queue = asyncio.Queue()

        def emit(event: dict) -> None:  # called synchronously from the pipeline
            outbox.put_nowait(event)

        async def pump():
            while True:
                event = await outbox.get()
                await self.events.emit(task_id, event)
                outbox.task_done()

        pumper = asyncio.create_task(pump())
        beat = asyncio.create_task(self._heartbeat(task_id))
        emit({"type": "task", "status": "running", "worker": self.worker_id,
              "requeues": int(task.get("requeues", 0))})
        try:
            result = await self.runner(task_id, request, emit)
            status, fields = "completed", {"result": result}
        except ShotsIncompleteError as e:
            status, fields = "partial", {"failed": e.failed, "pending": e.pending}
        except Exception as e:  # noqa: BLE001
            logger.exception(f"task {task_id} failed")
            status, fields = "failed", {"error": f"{type(e).__name__}: {e}"}
        except BaseException:
            # Process-level death (simulated in tests): leave the lease to expire so the
            # reaper hands the job to another worker; just stop our background tasks.
            pumper.cancel()
            raise
        finally:
            beat.cancel()
        await self.queue.finish(task_id, status, **fields)
        emit({"type": "task", "status": status, **fields})
        await outbox.join()
        pumper.cancel()
        return task_id

    async def _heartbeat(self, task_id: str) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_every)
            if not await self.queue.heartbeat(task_id, self.worker_id):
                logger.warning(f"lost lease on {task_id}")
                return

    async def run_forever(self, stop: asyncio.Event, reap_every: float = 5.0) -> None:
        last_reap = 0.0
        loop = asyncio.get_running_loop()
        while not stop.is_set():
            if loop.time() - last_reap > reap_every:
                for tid in await self.queue.reap():
                    logger.warning(f"requeued {tid} (lease expired)")
                last_reap = loop.time()
            await self.run_one(timeout=1.0)


def pixelle_runner(
    core_factory: Callable[[], object],
    store_path: str,
    bucket: Optional[TokenBucket] = None,
    **pipeline_kwargs,
) -> Runner:
    """Runner that executes a request with ReliablePipeline, resuming when a checkpoint exists."""

    async def run(task_id: str, request: dict, emit: Callable[[dict], None]) -> dict:
        core = core_factory()
        if bucket is not None:
            core.frame_processor = RateLimitedProcessor(core.frame_processor, bucket)
        pipe = ReliablePipeline(core, event_sink=emit, store_path=store_path, **pipeline_kwargs)
        if pipe.store.load_plan(task_id) is not None:
            res = await pipe.resume(task_id)
        else:
            res = await pipe(request["text"], **{**request.get("params", {}), "task_id": task_id})
        return {"video_path": res.video_path, "duration": res.duration}

    return run
