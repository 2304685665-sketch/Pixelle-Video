"""
ReliablePipeline: StandardPipeline with shot-level reliability.

Mechanisms (each can be switched off for ablation):

* ``isolate``     – a failed shot does not stop the remaining shots
* ``policy``      – per-error-class retry budget and backoff (``RetryPolicy``)
* ``rewrite``     – on CONTENT errors, rewrite the shot prompt before retrying
* ``checkpoint``  – plan + per-shot state in SQLite; ``resume(task_id)`` only
                    regenerates shots that are not done
* ``breaker``     – optional circuit breaker; while open, shots stay pending

A run that ends with unfinished shots raises ``ShotsIncompleteError`` *after*
committing all finished shots, so the caller can resume instead of rerunning.
"""

from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import Awaitable, Callable, List, Optional

from loguru import logger

from pixelle_video.pipelines.linear import PipelineContext
from pixelle_video.pipelines.standard import StandardPipeline
from pixelle_video.reliability.checkpoint import CheckpointStore
from pixelle_video.reliability.errors import (
    CircuitBreaker,
    CircuitOpenError,
    ErrorClass,
    RetryPolicy,
    classify,
)
from pixelle_video.utils.os_util import get_output_path, get_task_final_video_path


class ShotsIncompleteError(Exception):
    def __init__(self, task_id: str, failed: List[int], pending: List[int]):
        self.task_id, self.failed, self.pending = task_id, failed, pending
        super().__init__(f"task {task_id}: failed shots {failed}, pending shots {pending}")


def default_rewrite(frame) -> None:
    """Mark the prompt as rewritten. Real deployments plug in an LLM 'neutralise' step."""
    frame.prompt_rewrites = getattr(frame, "prompt_rewrites", 0) + 1


class ReliablePipeline(StandardPipeline):
    def __init__(
        self,
        core,
        *,
        isolate: bool = True,
        policy: Optional[RetryPolicy] = None,
        rewrite: Optional[Callable] = default_rewrite,
        checkpoint: bool = True,
        breaker: Optional[CircuitBreaker] = None,
        store_path: Optional[str | Path] = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        seed: int = 0,
        event_sink: Optional[Callable[[dict], None]] = None,
    ):
        super().__init__(core)
        self.isolate = isolate
        self.policy = policy or RetryPolicy()
        self.rewrite = rewrite
        self.checkpoint = checkpoint
        self.breaker = breaker
        self.sleep = sleep
        self._rng = random.Random(seed)
        self.store = (
            CheckpointStore(store_path or Path(get_output_path()) / ".checkpoints.sqlite")
            if checkpoint else None
        )
        self._resuming = False
        self.event_sink = event_sink

    def _emit(self, **event) -> None:
        if self.event_sink is not None:
            self.event_sink(event)

    async def setup_environment(self, ctx: PipelineContext):
        """Honour an externally assigned ``task_id`` (e.g. from a job queue) so a
        retried job maps onto the same checkpoint instead of a new task."""
        task_id = ctx.params.get("task_id")
        if not task_id:
            return await super().setup_environment(ctx)
        from pixelle_video.utils.os_util import create_task_output_dir

        ctx.task_dir, ctx.task_id = create_task_output_dir(task_id)
        ctx.final_video_path = get_task_final_video_path(task_id)

    # ------------------------------------------------------------------ #
    # Plan persistence: written before any paid shot call
    # ------------------------------------------------------------------ #

    async def initialize_storyboard(self, ctx: PipelineContext):
        await super().initialize_storyboard(ctx)
        if self.store and not self._resuming:
            plan = {
                "input_text": ctx.input_text,
                "params": ctx.params,
                "title": ctx.title,
                "narrations": ctx.narrations,
                "image_prompts": ctx.image_prompts,
            }
            self.store.save_plan(ctx.task_id, type(self).__name__, plan, len(ctx.storyboard.frames))

    async def resume(self, task_id: str, progress_callback=None):
        """Continue a checkpointed task; finished shots are reused, not regenerated."""
        if not self.store:
            raise RuntimeError("resume requires checkpoint=True")
        plan = self.store.load_plan(task_id)
        if plan is None:
            raise KeyError(f"no checkpoint for task {task_id}")
        ctx = PipelineContext(input_text=plan["input_text"], params=plan["params"], progress_callback=progress_callback)
        ctx.task_id, ctx.task_dir = task_id, get_output_path(task_id)
        ctx.final_video_path = get_task_final_video_path(task_id)
        ctx.title, ctx.narrations, ctx.image_prompts = plan["title"], plan["narrations"], plan["image_prompts"]
        self._resuming = True
        try:
            await self.initialize_storyboard(ctx)
            await self.produce_assets(ctx)
            await self.post_production(ctx)
            return await self.finalize(ctx)
        except Exception as e:
            await self.handle_exception(ctx, e)
            raise
        finally:
            self._resuming = False

    # ------------------------------------------------------------------ #
    # Shot execution
    # ------------------------------------------------------------------ #

    async def produce_assets(self, ctx: PipelineContext):
        storyboard, task_id = ctx.storyboard, ctx.task_id
        done = self.store.shots(task_id) if self.store else {}
        failed: List[int] = []
        pending: List[int] = []

        for i, frame in enumerate(storyboard.frames):
            rec = done.get(i)
            if rec and rec.status == "done" and rec.segment_path and Path(rec.segment_path).exists():
                frame.video_segment_path, frame.duration = rec.segment_path, rec.duration or 0.0
                storyboard.total_duration += frame.duration
                self._emit(type="shot", shot=i, status="reused")
                continue

            if failed and not self.isolate:
                pending.append(i)
                continue

            outcome = await self._run_shot(ctx, i, frame)
            if outcome == "done":
                storyboard.total_duration += frame.duration
            elif outcome == "failed":
                failed.append(i)
            else:
                pending.append(i)

        if failed or pending:
            if self.store:
                self.store.set_task_status(task_id, "partial")
            raise ShotsIncompleteError(task_id, failed, pending)

    async def _run_shot(self, ctx: PipelineContext, i: int, frame) -> str:
        task_id, attempt = ctx.task_id, 0
        while True:
            attempt += 1
            try:
                if self.breaker:
                    self.breaker.before_call()
                processed = await self.core.frame_processor(
                    frame=frame, storyboard=ctx.storyboard, config=ctx.config,
                    total_frames=len(ctx.storyboard.frames), progress_callback=None,
                )
            except CircuitOpenError:
                logger.warning(f"shot {i}: circuit open, left pending")
                self._emit(type="shot", shot=i, status="pending", attempt=attempt, reason="circuit_open")
                return "pending"
            except Exception as exc:  # noqa: BLE001 - classified below
                cls = classify(exc)
                if self.breaker:
                    self.breaker.record(cls)
                if self.store:
                    self.store.record_attempt(task_id, i, cls.value, str(exc))
                if attempt > self.policy.retries_for(cls):
                    logger.warning(f"shot {i}: giving up after {attempt} attempt(s) [{cls.value}] {exc}")
                    if self.store:
                        self.store.mark_failed(task_id, i)
                    self._emit(type="shot", shot=i, status="failed", attempt=attempt, error_class=cls.value)
                    return "failed"
                self._emit(type="shot", shot=i, status="retrying", attempt=attempt, error_class=cls.value)
                if cls is ErrorClass.CONTENT and self.rewrite is not None:
                    self.rewrite(frame)
                delay = self.policy.backoff(cls, attempt, self._rng)
                if delay:
                    await self.sleep(delay)
                continue

            if self.breaker:
                self.breaker.record(None)
            ctx.storyboard.frames[i] = processed
            if self.store:
                self.store.record_attempt(task_id, i, None, None)
                self.store.mark_done(task_id, i, processed.video_segment_path, processed.duration)
            self._emit(type="shot", shot=i, status="done", attempt=attempt)
            return "done"

    async def finalize(self, ctx: PipelineContext):
        result = await super().finalize(ctx)
        if self.store:
            self.store.set_task_status(ctx.task_id, "completed")
        return result
