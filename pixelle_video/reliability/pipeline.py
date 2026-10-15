"""
ReliablePipeline: StandardPipeline with shot-level reliability.

Mechanisms (each can be switched off for ablation):

* ``isolate``     – a failed shot does not stop the remaining shots
* ``policy``      – per-error-class retry budget and backoff (``RetryPolicy``)
* ``rewrite``     – on CONTENT errors, rewrite the shot prompt before retrying
* ``checkpoint``  – plan + per-shot state in SQLite; ``resume(task_id)`` only
                    regenerates shots that are not done
* ``breaker``     – optional circuit breaker; while open, shots stay pending
* ``concurrency`` – shots generated in parallel (bounded); with ``isolate=False``
                    the first failure stops new shots from starting

Finished segments are stored content-addressed (``<task>/cas/<fingerprint>``)
and indexed in the checkpoint, so ``revise(task_id, narrations)`` after a script
edit only regenerates shots whose inputs changed, even when shots are inserted
or deleted.

A run that ends with unfinished shots raises ``ShotsIncompleteError`` *after*
committing all finished shots, so the caller can resume instead of rerunning.
"""

from __future__ import annotations

import asyncio
import random
import shutil
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
from pixelle_video.reliability.fingerprint import shot_fingerprint
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
        concurrency: int = 1,
    ):
        super().__init__(core)
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self.concurrency = concurrency
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

    async def revise(self, task_id: str, narrations: List[str], progress_callback=None):
        """
        Apply a script edit (new narration list) to a checkpointed task and
        re-render. Shots whose fingerprint already has a cached segment are
        reused; only new or changed shots are generated. Image prompts of
        unchanged narrations are kept; prompts for changed ones are produced by
        the normal ``plan_visuals`` step on just those narrations.
        """
        if not self.store:
            raise RuntimeError("revise requires checkpoint=True")
        plan = self.store.load_plan(task_id)
        if plan is None:
            raise KeyError(f"no checkpoint for task {task_id}")
        if not narrations:
            raise ValueError("narrations must not be empty")

        known = dict(zip(plan["narrations"], plan["image_prompts"]))
        changed = [n for n in dict.fromkeys(narrations) if n not in known]
        if changed:
            tmp = PipelineContext(input_text=plan["input_text"], params=plan["params"])
            tmp.narrations = changed
            await self.plan_visuals(tmp)
            known.update(zip(changed, tmp.image_prompts))

        plan = {**plan, "narrations": list(narrations), "image_prompts": [known[n] for n in narrations],
                "revision": int(plan.get("revision", 0)) + 1}
        self.store.replace_plan(task_id, plan, len(narrations))
        self._emit(type="task", status="revised", revision=plan["revision"], shots=len(narrations),
                   new_or_changed=len(changed))
        return await self.resume(task_id, progress_callback=progress_callback)

    # ------------------------------------------------------------------ #
    # Shot execution
    # ------------------------------------------------------------------ #

    async def produce_assets(self, ctx: PipelineContext):
        storyboard, task_id = ctx.storyboard, ctx.task_id
        fps = [shot_fingerprint(f, ctx.config) for f in storyboard.frames]

        to_run: List[int] = []
        for i, frame in enumerate(storyboard.frames):
            art = self.store.get_artifact(task_id, fps[i]) if self.store else None
            if art and Path(art[0]).exists():
                frame.video_segment_path, frame.duration = art
                self.store.mark_done(task_id, i, art[0], art[1], fps[i])
                self._emit(type="shot", shot=i, status="reused")
            else:
                to_run.append(i)

        failed: List[int] = []
        pending: List[int] = []
        sem = asyncio.Semaphore(self.concurrency)
        stop = False
        aborted = False  # set when any shot raises; checked before starting a shot

        async def run(i: int) -> None:
            nonlocal stop, aborted
            async with sem:
                if stop or aborted:
                    pending.append(i)
                    return
                try:
                    outcome = await self._run_shot(ctx, i, storyboard.frames[i], fps[i])
                except BaseException:
                    # The semaphore is released on the way out and a queued shot may be
                    # scheduled before the caller gets to cancel it; the flag closes that race.
                    aborted = True
                    raise
                if outcome == "failed":
                    failed.append(i)
                    stop = stop or not self.isolate
                elif outcome == "pending":
                    pending.append(i)

        # Semaphore acquisition is FIFO, so concurrency=1 keeps the original shot order.
        tasks = [asyncio.create_task(run(i)) for i in to_run]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            # Unexpected error / cancellation / process shutdown: stop sibling shots so no
            # orphaned task keeps calling (and paying) the provider in the background.
            for t in tasks:
                t.cancel()
            await asyncio.wait(tasks)
            raise
        storyboard.total_duration = sum(f.duration for f in storyboard.frames if f.video_segment_path)

        if failed or pending:
            if self.store:
                self.store.set_task_status(task_id, "partial")
            raise ShotsIncompleteError(task_id, sorted(failed), sorted(pending))

    def _store_content_addressed(self, ctx: PipelineContext, segment_path: str, fp: str) -> str:
        """Copy a fresh segment to <task>/cas/<fingerprint><ext>.

        Upstream names segments by shot index, so after an insertion a new shot
        would overwrite an old shot's file; the content-addressed copy keeps
        every cached segment valid regardless of index changes.
        """
        src = Path(segment_path)
        dst = Path(ctx.task_dir or get_output_path(ctx.task_id)) / "cas" / f"{fp}{src.suffix}"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
        return str(dst)

    async def _run_shot(self, ctx: PipelineContext, i: int, frame, fp: Optional[str] = None) -> str:
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
                if fp:
                    processed.video_segment_path = self._store_content_addressed(
                        ctx, processed.video_segment_path, fp
                    )
                self.store.record_attempt(task_id, i, None, None)
                self.store.mark_done(task_id, i, processed.video_segment_path, processed.duration, fp)
            self._emit(type="shot", shot=i, status="done", attempt=attempt)
            return "done"

    async def finalize(self, ctx: PipelineContext):
        result = await super().finalize(ctx)
        if self.store:
            self.store.set_task_status(ctx.task_id, "completed")
        return result
