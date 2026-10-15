"""
ReliablePipeline behaviour. The first three tests are the target spec that the
baseline suite (test_baseline_reliability.py) shows upstream failing.
"""

from __future__ import annotations

import random

import pytest

from pixelle_video.reliability import (
    CheckpointStore,
    CircuitBreaker,
    CircuitOpenError,
    ErrorClass,
    RetryPolicy,
    ShotsIncompleteError,
    classify,
)
from pixelle_video.testing import FakeFrameProcessor, FaultProfile
from pixelle_video.testing.fault_injection import (
    BadInputError,
    CorruptOutputError,
    ModerationRejectedError,
    ProviderTimeoutError,
    RateLimitedError,
)

from tests.conftest import PIPELINE_KWARGS, SCRIPT_5_SHOTS


async def _tasks(pipeline):
    return pipeline.store.tasks()


# --------------------------------------------------------------------------- #
# Target spec (upstream fails all three)
# --------------------------------------------------------------------------- #


async def test_failed_shot_does_not_block_other_shots(make_reliable):
    processor = FakeFrameProcessor(fail_on={2: ["moderation", "moderation", "moderation"]})
    pipeline, _ = make_reliable(processor)

    with pytest.raises(ShotsIncompleteError) as err:
        await pipeline(SCRIPT_5_SHOTS, **PIPELINE_KWARGS)

    assert err.value.failed == [2]
    assert processor.successful_frames() == [0, 1, 3, 4]


async def test_transient_error_is_retried_and_video_completes(make_reliable):
    processor = FakeFrameProcessor(fail_on={2: ["timeout", None]})
    pipeline, _ = make_reliable(processor)

    result = await pipeline(SCRIPT_5_SHOTS, **PIPELINE_KWARGS)

    assert len(result.storyboard.frames) == 5
    assert processor.calls_for(2) == 2


async def test_resume_regenerates_only_unfinished_shots(make_reliable):
    processor = FakeFrameProcessor(fail_on={2: ["timeout"] * 4})
    pipeline, _ = make_reliable(processor)

    with pytest.raises(ShotsIncompleteError):
        await pipeline(SCRIPT_5_SHOTS, **PIPELINE_KWARGS)
    (task_id,) = await _tasks(pipeline)
    assert pipeline.store.task_status(task_id) == "partial"

    processor.fail_on.clear()
    result = await pipeline.resume(task_id)

    assert processor.duplicate_successes() == 0
    assert processor.calls_for(0) == 1 and processor.calls_for(4) == 1
    assert result.storyboard.total_duration == pytest.approx(15.0)
    assert pipeline.store.task_status(task_id) == "completed"


# --------------------------------------------------------------------------- #
# Crash safety
# --------------------------------------------------------------------------- #


class Crash(BaseException):
    """Simulated process kill (BaseException so no handler swallows it)."""


@pytest.mark.parametrize("crash_at_call", [1, 2, 3, 4, 5])
async def test_crash_then_resume_never_repays_finished_shots(make_reliable, crash_at_call):
    calls = {"n": 0}

    def maybe_crash(idx, attempt):
        calls["n"] += 1
        if calls["n"] == crash_at_call:
            raise Crash()

    processor = FakeFrameProcessor(on_call=maybe_crash)
    pipeline, _ = make_reliable(processor)
    with pytest.raises(Crash):
        await pipeline(SCRIPT_5_SHOTS, **PIPELINE_KWARGS)

    processor.on_call = None
    (task_id,) = await _tasks(pipeline)
    await pipeline.resume(task_id)

    assert processor.duplicate_successes() == 0
    assert processor.successful_frames() == [0, 1, 2, 3, 4]


async def test_missing_artifact_is_regenerated_on_resume(make_reliable, tmp_path):
    from pathlib import Path

    processor = FakeFrameProcessor(fail_on={4: ["bad_input"]})
    pipeline, _ = make_reliable(processor)
    with pytest.raises(ShotsIncompleteError):
        await pipeline(SCRIPT_5_SHOTS, **PIPELINE_KWARGS)
    (task_id,) = await _tasks(pipeline)
    Path(pipeline.store.shots(task_id)[1].segment_path).unlink()  # artifact lost on disk

    processor.fail_on.clear()
    await pipeline.resume(task_id)
    assert processor.calls_for(1) == 2  # regenerated because its file vanished
    assert processor.calls_for(0) == 1


# --------------------------------------------------------------------------- #
# Error classes and policy
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "exc, expected",
    [
        (ProviderTimeoutError("injected timeout"), ErrorClass.TRANSIENT),
        (RateLimitedError("injected rate_limit"), ErrorClass.TRANSIENT),
        (TimeoutError(), ErrorClass.TRANSIENT),
        (RuntimeError("HTTP 429 Too Many Requests"), ErrorClass.TRANSIENT),
        (RuntimeError("503 Service Temporarily Unavailable"), ErrorClass.TRANSIENT),
        (ModerationRejectedError("injected moderation"), ErrorClass.CONTENT),
        (RuntimeError("DataInspectionFailed: Input data may contain inappropriate content"), ErrorClass.CONTENT),
        (CorruptOutputError("injected corrupt"), ErrorClass.OUTPUT),
        (RuntimeError("moov atom not found"), ErrorClass.OUTPUT),
        (BadInputError("injected bad_input"), ErrorClass.INPUT),
        (ValueError("width must be positive"), ErrorClass.INPUT),
    ],
)
def test_classify(exc, expected):
    assert classify(exc) is expected


async def test_input_error_is_not_retried(make_reliable):
    processor = FakeFrameProcessor(fail_on={1: ["bad_input"] * 10})
    pipeline, _ = make_reliable(processor)
    with pytest.raises(ShotsIncompleteError):
        await pipeline(SCRIPT_5_SHOTS, **PIPELINE_KWARGS)
    assert processor.calls_for(1) == 1


async def test_uniform_retry_wastes_calls_on_input_errors(make_reliable):
    processor = FakeFrameProcessor(fail_on={1: ["bad_input"] * 10})
    pipeline, _ = make_reliable(processor, policy=RetryPolicy.uniform(3))
    with pytest.raises(ShotsIncompleteError):
        await pipeline(SCRIPT_5_SHOTS, **PIPELINE_KWARGS)
    assert processor.calls_for(1) == 4


async def test_content_error_rewrites_prompt_before_retry(make_reliable):
    profile = FaultProfile(failure_rate=0.0, rewrite_fix_rate=1.0, seed=1)
    processor = FakeFrameProcessor(profile=profile, fail_on={3: ["moderation"]})
    pipeline, _ = make_reliable(processor)
    result = await pipeline(SCRIPT_5_SHOTS, **PIPELINE_KWARGS)
    assert getattr(result.storyboard.frames[3], "prompt_rewrites", 0) == 1
    assert processor.calls_for(3) == 2


async def test_sticky_moderation_without_rewrite_is_futile(make_reliable):
    profile = FaultProfile(failure_rate=0.0, seed=1)
    processor = FakeFrameProcessor(profile=profile)
    processor._moderated[("Shot 3 narration.", 0)] = True  # this prompt is always rejected
    pipeline, _ = make_reliable(processor, rewrite=None, policy=RetryPolicy.uniform(3))
    with pytest.raises(ShotsIncompleteError):
        await pipeline(SCRIPT_5_SHOTS, **PIPELINE_KWARGS)
    assert processor.futile_calls() == 4


def test_backoff_is_exponential_and_capped():
    pol = RetryPolicy(base_delay=1.0, max_delay=5.0, jitter=0.0)
    rng = random.Random(0)
    assert [pol.backoff(ErrorClass.TRANSIENT, a, rng) for a in (1, 2, 3, 4)] == [1, 2, 4, 5]
    assert pol.backoff(ErrorClass.CONTENT, 1, rng) == 0.0


# --------------------------------------------------------------------------- #
# Circuit breaker
# --------------------------------------------------------------------------- #


def test_breaker_opens_half_opens_and_closes():
    now = {"t": 0.0}
    br = CircuitBreaker(threshold=2, cooldown=10, clock=lambda: now["t"])
    br.record(ErrorClass.TRANSIENT)
    br.before_call()
    br.record(ErrorClass.TRANSIENT)
    assert br.state == "open"
    with pytest.raises(CircuitOpenError):
        br.before_call()
    now["t"] = 10.0
    assert br.state == "half-open"
    br.before_call()
    br.record(ErrorClass.TRANSIENT)  # probe failed -> open again
    assert br.state == "open"
    now["t"] = 20.0
    br.record(None)  # probe succeeded
    assert br.state == "closed"


async def test_open_breaker_leaves_shots_pending_and_unpaid(make_reliable):
    processor = FakeFrameProcessor(fail_on={0: ["timeout"] * 4, 1: ["timeout"] * 4})
    breaker = CircuitBreaker(threshold=3, cooldown=1e9)
    pipeline, _ = make_reliable(processor, breaker=breaker)
    with pytest.raises(ShotsIncompleteError) as err:
        await pipeline(SCRIPT_5_SHOTS, **PIPELINE_KWARGS)
    # Breaker opens on shot 0's third transient failure: shot 0 is left pending
    # (resumable) rather than marked failed, and no further calls are paid.
    assert err.value.failed == []
    assert err.value.pending == [0, 1, 2, 3, 4]
    assert processor.total_calls == 3


def test_checkpoint_store_roundtrip(tmp_path):
    store = CheckpointStore(tmp_path / "c.sqlite")
    store.save_plan("t1", "P", {"a": 1}, n_shots=3)
    store.record_attempt("t1", 1, "transient", "boom")
    store.mark_done("t1", 0, "/x.mp4", 2.5)
    store.mark_failed("t1", 1)
    shots = store.shots("t1")
    assert (shots[0].status, shots[0].duration) == ("done", 2.5)
    assert (shots[1].status, shots[1].attempts, shots[1].error_class) == ("failed", 1, "transient")
    assert shots[2].status == "pending"
    assert store.load_plan("t1") == {"a": 1}
    store.save_plan("t1", "P", {"a": 1}, n_shots=3)  # idempotent: shot state preserved
    assert store.shots("t1")[0].status == "done"
