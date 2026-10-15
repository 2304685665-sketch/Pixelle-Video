"""Incremental regeneration after script edits, and bounded parallel shot execution."""

from __future__ import annotations

import time

import pytest

from pixelle_video.reliability import ShotsIncompleteError
from pixelle_video.reliability.fingerprint import shot_fingerprint, template_uses_index
from pixelle_video.testing import FakeFrameProcessor

from tests.conftest import PIPELINE_KWARGS, SCRIPT_5_SHOTS

LINES = [f"Shot {i} narration." for i in range(5)]


async def _first_run(make_reliable, **kw):
    processor = FakeFrameProcessor()
    pipeline, _ = make_reliable(processor, **kw)
    await pipeline(SCRIPT_5_SHOTS, **PIPELINE_KWARGS)
    (task_id,) = pipeline.store.tasks()
    return pipeline, processor, task_id


def _new_calls(processor, before):
    return [c.frame_index for c in processor.ledger[before:]]


# --------------------------------------------------------------------------- #
# Incremental regeneration
# --------------------------------------------------------------------------- #


async def test_edit_one_line_regenerates_one_shot(make_reliable):
    pipeline, proc, task_id = await _first_run(make_reliable)
    before = proc.total_calls
    edited = LINES.copy()
    edited[2] = "A rewritten third line."
    result = await pipeline.revise(task_id, edited)
    assert _new_calls(proc, before) == [2]
    assert [f.narration for f in result.storyboard.frames] == edited


async def test_insert_reuses_all_existing_shots(make_reliable):
    pipeline, proc, task_id = await _first_run(make_reliable)
    before = proc.total_calls
    edited = LINES[:2] + ["An inserted line."] + LINES[2:]
    result = await pipeline.revise(task_id, edited)
    assert _new_calls(proc, before) == [2]  # only the inserted shot
    assert len(result.storyboard.frames) == 6
    assert result.storyboard.total_duration == pytest.approx(18.0)


async def test_delete_and_reorder_need_no_generation(make_reliable):
    pipeline, proc, task_id = await _first_run(make_reliable)
    before = proc.total_calls
    edited = [LINES[4], LINES[0], LINES[3]]
    await pipeline.revise(task_id, edited)
    assert _new_calls(proc, before) == []


async def test_reused_segments_survive_index_shifts(make_reliable):
    """Upstream writes segments by index; an insertion must not clobber cached ones."""
    pipeline, proc, task_id = await _first_run(make_reliable)
    old = {r.input_hash: r.segment_path for r in pipeline.store.shots(task_id).values()}
    await pipeline.revise(task_id, ["New first line."] + LINES)
    shots = pipeline.store.shots(task_id)
    for idx in range(1, 6):
        assert shots[idx].segment_path == old[shots[idx].input_hash]
        assert "/cas/" in shots[idx].segment_path


async def test_generation_param_change_invalidates_cache(make_reliable):
    pipeline, proc, task_id = await _first_run(make_reliable)
    before = proc.total_calls
    plan = pipeline.store.load_plan(task_id)
    plan["params"]["tts_speed"] = 1.5  # different voice speed -> different audio/duration
    pipeline.store.replace_plan(task_id, plan, len(plan["narrations"]))
    await pipeline.resume(task_id)
    assert sorted(_new_calls(proc, before)) == [0, 1, 2, 3, 4]


async def test_revise_after_partial_failure(make_reliable):
    proc = FakeFrameProcessor(fail_on={3: ["bad_input"]})
    pipeline, _ = make_reliable(proc)
    with pytest.raises(ShotsIncompleteError):
        await pipeline(SCRIPT_5_SHOTS, **PIPELINE_KWARGS)
    (task_id,) = pipeline.store.tasks()
    before = proc.total_calls
    edited = LINES.copy()
    edited[3] = "Fixed line that the provider accepts."
    await pipeline.revise(task_id, edited)
    assert _new_calls(proc, before) == [3]
    assert pipeline.store.task_status(task_id) == "completed"


def test_fingerprint_covers_index_only_for_index_templates():
    class F:
        def __init__(self, idx):
            self.index, self.narration, self.image_prompt = idx, "same", None

    class C:
        frame_template = "1080x1920/static_default.html"

    assert template_uses_index("1080x1920/static_default.html") is False
    assert template_uses_index("1080x1920/image_healing.html") is True
    assert template_uses_index("1080x1920/does_not_exist.html") is True  # conservative
    assert shot_fingerprint(F(0), C()) == shot_fingerprint(F(7), C())
    C.frame_template = "1080x1920/image_healing.html"
    assert shot_fingerprint(F(0), C()) != shot_fingerprint(F(7), C())


def test_checkpoint_migrates_old_schema(tmp_path):
    import sqlite3

    from pixelle_video.reliability import CheckpointStore

    db = tmp_path / "old.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE shots (task_id TEXT, idx INTEGER, status TEXT, attempts INTEGER DEFAULT 0, "
        "error_class TEXT, last_error TEXT, segment_path TEXT, duration REAL, updated_at REAL, "
        "PRIMARY KEY (task_id, idx));"
    )
    conn.close()
    store = CheckpointStore(db)
    store.save_plan("t", "P", {}, 1)
    store.mark_done("t", 0, "/x.mp4", 1.0, "h1")
    assert store.get_artifact("t", "h1") == ("/x.mp4", 1.0)


# --------------------------------------------------------------------------- #
# Parallel execution
# --------------------------------------------------------------------------- #


async def test_concurrency_is_bounded_and_faster(make_reliable):
    timings = {}
    for c in (1, 3):
        proc = FakeFrameProcessor(latency=lambda: 0.05)
        pipeline, _ = make_reliable(proc, concurrency=c, store_path=f"cp{c}.sqlite")
        t0 = time.perf_counter()
        await pipeline(SCRIPT_5_SHOTS, **PIPELINE_KWARGS)
        timings[c] = time.perf_counter() - t0
        assert proc.max_in_flight == c
    assert timings[3] < timings[1] * 0.7


async def test_parallel_results_keep_shot_order(make_reliable):
    lat = iter([0.05, 0.01, 0.04, 0.02, 0.03])
    proc = FakeFrameProcessor(latency=lambda: next(lat))
    pipeline, _ = make_reliable(proc, concurrency=5)
    result = await pipeline(SCRIPT_5_SHOTS, **PIPELINE_KWARGS)
    assert [f.narration for f in result.storyboard.frames] == LINES


async def test_fail_fast_stops_starting_new_shots(make_reliable):
    proc = FakeFrameProcessor(fail_on={0: ["bad_input"]}, latency=lambda: 0.01)
    pipeline, _ = make_reliable(proc, concurrency=2, isolate=False)
    with pytest.raises(ShotsIncompleteError) as err:
        await pipeline(SCRIPT_5_SHOTS, **PIPELINE_KWARGS)
    assert err.value.failed == [0]
    # shot 1 was already in flight; shots 2-4 were never started
    assert err.value.pending == [2, 3, 4]
    assert proc.total_calls == 2


class Crash(BaseException):
    pass


async def test_crash_cancels_in_flight_siblings(make_reliable):
    calls = {"n": 0}

    def crash_on_second(idx, attempt):
        calls["n"] += 1
        if idx == 1:
            raise Crash()

    proc = FakeFrameProcessor(on_call=crash_on_second, latency=lambda: 0.05)
    pipeline, _ = make_reliable(proc, concurrency=3)
    with pytest.raises(Crash):
        await pipeline(SCRIPT_5_SHOTS, **PIPELINE_KWARGS)
    import asyncio

    await asyncio.sleep(0.2)  # any orphaned task would finish (and be billed) by now
    assert proc.total_calls == 0  # shots 0 and 2 were cancelled mid-call, never completed
