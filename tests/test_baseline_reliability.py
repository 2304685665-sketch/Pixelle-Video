"""
Baseline reliability behaviour of StandardPipeline (upstream commit 848b054).

Two kinds of tests live here:

* Characterization tests (normal asserts) pin down what the upstream pipeline
  does today when a shot fails. They pass on the baseline and document the
  defects with evidence.
* Target-spec tests (``xfail(strict=True)``) describe the behaviour the
  reliability work must deliver. They fail on the baseline; when a change makes
  one pass, strict xfail turns it into an error so the marker must be removed.
"""

from __future__ import annotations

import pytest

from pixelle_video.testing import FakeFrameProcessor
from pixelle_video.testing import InjectedFault as _Fault
from tests.conftest import PIPELINE_KWARGS, SCRIPT_5_SHOTS

# --------------------------------------------------------------------------- #
# Sanity
# --------------------------------------------------------------------------- #


async def test_happy_path_generates_each_shot_once(make_pipeline):
    processor = FakeFrameProcessor()
    pipeline, persistence = make_pipeline(processor)

    result = await pipeline(SCRIPT_5_SHOTS, **PIPELINE_KWARGS)

    assert len(result.storyboard.frames) == 5
    assert processor.total_calls == 5
    assert processor.duplicate_successes() == 0
    assert await persistence.load_storyboard(result.storyboard.config.task_id) is not None


# --------------------------------------------------------------------------- #
# Characterization: current behaviour on a single shot failure
# --------------------------------------------------------------------------- #


async def test_single_shot_failure_aborts_whole_video(make_pipeline):
    """Defect 1: a timeout on shot 2 fails the entire video; shots 3-4 never run."""
    processor = FakeFrameProcessor(fail_on={2: ["timeout"]})
    pipeline, _ = make_pipeline(processor)

    with pytest.raises(_Fault):
        await pipeline(SCRIPT_5_SHOTS, **PIPELINE_KWARGS)

    assert processor.successful_frames() == [0, 1]
    assert processor.calls_for(3) == 0
    assert processor.calls_for(4) == 0


async def test_failed_run_persists_no_storyboard(make_pipeline):
    """Defect 2: state is only persisted in finalize(), so a failed run leaves nothing to resume from."""
    processor = FakeFrameProcessor(fail_on={2: ["timeout"]})
    pipeline, persistence = make_pipeline(processor)

    with pytest.raises(_Fault):
        await pipeline(SCRIPT_5_SHOTS, **PIPELINE_KWARGS)

    task_ids = [t["task_id"] for t in await persistence.list_tasks()]
    assert task_ids == []


async def test_rerun_after_failure_pays_again_for_finished_shots(make_pipeline):
    """Defect 3: the only recovery is a full rerun, which regenerates shots that already succeeded."""
    processor = FakeFrameProcessor(fail_on={2: ["timeout"]})
    pipeline, _ = make_pipeline(processor)

    with pytest.raises(_Fault):
        await pipeline(SCRIPT_5_SHOTS, **PIPELINE_KWARGS)
    # Second run: new task id, every shot is generated again.
    processor.fail_on.clear()
    await pipeline(SCRIPT_5_SHOTS, **PIPELINE_KWARGS)

    # 3 calls in run 1 (0, 1, failed 2) + 5 calls in run 2.
    assert processor.total_calls == 8
    # Shots 0 and 1 were paid for twice.
    assert sorted(c.frame_index for c in processor.ledger if c.outcome == "ok")[:4] == [0, 0, 1, 1]


async def test_moderation_rejection_is_not_distinguished_from_transient_errors(make_pipeline):
    """Defect 4: every failure kind ends the run the same way; no per-kind policy exists at pipeline level."""
    outcomes = {}
    for kind in ("timeout", "rate_limit", "moderation", "corrupt"):
        processor = FakeFrameProcessor(fail_on={0: [kind]})
        pipeline, _ = make_pipeline(processor)
        with pytest.raises(_Fault):
            await pipeline(SCRIPT_5_SHOTS, **PIPELINE_KWARGS)
        outcomes[kind] = processor.total_calls

    # Identical handling: one call, then abort, regardless of kind.
    assert set(outcomes.values()) == {1}
