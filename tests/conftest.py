"""
Shared fixtures.

Each test runs with PIXELLE_VIDEO_ROOT and the working directory pointed at a
fresh temp dir, so task outputs and the persistence index never touch the repo.
FFmpeg concatenation is replaced by a stub: these tests are about orchestration,
not media encoding.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pixelle_video.services.persistence import PersistenceService
from pixelle_video.testing import FakeCore, FakeFrameProcessor

SCRIPT_5_SHOTS = "\n\n".join(f"Shot {i} narration." for i in range(5))

PIPELINE_KWARGS = dict(
    mode="fixed",
    split_mode="paragraph",
    title="Test video",
    frame_template="1080x1920/static_default.html",
    media_width=1080,
    media_height=1920,
)


@pytest.fixture
def workdir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("PIXELLE_VIDEO_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def stub_concat(monkeypatch):
    """Replace ffmpeg concat with a stub that just writes the output file."""
    from pixelle_video.services import video as video_module

    calls = []

    def fake_concat(self, videos, output, **kwargs):
        if any(v is None for v in videos):
            raise ValueError("cannot concat: missing segment")
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"fake-final-mp4")
        calls.append(list(videos))
        return output

    monkeypatch.setattr(video_module.VideoService, "concat_videos", fake_concat)
    return calls


@pytest.fixture
def make_pipeline(workdir, stub_concat):
    """Factory: build a StandardPipeline wired to a FakeFrameProcessor."""
    from pixelle_video.pipelines.standard import StandardPipeline

    def _make(processor: FakeFrameProcessor):
        persistence = PersistenceService(output_dir=str(workdir / "output"))
        core = FakeCore(frame_processor=processor, persistence=persistence)
        return StandardPipeline(core), persistence

    return _make


@pytest.fixture
def make_reliable(workdir, stub_concat):
    """Factory: ReliablePipeline wired to a FakeFrameProcessor, no real sleeping."""
    from pixelle_video.reliability import ReliablePipeline

    async def _no_sleep(_):
        return None

    def _make(processor: FakeFrameProcessor, **kwargs):
        persistence = PersistenceService(output_dir=str(workdir / "output"))
        core = FakeCore(frame_processor=processor, persistence=persistence)
        kwargs.setdefault("sleep", _no_sleep)
        return ReliablePipeline(core, **kwargs), persistence

    return _make
