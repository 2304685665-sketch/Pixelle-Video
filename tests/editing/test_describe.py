from __future__ import annotations

import asyncio

from pixelle_video.editing.describe import LLMAssetBaseline, caption_segments, describe_assets
from pixelle_video.editing.evaluate import Label, evaluate_method
from pixelle_video.editing.retrieval import BM25Channel, SegmentRetriever
from pixelle_video.testing.fake_encoders import FakeAssignLLM, FakeDescriber
from tests.editing.conftest import LABELS


def test_captions_feed_bm25_and_are_cached(fake_index):
    index, out = fake_index
    d = FakeDescriber()
    stats = asyncio.run(caption_segments(index, out, d))
    assert stats == {"cached": 0, "called": len(index.segments), "failed": 0}
    assert "绿色" in index.texts["caption"]["clips/rgb.mp4#001"]

    again = asyncio.run(caption_segments(index, out, FakeDescriber()))
    assert again["called"] == 0 and again["cached"] == len(index.segments)

    r = SegmentRetriever(index, [BM25Channel(index, "caption")])
    assert r.search("绿色的窗台", 1)[0].segment_id == "clips/rgb.mp4#001"


def test_caption_failures_are_isolated(fake_index, tmp_path):
    index, out = fake_index
    import shutil

    copy = tmp_path / "idx"
    shutil.copytree(out, copy)
    from pixelle_video.editing.index import SegmentIndex

    idx = SegmentIndex.load(copy)
    stats = asyncio.run(caption_segments(idx, copy, FakeDescriber(fail_on={"rgb.mp4_001"})))
    # the image keyframe is an absolute path shared with the first index, so it may be cached
    assert stats["failed"] == 1
    assert stats["called"] + stats["cached"] == len(idx.segments) - 1


def test_llm_asset_baseline_matches_upstream_granularity(fake_index, tmp_path):
    index, out = fake_index
    desc = asyncio.run(describe_assets(index, out, FakeDescriber()))
    assert set(desc) == set(index.assets())
    desc = {**desc, "clips/rgb.mp4": "红色 绿色 蓝色 的片段", "walk/white.mp4": "白色走廊",
            "stills/yellow.png": "黄色照片"}
    llm = FakeAssignLLM(desc, hallucinate=True)
    base = LLMAssetBaseline(index, desc, llm, cache_path=tmp_path / "llm.json", llm_name="fake")
    labels = [Label.from_dict(d) for d in LABELS]
    rep = evaluate_method(base, labels)
    assert rep.asset_acc_at1 == 1.0       # picks the right files ...
    assert rep.hit_at[1] == 2 / 5         # ... but whole files miss the 3 s shots
    assert base.invalid_paths == len(labels)  # hallucinated path dropped every time

    calls = llm.calls
    evaluate_method(base, labels)          # second run served from cache
    assert llm.calls == calls
