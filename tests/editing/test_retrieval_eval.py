from __future__ import annotations

import json

import numpy as np
import pytest

from pixelle_video.editing.evaluate import (
    Label,
    Span,
    evaluate_method,
    is_hit,
    load_labels,
    mcnemar_exact,
    paired_hit1,
    temporal_iou,
)
from pixelle_video.editing.index import SegmentIndex
from pixelle_video.editing.retrieval import (
    AssetLevelBaseline,
    BM25Channel,
    Prediction,
    SegmentRetriever,
    VisualChannel,
    rrf_fuse,
    tokenize,
)
from pixelle_video.testing.fake_encoders import FakeColourEncoder
from tests.editing.conftest import LABELS

# --------------------------------------------------------------------------- #
# Pure functions
# --------------------------------------------------------------------------- #


def test_tokenize_mixes_jieba_and_latin():
    toks = tokenize("我守在电话旁边 IMG_1234 Phone")
    assert "电话" in toks and "旁边" in toks
    assert "img_1234" in toks and "phone" in toks


def test_rrf_fuse_rewards_agreement():
    fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert [d for d, _ in fused][:2] in (["a", "b"], ["b", "a"])
    assert dict(fused)["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert dict(fused)["d"] == pytest.approx(1 / 63)


def test_rrf_fuse_weights():
    fused = rrf_fuse([["a"], ["b"]], weights=[1.0, 2.0])
    assert fused[0][0] == "b"
    with pytest.raises(ValueError):
        rrf_fuse([["a"]], weights=[1.0, 2.0])


def test_temporal_iou():
    assert temporal_iou(0, 3, 0, 3) == 1.0
    assert temporal_iou(0, 3, 3, 6) == 0.0
    assert temporal_iou(0, 18, 0, 3) == pytest.approx(3 / 18)


class TestHitRule:
    gold = Span("a.mp4", 10.0, 20.0)

    def _p(self, asset, s, e):
        return Prediction("x", asset, s, e, 1.0)

    def test_wrong_asset(self):
        assert not is_hit(self._p("b.mp4", 10, 20), self.gold)

    def test_whole_file_for_short_gold_is_miss(self):
        assert not is_hit(self._p("a.mp4", 0, 120), self.gold)

    def test_sub_shot_covering_half_is_hit(self):
        assert is_hit(self._p("a.mp4", 12, 17), self.gold)

    def test_tiny_sub_shot_is_miss(self):
        assert not is_hit(self._p("a.mp4", 12, 13), self.gold)

    def test_overlapping_shot_with_enough_iou(self):
        assert is_hit(self._p("a.mp4", 15, 25), self.gold)  # IoU 5/15

    def test_image_asset_match_is_hit(self):
        assert is_hit(self._p("p.png", 0, 0), Span("p.png", 0, 0))


def test_mcnemar_exact_known_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 5) == 1.0
    assert mcnemar_exact(10, 2) == pytest.approx(0.03857, abs=1e-4)
    assert mcnemar_exact(8, 0) == pytest.approx(2 / 256)


def test_load_labels_alts_and_duplicates(tmp_path):
    p = tmp_path / "l.jsonl"
    p.write_text(
        '# comment\n'
        '{"id": "1", "query": "q", "asset": "a.mp4", "start": 1, "end": 2, '
        '"alts": [{"asset": "b.mp4", "start": 0, "end": 3}]}\n'
    )
    (lb,) = load_labels(p)
    assert [g.asset for g in lb.gold] == ["a.mp4", "b.mp4"]
    p.write_text('{"id": "1", "query": "q", "asset": "a"}\n{"id": "1", "query": "q", "asset": "a"}\n')
    with pytest.raises(ValueError):
        load_labels(p)


# --------------------------------------------------------------------------- #
# Index + retrieval on synthetic footage
# --------------------------------------------------------------------------- #


def test_index_roundtrip(fake_index):
    index, out = fake_index
    loaded = SegmentIndex.load(out)
    assert [s.segment_id for s in loaded.segments] == [s.segment_id for s in index.segments]
    assert np.allclose(loaded.keyframe_emb, index.keyframe_emb)
    assert loaded.texts["filename"]["clips/rgb.mp4#000"] == "clips rgb"
    # 5 segments for rgb (magenta split), 3 for the 14 s white take, 1 image
    assert len(loaded.segments) == 9


def test_visual_channel_rejects_other_encoder(fake_index):
    index, _ = fake_index

    class Other(FakeColourEncoder):
        name = "other"

    with pytest.raises(ValueError):
        VisualChannel(index, Other())


def test_bm25_channel_over_custom_field(fake_index):
    index, _ = fake_index
    index.set_text("caption", {"clips/rgb.mp4#001": "窗台上的绿植", "clips/rgb.mp4#000": "桌上的电话"})
    ch = BM25Channel(index, "caption")
    assert ch.rank("窗台", 3)[0][0] == "clips/rgb.mp4#001"
    with pytest.raises(KeyError):
        index.set_text("caption", {"nope#000": "x"})


def _labels():
    return [Label.from_dict(d) for d in LABELS]


def test_segment_retrieval_beats_file_level_baseline(fake_index):
    index, _ = fake_index
    enc = FakeColourEncoder()
    seg = evaluate_method(SegmentRetriever(index, [VisualChannel(index, enc)], name="visual"), _labels())
    asset_vis = evaluate_method(AssetLevelBaseline(index, "visual", enc), _labels())
    asset_name = evaluate_method(AssetLevelBaseline(index, "filename"), _labels())

    assert seg.hit_at[1] == 1.0
    # Same visual evidence picks the right file every time ...
    assert asset_vis.asset_acc_at1 == 1.0
    # ... but whole-file spans miss the three 3 s shots inside the 18 s clip.
    assert asset_vis.hit_at[1] == pytest.approx(2 / 5)
    # Uninformative file names give the filename baseline almost nothing.
    assert asset_name.hit_at[1] < seg.hit_at[1]

    cmp = paired_hit1(seg, asset_vis)
    assert (cmp["a_only"], cmp["b_only"]) == (3, 0)


def test_fused_retriever_runs(fake_index):
    index, _ = fake_index
    enc = FakeColourEncoder()
    index.set_text("caption", {"clips/rgb.mp4#000": "红色 电话"})
    r = SegmentRetriever(index, [VisualChannel(index, enc), BM25Channel(index, "caption")])
    top = r.search("红色的电话画面", k=3)
    assert top[0].segment_id == "clips/rgb.mp4#000"
    assert r.name == "visual-max+bm25-caption"


def test_cli_eval_end_to_end(fake_index, tmp_path, monkeypatch):
    import sys

    from benchmarks import retrieval_eval

    _, out = fake_index
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in LABELS), encoding="utf-8")
    result = tmp_path / "res.json"
    monkeypatch.setattr(
        sys, "argv",
        ["retrieval_eval.py", "eval", "--index", str(out), "--labels", str(labels), "--fake",
         "--methods", "asset-filename", "asset-visual", "visual", "--out", str(result)],
    )
    retrieval_eval.main()
    data = json.loads(result.read_text())
    assert data["protocol"] == "retrieval-v1"
    assert data["reports"]["visual"]["hit_at"]["1"] == 1.0
    assert {c["a"] for c in data["comparisons"]} == {"asset-visual", "visual"}

    monkeypatch.setattr(sys, "argv", ["retrieval_eval.py", "sheet", "--index", str(out)])
    retrieval_eval.main()
    assert (out / "contact_sheet.html").read_text().count("<tr>") == 9


def test_cli_caption_describe_and_llm_baseline(fake_index, tmp_path, monkeypatch):
    import shutil
    import sys

    from benchmarks import retrieval_eval

    _, out = fake_index
    idx_dir = tmp_path / "idx"
    shutil.copytree(out, idx_dir)
    labels = tmp_path / "labels.jsonl"
    labels.write_text("\n".join(json.dumps(d, ensure_ascii=False) for d in LABELS), encoding="utf-8")
    for argv in (["caption", "--index", str(idx_dir), "--fake"],
                 ["describe-assets", "--index", str(idx_dir), "--fake"]):
        monkeypatch.setattr(sys, "argv", ["retrieval_eval.py", *argv])
        retrieval_eval.main()
    assert (idx_dir / "asset_descriptions.json").exists()
    result = tmp_path / "res.json"
    monkeypatch.setattr(sys, "argv", [
        "retrieval_eval.py", "eval", "--index", str(idx_dir), "--labels", str(labels), "--fake",
        "--methods", "llm-asset", "visual", "bm25-caption", "visual+bm25-caption",
        "--baseline", "llm-asset", "--out", str(result)])
    retrieval_eval.main()
    data = json.loads(result.read_text())
    assert set(data["reports"]) == {"llm-asset", "visual", "bm25-caption", "visual+bm25-caption"}
    assert data["reports"]["visual+bm25-caption"]["hit_at"]["1"] >= data["reports"]["llm-asset"]["hit_at"]["1"]


def test_chinese_clip_feature_path_without_torch():
    """Guards the version-independent feature computation (no weights or torch needed)."""
    from types import SimpleNamespace

    from pixelle_video.editing.encoders import ChineseClipEncoder

    enc = object.__new__(ChineseClipEncoder)
    enc.dim = 4
    hidden = np.arange(2 * 3 * 6, dtype=np.float32).reshape(2, 3, 6)   # batch, tokens, hidden
    pooled = np.ones((2, 6), dtype=np.float32)
    proj = np.ones((6, 4), dtype=np.float32)
    enc.model = SimpleNamespace(
        text_model=lambda **kw: (hidden,),
        vision_model=lambda **kw: (None, pooled),
        text_projection=lambda x: x @ proj,
        visual_projection=lambda x: x @ proj,
    )
    t = enc._text_features({"input_ids": None})
    assert t.shape == (2, 4) and np.allclose(t[0], hidden[0, 0].sum())  # uses the [CLS] token
    assert enc._image_features({"pixel_values": None}).shape == (2, 4)
    enc.dim = 5
    with pytest.raises(RuntimeError):
        enc._image_features({"pixel_values": None})


def test_label_sheet_parser():
    from benchmarks.labels_from_text import parse, parse_time

    assert parse_time("1:05") == 65 and parse_time("12.5") == 12.5
    durations = {"a.mp4": 60.0, "b.mp4": 10.0}
    text = """# comment
    我在窗边 | a.mp4 | 3 | 7
    雪 | a.mp4 | 0:10 | 0:14 | b.mp4 | 1 | 4
    坏行 | c.mp4 | 1 | 2
    越界 | b.mp4 | 8 | 12
    格式错
    """
    rows, errors = parse(text, durations)
    assert [r["query"] for r in rows] == ["我在窗边", "雪"]
    assert rows[1]["alts"] == [{"asset": "b.mp4", "start": 1.0, "end": 4.0}]
    assert len(errors) == 3
