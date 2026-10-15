from __future__ import annotations

import json
import sys

import pytest

from benchmarks import judge


def test_narration_seconds_clamped():
    assert judge.narration_seconds("短") == 2.0
    assert judge.narration_seconds("一" * 18) == 4.0
    assert judge.narration_seconds("一" * 100) == 8.0


def test_pool_page_and_score(fake_index, tmp_path, monkeypatch, capsys):
    _, idx_dir = fake_index
    q = tmp_path / "queries.txt"
    q.write_text("红色的电话画面\n白色的走廊\n# 注释\n一张黄色的照片\n", encoding="utf-8")
    out = tmp_path / "pool"
    monkeypatch.setattr(sys, "argv", ["j", "pool", "--index", str(idx_dir), "--queries", str(q), "--fake",
                                      "--methods", "visual", "asset-visual", "--out", str(out)])
    judge.main()
    pool = json.loads((out / "pool.json").read_text())
    assert len(pool["items"]) == 3
    page = (out / "judge.html").read_text()
    assert "能用" in page and "file://" in page and "visual" not in page.split("<script>")[0]  # blind

    with pytest.raises(SystemExit):  # frozen pool
        judge.main()

    red = pool["items"][0]
    # file-level clip for the red query is [0, narration length] of rgb.mp4
    asset_clip = next(c for c in red["clips"] if c["cid"] == red["ranks"]["asset-visual"][0])
    assert asset_clip["start"] == 0.0 and asset_clip["end"] == judge.narration_seconds("红色的电话画面")
    seg_top = red["ranks"]["visual"][0]
    usable = [seg_top] + [it["ranks"]["visual"][0] for it in pool["items"][1:]]
    jf = tmp_path / "judgments.json"
    jf.write_text(json.dumps({"pool_id": pool["pool_id"], "usable": usable}))
    res = tmp_path / "res.json"
    monkeypatch.setattr(sys, "argv", ["j", "score", "--pool", str(out / "pool.json"), "--judgments", str(jf),
                                      "--out", str(res)])
    judge.main()
    data = json.loads(res.read_text())
    assert data["reports"]["visual"]["hit@1"] == 1.0
    assert data["comparisons"][0]["a"] == "visual"

    jf.write_text(json.dumps({"pool_id": "other", "usable": []}))
    with pytest.raises(SystemExit):
        judge.main()
