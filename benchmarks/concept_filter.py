"""
Pre-filter footage by a visual concept (e.g. "food") with Chinese-CLIP, before editing.

Each indexed segment is scored against a set of positive prompts and a set of
negative prompts (max over keyframes, then max over prompts). A segment is kept
when its best positive score beats its best negative score (margin > --margin).
Kept segments that touch inside the same file are merged into spans.

Outputs in --out:
  spans.json    kept spans (file, start, end, margin) + every segment's scores
  review.html   all segments sorted by margin, kept ones marked, for a quick check
  clips/*.mp4   each kept span cut from the source footage (with --cut), usable as
                the media folder of an editing run

    uv run python benchmarks/concept_filter.py --index runs/footage_idx \
        --model-dir models/chinese-clip-vit-base-patch16 --preset food --out runs/food --cut

Zero-shot CLIP makes mistakes; review.html is there so a person can check the list.
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pixelle_video.editing.index import SegmentIndex  # noqa: E402
from pixelle_video.editing.retrieval import VisualChannel  # noqa: E402

PRESETS = {
    "food": {
        "positive": ["一盘食物的特写", "桌上的饭菜", "正在吃东西", "甜品和冰淇淋", "寿司", "一碗拉面",
                     "烤肉和烤串", "饮料和汽水", "便利店的便当和饭团"],
        "negative": ["城市街道", "商店里的货架和商品", "酒店房间", "游乐园和人群", "车站和火车",
                     "人物对着镜头说话", "美容院", "棒球场", "天空"],
    },
}


def score_segments(index: SegmentIndex, channel, positive: list[str], negative: list[str]):
    pos = np.max(np.stack([channel.segment_scores(q) for q in positive]), axis=0)
    neg = np.max(np.stack([channel.segment_scores(q) for q in negative]), axis=0)
    return pos, neg


def merge_spans(segments, keep: list[bool], margins, gap: float = 0.5) -> list[dict]:
    spans: list[dict] = []
    for seg, k, m in sorted(zip(segments, keep, margins), key=lambda t: (t[0].asset_path, t[0].start)):
        if not k:
            continue
        last = spans[-1] if spans else None
        if last and last["asset"] == seg.asset_path and seg.start - last["end"] <= gap:
            last["end"] = max(last["end"], seg.end)
            last["segments"].append(seg.segment_id)
            last["margin"] = max(last["margin"], float(m))
        else:
            spans.append({"asset": seg.asset_path, "start": seg.start, "end": seg.end,
                          "segments": [seg.segment_id], "margin": float(m)})
    return spans


def cut(src: Path, start: float, end: float, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{end - start:.3f}",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", str(dst)], check=True)


def render_review(index_dir: Path, rows: list[dict]) -> str:
    cards = []
    for r in rows:
        img = (index_dir / r["keyframe"]).resolve().as_uri() if r["keyframe"] else ""
        cls = "keep" if r["keep"] else "drop"
        cards.append(f'<div class="c {cls}"><img src="{html.escape(img)}" loading="lazy"><div>'
                     f'{"✅" if r["keep"] else "—"} {html.escape(r["asset"])} {r["start"]:.1f}-{r["end"]:.1f}s '
                     f'· {r["margin"]:+.3f}</div></div>')
    return ("<!doctype html><meta charset=utf-8><title>Concept filter review</title><style>"
            "body{font:13px system-ui;margin:16px;background:#fafaf8;color:#1d1d1b}"
            ".g{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px}"
            ".c{border:2px solid #ddd;border-radius:6px;padding:4px}.keep{border-color:#2f6f5e}"
            ".drop{opacity:.55}img{width:100%;border-radius:4px}"
            "@media(prefers-color-scheme:dark){body{background:#151514;color:#eee}.c{border-color:#333}}"
            f"</style><h3>{sum(r['keep'] for r in rows)} / {len(rows)} segments kept (sorted by margin)</h3>"
            f"<div class=g>{''.join(cards)}</div>")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="food")
    ap.add_argument("--margin", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cut", action="store_true", help="cut kept spans into --out/clips")
    ap.add_argument("--source-root", default=None, help="footage to cut from (default: the index root)")
    args = ap.parse_args()

    from pixelle_video.editing.encoders import ChineseClipEncoder

    idx = SegmentIndex.load(args.index)
    channel = VisualChannel(idx, ChineseClipEncoder(args.model_dir))
    preset = PRESETS[args.preset]
    pos, neg = score_segments(idx, channel, preset["positive"], preset["negative"])
    margin = pos - neg
    keep = [bool(m > args.margin) for m in margin]
    spans = merge_spans(idx.segments, keep, margin)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [{"segment_id": s.segment_id, "asset": s.asset_path, "start": s.start, "end": s.end,
             "pos": float(p), "neg": float(n), "margin": float(m), "keep": k,
             "keyframe": s.keyframes[len(s.keyframes) // 2] if s.keyframes else ""}
            for s, p, n, m, k in zip(idx.segments, pos, neg, margin, keep)]
    rows.sort(key=lambda r: -r["margin"])
    (out / "spans.json").write_text(json.dumps({"preset": args.preset, "prompts": preset, "margin": args.margin,
                                                "spans": spans, "segments": rows}, ensure_ascii=False, indent=1))
    (out / "review.html").write_text(render_review(Path(args.index), rows), encoding="utf-8")
    total = sum(s["end"] - s["start"] for s in spans)
    print(f"kept {sum(keep)}/{len(keep)} segments -> {len(spans)} spans, {total:.0f}s "
          f"from {len({s['asset'] for s in spans})} files")
    if args.cut:
        root = Path(args.source_root or idx.root).expanduser()
        for i, s in enumerate(spans):
            stem = Path(s["asset"]).stem
            cut(root / s["asset"], s["start"], s["end"], out / "clips" / f"{stem}_{s['start']:06.1f}.mp4")
        print(f"cut {len(spans)} clips -> {out / 'clips'}")
    print(f"review: open {out / 'review.html'}")


if __name__ == "__main__":
    main()
