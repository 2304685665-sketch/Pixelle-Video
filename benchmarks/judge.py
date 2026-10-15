"""
Blind pooled relevance judgment for narration -> footage retrieval.

Instead of hand-labelling timestamps, the editor writes narration lines only.
Each method returns its top-k clips; clips are pooled, de-duplicated, shuffled
and shown without method names on a local HTML page with playback. The editor
ticks every clip that could be used for the line. Scores are then recovered per
method (Hit@1 = top-1 clip judged usable).

How file-level methods become clips: upstream AssetBasedPipeline trims an
assigned video from 0 s to the narration duration (VideoService
._trim_video_to_duration), so a whole-file prediction is judged as
[0, narration_duration]. Narration duration is estimated from text length
(4.5 Chinese characters per second, clamped to 2-8 s).

    # 1. pool candidates and build the judging page
    uv run python benchmarks/judge.py pool --index runs/footage_idx \
        --queries benchmarks/labels/pilot_queries.txt \
        --model-dir models/chinese-clip-vit-base-patch16 --out runs/judge_pilot_v1
    open runs/judge_pilot_v1/judge.html        # tick usable clips, click "导出结果"

    # 2. score (judgments.json is saved by the page to ~/Downloads)
    uv run python benchmarks/judge.py score --pool runs/judge_pilot_v1/pool.json \
        --judgments ~/Downloads/judgments.json --baseline asset-visual \
        --out benchmarks/results/judge_pilot_v1.json

Limitation: pooled judgment measures whether returned clips are usable
(precision at the top); it cannot see usable clips that no method returned.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pixelle_video.editing.evaluate import mcnemar_exact  # noqa: E402
from pixelle_video.editing.index import SegmentIndex  # noqa: E402

CHARS_PER_SECOND = 4.5


def narration_seconds(text: str) -> float:
    return round(min(8.0, max(2.0, len(text.strip()) / CHARS_PER_SECOND)), 1)


def read_queries(path: str | Path) -> list[str]:
    lines = [ln.split("#", 1)[0].strip() for ln in Path(path).read_text(encoding="utf-8").splitlines()]
    queries = [q for q in lines if q]
    if len(set(queries)) != len(queries):
        sys.exit("duplicate lines in queries file")
    return queries


def candidates_for(method, query: str, k: int) -> list[dict]:
    out = []
    for p in method.search(query, k=k):
        start, end = p.start, p.end
        if p.segment_id.endswith("#asset"):  # file-level: used from 0 s for the narration length
            start, end = 0.0, min(p.end, narration_seconds(query)) if p.end > 0 else 0.0
        out.append({"asset": p.asset_path, "start": round(start, 2), "end": round(end, 2)})
    return out


def build_pool(index: SegmentIndex, methods: dict, queries: list[str], k: int, seed: int) -> dict:
    rng = random.Random(seed)
    items = []
    for qi, query in enumerate(queries):
        clips: dict[tuple, dict] = {}
        ranks: dict[str, list[str]] = {}
        for name, method in methods.items():
            ranks[name] = []
            for c in candidates_for(method, query, k):
                key = (c["asset"], c["start"], c["end"])
                if key not in clips:
                    clips[key] = {**c, "cid": f"q{qi:02d}c{len(clips):02d}"}
                ranks[name].append(clips[key]["cid"])
        shown = list(clips.values())
        rng.shuffle(shown)
        items.append({"qid": f"q{qi:02d}", "query": query, "clips": shown, "ranks": ranks})
    return {"index_root": index.root, "encoder": index.encoder_name, "k": k, "seed": seed,
            "methods": list(methods), "items": items}


PAGE = """<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clip Judging</title>
<style>
:root{{--bg:#fafaf8;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#e4e3de;--accent:#2f6f5e}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151514;--fg:#ecebe6;--muted:#9b9a94;--card:#1f1f1d;--line:#33332f;--accent:#7cc4ad}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,system-ui,sans-serif;margin:0;padding:16px;max-width:1100px;margin:auto}}
header{{position:sticky;top:0;background:var(--bg);padding:8px 0 12px;border-bottom:1px solid var(--line);z-index:2}}
.q{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:16px 0}}
.q h2{{font-size:17px;margin:0 0 10px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:12px}}
.clip{{border:1px solid var(--line);border-radius:8px;padding:8px}}
.clip.on{{border-color:var(--accent);box-shadow:0 0 0 2px var(--accent)}}
video{{width:100%;border-radius:6px;background:#000;aspect-ratio:4/3}}
label{{display:flex;gap:8px;align-items:center;margin-top:6px;cursor:pointer}}
.meta{{color:var(--muted);font-size:12px}}
button{{font:inherit;padding:6px 14px;border-radius:6px;border:1px solid var(--accent);background:var(--accent);color:var(--bg);cursor:pointer}}
</style>
<header><b>镜头盲评</b> · 共 {n} 句 · 每段都看一下，只要<b>能直接拿来配这句话</b>就勾选（可多选，也可一个都不选）。
<span id="prog" class="meta"></span> <button onclick="exportJudgments()">导出结果</button></header>
{body}
<script>
const KEY = "judge:{pool_id}";
let state = {{}};
try {{ state = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{}}
function save() {{ try {{ localStorage.setItem(KEY, JSON.stringify(state)); }} catch (e) {{}} progress(); }}
function progress() {{
  const seen = document.querySelectorAll('.q[data-done="1"]').length;
  document.getElementById("prog").textContent = `已看完 ${{seen}} / {n} 句`;
}}
document.querySelectorAll("input[type=checkbox]").forEach(cb => {{
  cb.checked = !!state[cb.dataset.cid];
  cb.closest(".clip").classList.toggle("on", cb.checked);
  cb.addEventListener("change", () => {{
    state[cb.dataset.cid] = cb.checked; cb.closest(".clip").classList.toggle("on", cb.checked); save();
  }});
}});
document.querySelectorAll(".q").forEach(q => {{
  const mark = () => {{ q.dataset.done = "1"; progress(); }};
  q.querySelectorAll("input,video").forEach(el => el.addEventListener("click", mark));
  if (q.querySelector("input:checked")) q.dataset.done = "1";
}});
document.querySelectorAll("video").forEach(v => {{
  const s = +v.dataset.start, e = +v.dataset.end;
  v.addEventListener("loadedmetadata", () => {{ v.currentTime = s; }});
  v.addEventListener("play", () => {{ if (v.currentTime < s || v.currentTime >= e) v.currentTime = s; }});
  v.addEventListener("timeupdate", () => {{ if (e > s && v.currentTime >= e) {{ v.pause(); v.currentTime = s; }} }});
}});
function exportJudgments() {{
  const out = {{pool_id: "{pool_id}", usable: Object.keys(state).filter(k => state[k])}};
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {{type: "application/json"}}));
  a.download = "judgments.json"; a.click();
}}
progress();
</script></html>"""


def render_page(pool: dict, pool_id: str) -> str:
    root = Path(pool["index_root"])
    blocks = []
    for it in pool["items"]:
        clips = []
        for c in it["clips"]:
            src = (root / c["asset"]).as_uri()
            span = f"{c['start']:.1f}–{c['end']:.1f}s" if c["end"] > c["start"] else "图片"
            media = (f'<video src="{html.escape(src)}" data-start="{c["start"]}" data-end="{c["end"]}" '
                     f'controls preload="metadata" playsinline muted></video>')
            clips.append(f'<div class="clip">{media}<div class="meta">{html.escape(c["asset"])} · {span}</div>'
                         f'<label><input type="checkbox" data-cid="{c["cid"]}"> 能用</label></div>')
        blocks.append(f'<section class="q"><h2>{html.escape(it["query"])}</h2>'
                      f'<div class="grid">{"".join(clips) or "<i>没有候选</i>"}</div></section>')
    return PAGE.format(n=len(pool["items"]), body="\n".join(blocks), pool_id=pool_id)


def score(pool: dict, usable: set[str], ks=(1, 3)) -> dict:
    reports = {}
    for name in pool["methods"]:
        per_query = []
        for it in pool["items"]:
            ranked = it["ranks"].get(name, [])
            first = next((r for r, cid in enumerate(ranked, start=1) if cid in usable), None)
            per_query.append({"qid": it["qid"], "first_usable_rank": first,
                              "top1": next((c for c in it["clips"] if ranked and c["cid"] == ranked[0]), None)})
        n = len(per_query)
        reports[name] = {
            "n": n,
            **{f"hit@{k}": round(sum(1 for q in per_query if q["first_usable_rank"] and
                                     q["first_usable_rank"] <= k) / n, 4) for k in ks},
            "per_query": per_query,
        }
    return reports


def cmd_pool(args) -> None:
    from benchmarks.retrieval_eval import build_method, make_encoder

    idx = SegmentIndex.load(args.index)
    queries = read_queries(args.queries)
    q_sha = hashlib.sha256(Path(args.queries).read_bytes()).hexdigest()
    encoder = make_encoder(args)
    methods = {m: build_method(m, idx, encoder, args) for m in args.methods}
    pool = build_pool(idx, methods, queries, args.k, args.seed)
    pool["queries_file"], pool["queries_sha256"] = str(args.queries), q_sha
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "pool.json").exists():
        sys.exit(f"{out}/pool.json exists: this pool is frozen. Use a new --out for a new round.")
    pool_id = hashlib.sha256(json.dumps(pool, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:12]
    pool["pool_id"] = pool_id
    (out / "pool.json").write_text(json.dumps(pool, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "judge.html").write_text(render_page(pool, pool_id), encoding="utf-8")
    n_clips = sum(len(it["clips"]) for it in pool["items"])
    print(f"{len(queries)} queries, {n_clips} clips to judge (avg {n_clips / len(queries):.1f}/query)")
    print(f"queries SHA-256: {q_sha}")
    print(f"open {out / 'judge.html'}")


def cmd_score(args) -> None:
    pool = json.loads(Path(args.pool).read_text(encoding="utf-8"))
    judg = json.loads(Path(args.judgments).read_text(encoding="utf-8"))
    if judg.get("pool_id") != pool["pool_id"]:
        sys.exit("judgments were made on a different pool")
    usable = set(judg["usable"])
    reports = score(pool, usable)
    print(f"blind pooled judgment · {len(pool['items'])} queries · usable clips ticked: {len(usable)}")
    print(f"{'method':<18} {'Hit@1':>6} {'Hit@3':>6}")
    for name, r in reports.items():
        print(f"{name:<18} {r['hit@1']:>6.3f} {r['hit@3']:>6.3f}")
    comparisons = []
    if args.baseline:
        base = reports[args.baseline]["per_query"]
        for name, r in reports.items():
            if name == args.baseline:
                continue
            a_only = sum(1 for x, y in zip(r["per_query"], base) if x["first_usable_rank"] == 1 and y["first_usable_rank"] != 1)
            b_only = sum(1 for x, y in zip(r["per_query"], base) if y["first_usable_rank"] == 1 and x["first_usable_rank"] != 1)
            comparisons.append({"a": name, "b": args.baseline, "a_only": a_only, "b_only": b_only,
                                "p": mcnemar_exact(a_only, b_only)})
            print(f"McNemar {name} vs {args.baseline}: only-{name}={a_only} only-baseline={b_only} "
                  f"p={mcnemar_exact(a_only, b_only):.4f}")
    if args.show_failures:
        queries = {it["qid"]: it["query"] for it in pool["items"]}
        misses = [q for q in reports[args.show_failures]["per_query"] if q["first_usable_rank"] != 1]
        print(f"\nfailures of {args.show_failures}: {len(misses)}/{len(pool['items'])}")
        for q in misses:
            t = q["top1"] or {}
            where = f"usable at rank {q['first_usable_rank']}" if q["first_usable_rank"] else "none usable in top-k"
            print(f"- [{q['qid']}] {queries[q['qid']]}\n    top1: {t.get('asset')} "
                  f"{t.get('start', 0):.1f}-{t.get('end', 0):.1f}s ({where})")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "protocol": "blind-pooled-judgment-v1", "pool_id": pool["pool_id"], "queries_sha256": pool["queries_sha256"],
            "k": pool["k"], "encoder": pool["encoder"], "usable_ticked": len(usable),
            "reports": reports, "comparisons": comparisons,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"saved {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pool")
    p.add_argument("--index", required=True)
    p.add_argument("--queries", required=True)
    p.add_argument("--methods", nargs="+", default=["visual", "asset-visual", "asset-filename"])
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--seed", type=int, default=20261016)
    p.add_argument("--out", required=True)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--fake", action="store_true")
    p.add_argument("--llm-model", default=None)
    p.set_defaults(func=cmd_pool)
    p = sub.add_parser("score")
    p.add_argument("--pool", required=True)
    p.add_argument("--judgments", required=True)
    p.add_argument("--baseline", default="asset-visual")
    p.add_argument("--show-failures", default="visual")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_score)
    args = ap.parse_args()
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    args.func(args)


if __name__ == "__main__":
    main()
