"""
Measure the real error distribution of a media generation provider.

Calls the upstream APIProviderMediaService (keys come from config.yaml, same as
the WebUI) N times with real prompts, classifies every failure with
``pixelle_video.reliability.classify`` and writes:

* ``<out>.jsonl``  one row per call (latency, ok / error class, message)
* ``<out>.json``   summary incl. ``measured_mix`` for
                   ``reliability_ablation.py --mix-file``

Notes
* Upstream's own one-shot "neutralise prompt" retry is disabled
  (prompt_safety_retries=0) so moderation rejections are observed directly.
  Provider-client network retries stay on: the distribution is what the
  pipeline actually sees.
* Run with the concurrency you plan to deploy with; 429s depend on it.
* This spends real money. Start with --n 10.

    uv run python benchmarks/measure_provider_errors.py --list
    uv run python benchmarks/measure_provider_errors.py --workflow api/ark/<model> \
        --prompts benchmarks/prompts/real_prompts.txt --n 60 --concurrency 4 \
        --media-type video --image some_frame.png --out benchmarks/results/provider_errors_ark
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pixelle_video.reliability import ErrorClass, classify  # noqa: E402

# Map error classes to the simulator's fault kinds.
CLASS_TO_KIND = {
    ErrorClass.TRANSIENT: "timeout",  # split below using the message
    ErrorClass.CONTENT: "moderation",
    ErrorClass.OUTPUT: "corrupt",
    ErrorClass.INPUT: "bad_input",
}


def kind_of(row: dict) -> str:
    if row["ok"]:
        return "ok"
    if row["error_class"] == "transient":
        msg = row["error"].lower()
        return "rate_limit" if any(m in msg for m in ("429", "rate", "throttl", "too many")) else "timeout"
    return CLASS_TO_KIND[ErrorClass(row["error_class"])]


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    kinds = Counter(kind_of(r) for r in rows)
    failures = n - kinds.get("ok", 0)
    sim_kinds = ("timeout", "rate_limit", "moderation", "corrupt")
    sim_fail = sum(kinds.get(k, 0) for k in sim_kinds)
    lat = sorted(r["latency_s"] for r in rows if r["ok"])
    return {
        "calls": n,
        "failures": failures,
        "failure_rate": round(failures / n, 4) if n else 0.0,
        "by_kind": dict(kinds),
        "bad_input_rate": round(kinds.get("bad_input", 0) / n, 4) if n else 0.0,
        # Mix among the per-call fault kinds the simulator draws from.
        "measured_mix": {k: round(kinds.get(k, 0) / sim_fail, 4) for k in sim_kinds} if sim_fail else None,
        "latency_ok_p50_s": lat[len(lat) // 2] if lat else None,
        "latency_ok_p95_s": lat[int(len(lat) * 0.95) - 1] if lat else None,
    }


async def run(args) -> None:
    from pixelle_video.services.api_media import APIProviderMediaService

    svc = APIProviderMediaService({})
    if args.list:
        for w in svc.list_workflows():
            print(f"{w['key']:<55} {w['media_type']}")
        return

    prompts = [p.strip() for p in Path(args.prompts).read_text(encoding="utf-8").splitlines() if p.strip()]
    if not prompts:
        sys.exit("no prompts")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    sem = asyncio.Semaphore(args.concurrency)
    tmp = Path(tempfile.mkdtemp())

    async def one(i: int):
        prompt = prompts[i % len(prompts)]
        async with sem:
            t0 = time.perf_counter()
            row = {"i": i, "prompt": prompt, "workflow": args.workflow}
            try:
                await svc(
                    prompt=prompt, workflow=args.workflow, media_type=args.media_type,
                    width=args.width, height=args.height, duration=args.duration,
                    output_path=str(tmp / f"{i}.{'mp4' if args.media_type == 'video' else 'png'}"),
                    image_path=args.image, prompt_safety_retries=0,
                )
                row.update(ok=True, error_class=None, error=None)
            except Exception as exc:  # noqa: BLE001
                row.update(ok=False, error_class=classify(exc).value, error=f"{type(exc).__name__}: {exc}"[:500])
            row["latency_s"] = round(time.perf_counter() - t0, 2)
            rows.append(row)
            print(f"[{len(rows)}/{args.n}] {'ok' if row['ok'] else row['error_class']:<9} "
                  f"{row['latency_s']:>6.1f}s {'' if row['ok'] else row['error'][:80]}", flush=True)

    await asyncio.gather(*(one(i) for i in range(args.n)))
    rows.sort(key=lambda r: r["i"])
    with open(out.with_suffix(".jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"workflow": args.workflow, "concurrency": args.concurrency, **summarize(rows)}
    out.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list available api/... workflows and exit")
    ap.add_argument("--workflow")
    ap.add_argument("--prompts")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--media-type", choices=["image", "video"], default="video")
    ap.add_argument("--image", default=None, help="input image for image-to-video models")
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=1280)
    ap.add_argument("--duration", type=float, default=5)
    ap.add_argument("--out", default="benchmarks/results/provider_errors")
    args = ap.parse_args()
    if not args.list and not (args.workflow and args.prompts):
        ap.error("--workflow and --prompts are required (or use --list)")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
