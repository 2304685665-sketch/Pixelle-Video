"""
Reliability baseline benchmark (fault injection, no real API calls).

Measures the upstream StandardPipeline under injected shot failures with two
recovery strategies:

* ``single``      – run once; a failed video is a failed video.
* ``whole_retry`` – rerun the whole video up to N times (the naive fix).
  This is the hypothesis to rule out: it raises completion but re-pays for
  shots that had already succeeded.

Metrics per (strategy, failure_rate):
* completion_rate          – finished videos / attempted videos
* calls_per_finished_video – paid shot calls / finished videos
* wasted_calls_per_video   – successful calls beyond the first per shot,
                             averaged over attempted videos

Usage:
    uv run python benchmarks/reliability_baseline.py --videos 200 --shots 5 \
        --rates 0 0.05 0.1 0.3 --whole-retries 3 --out benchmarks/results/baseline.json

Note: the fault mix in FaultProfile is a placeholder until it is replaced by the
distribution measured from real provider runs; report numbers accordingly.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import tempfile
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pixelle_video.services.persistence import PersistenceService  # noqa: E402
from pixelle_video.testing import (  # noqa: E402
    FakeCore,
    FakeFrameProcessor,
    FaultProfile,
    InjectedFault,
)


def _stub_concat() -> None:
    from pixelle_video.services import video as video_module

    def fake_concat(self, videos, output, **kwargs):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"fake-final-mp4")
        return output

    video_module.VideoService.concat_videos = fake_concat


async def run_one_video(pipeline, processor, script, kwargs, strategy, whole_retries):
    """Return (finished: bool, calls_used: int, wasted: int) for one video."""
    start_calls = processor.total_calls
    start_ledger = len(processor.ledger)
    attempts = 1 if strategy == "single" else 1 + whole_retries
    finished = False
    for _ in range(attempts):
        try:
            await pipeline(script, **kwargs)
            finished = True
            break
        except InjectedFault:
            continue

    # Count wasted calls for this video only: a "shot" is identified by its index,
    # and every run of the same video re-generates the same shots.
    per_shot_ok: dict[int, int] = {}
    for call in processor.ledger[start_ledger:]:
        if call.outcome == "ok":
            per_shot_ok[call.frame_index] = per_shot_ok.get(call.frame_index, 0) + 1
    wasted = sum(n - 1 for n in per_shot_ok.values() if n > 1)
    return finished, processor.total_calls - start_calls, wasted


async def bench(videos: int, shots: int, rates: list[float], whole_retries: int, seed: int):
    from pixelle_video.pipelines.standard import StandardPipeline

    script = "\n\n".join(f"Shot {i} narration." for i in range(shots))
    kwargs = dict(
        mode="fixed",
        split_mode="paragraph",
        title="bench",
        frame_template="1080x1920/static_default.html",
        media_width=1080,
        media_height=1920,
    )

    rows = []
    for strategy in ("single", "whole_retry"):
        for rate in rates:
            with tempfile.TemporaryDirectory() as tmp:
                os.environ["PIXELLE_VIDEO_ROOT"] = tmp
                os.chdir(tmp)
                processor = FakeFrameProcessor(profile=FaultProfile(failure_rate=rate, seed=seed))
                persistence = PersistenceService(output_dir=str(Path(tmp) / "output"))
                pipeline = StandardPipeline(FakeCore(processor, persistence))

                finished_n, calls, wasted = 0, [], []
                for _ in range(videos):
                    ok, used, w = await run_one_video(
                        pipeline, processor, script, kwargs, strategy, whole_retries
                    )
                    finished_n += ok
                    calls.append(used)
                    wasted.append(w)

            rows.append(
                {
                    "strategy": strategy,
                    "failure_rate": rate,
                    "videos": videos,
                    "shots_per_video": shots,
                    "completion_rate": round(finished_n / videos, 4),
                    "calls_per_finished_video": round(sum(calls) / finished_n, 3) if finished_n else None,
                    "wasted_calls_per_video": round(statistics.mean(wasted), 3),
                }
            )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=int, default=200)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--rates", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.3])
    ap.add_argument("--whole-retries", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="CRITICAL")
    _stub_concat()

    out_path = Path(args.out).resolve() if args.out else None
    rows = asyncio.run(bench(args.videos, args.shots, args.rates, args.whole_retries, args.seed))

    header = f"{'strategy':<12} {'fail':>5} {'complete':>9} {'calls/finished':>15} {'wasted/video':>13}"
    print(header)
    print("-" * len(header))
    for r in rows:
        cpf = "-" if r["calls_per_finished_video"] is None else f"{r['calls_per_finished_video']:.2f}"
        print(
            f"{r['strategy']:<12} {r['failure_rate']:>5.2f} {r['completion_rate']:>9.1%} "
            f"{cpf:>15} {r['wasted_calls_per_video']:>13.2f}"
        )

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"args": vars(args), "rows": rows}, indent=2))
        print(f"\nsaved: {out_path}")


if __name__ == "__main__":
    main()
