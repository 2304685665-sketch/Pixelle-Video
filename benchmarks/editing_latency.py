"""
Two benchmarks for the reliable pipeline (simulated provider, no real API calls).

1. Editing session: each video (10 shots) goes through 5 successive script
   edits drawn from {modify 1-2 lines, insert, delete, reorder, change TTS speed}.
   Incremental = ReliablePipeline.revise (content-addressed reuse);
   baseline    = upstream behaviour, every edit regenerates every shot.

2. Parallelism x isolation: provider latency is log-normal; per-call faults at
   --fail. Time to a finished video (initial run + up to 2 resumes) is measured
   for concurrency {1, 4} with shot isolation on/off (fail-fast). Times are in
   units of the median single-call latency.

    uv run python benchmarks/editing_latency.py --out benchmarks/results/editing_latency.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pixelle_video.reliability import ReliablePipeline, ShotsIncompleteError  # noqa: E402
from pixelle_video.services.persistence import PersistenceService  # noqa: E402
from pixelle_video.testing import FakeCore, FakeFrameProcessor, FaultProfile  # noqa: E402

KW = dict(mode="fixed", split_mode="paragraph", title="b",
          frame_template="1080x1920/static_default.html", media_width=1080, media_height=1920)


async def _no_sleep(_):
    return None


def _pipeline(proc, tmp: Path, name: str, **kw):
    core = FakeCore(proc, PersistenceService(output_dir=str(tmp / "output")))
    return ReliablePipeline(core, sleep=_no_sleep, store_path=tmp / f"{name}.sqlite", **kw)


# --------------------------------------------------------------------------- #
# 1. Editing session
# --------------------------------------------------------------------------- #

OPS = ("modify1", "modify2", "insert", "delete", "reorder", "speed")


def apply_edit(lines, op, rng, v, step):
    lines = list(lines)
    if op in ("modify1", "modify2"):
        for i in rng.sample(range(len(lines)), 1 if op == "modify1" else 2):
            lines[i] = f"Video {v} edit {step} line {i} rewritten."
    elif op == "insert":
        lines.insert(rng.randrange(len(lines) + 1), f"Video {v} edit {step} inserted.")
    elif op == "delete" and len(lines) > 3:
        lines.pop(rng.randrange(len(lines)))
    elif op == "reorder":
        i, j = rng.sample(range(len(lines)), 2)
        lines[i], lines[j] = lines[j], lines[i]
    return lines


async def editing_session(videos: int, shots: int, edits: int, tmp: Path) -> dict:
    rng = random.Random(11)
    per_op = {op: {"n": 0, "incremental": 0, "full": 0} for op in OPS}
    totals = {"incremental": 0, "full": 0, "initial": 0}
    for v in range(videos):
        lines = [f"Video {v} line {i}." for i in range(shots)]
        proc = FakeFrameProcessor()
        pipe = _pipeline(proc, tmp, f"edit{v}")
        await pipe("\n\n".join(lines), **KW)
        totals["initial"] += proc.total_calls
        (task_id,) = pipe.store.tasks()
        for step in range(edits):
            op = rng.choice(OPS)
            before = proc.total_calls
            if op == "speed":
                plan = pipe.store.load_plan(task_id)
                plan["params"]["tts_speed"] = round(1.0 + 0.1 * rng.randint(1, 5), 1)
                pipe.store.replace_plan(task_id, plan, len(plan["narrations"]))
                await pipe.resume(task_id)
            else:
                lines = apply_edit(lines, op, rng, v, step)
                await pipe.revise(task_id, lines)
            inc = proc.total_calls - before
            n_now = len(pipe.store.load_plan(task_id)["narrations"])
            per_op[op]["n"] += 1
            per_op[op]["incremental"] += inc
            per_op[op]["full"] += n_now
            totals["incremental"] += inc
            totals["full"] += n_now
    return {
        "videos": videos, "shots": shots, "edits_per_video": edits,
        "edit_calls_incremental": totals["incremental"], "edit_calls_full_rerun": totals["full"],
        "saving": round(1 - totals["incremental"] / totals["full"], 4),
        "per_op_mean_calls": {
            op: {"incremental": round(d["incremental"] / d["n"], 2), "full_rerun": round(d["full"] / d["n"], 2),
                 "n": d["n"]}
            for op, d in per_op.items() if d["n"]
        },
    }


# --------------------------------------------------------------------------- #
# 2. Parallelism x isolation
# --------------------------------------------------------------------------- #


async def time_to_video(v, shots, fail, concurrency, isolate, median, tmp, runs=3):
    rng = random.Random(1000 + v)
    proc = FakeFrameProcessor(
        profile=FaultProfile(failure_rate=fail, seed=v, bad_input_rate=0.0),
        latency=lambda: median * rng.lognormvariate(0, 0.5),
    )
    pipe = _pipeline(proc, tmp, f"lat{v}-{concurrency}-{isolate}", concurrency=concurrency, isolate=isolate)
    script = "\n\n".join(f"Video {v} shot {i}." for i in range(shots))
    t0 = time.perf_counter()
    task_id = None
    for _ in range(runs):
        try:
            if task_id is None:
                await pipe(script, **KW)
            else:
                await pipe.resume(task_id)
            return (time.perf_counter() - t0) / median, True
        except ShotsIncompleteError as e:
            task_id = e.task_id
    return (time.perf_counter() - t0) / median, False


async def latency_grid(videos, shots, median, fails, tmp) -> list:
    rows = []
    for fail in fails:
        for concurrency in (1, 4):
            for isolate in (True, False):
                times, done = [], 0
                for v in range(videos):
                    t, ok = await time_to_video(v, shots, fail, concurrency, isolate, median, tmp)
                    done += ok
                    if ok:
                        times.append(t)
                times.sort()
                rows.append({
                    "fail": fail, "concurrency": concurrency, "isolate": isolate,
                    "completion": round(done / videos, 3),
                    "p50_units": round(statistics.median(times), 2) if times else None,
                    "p95_units": round(times[int(0.95 * len(times)) - 1], 2) if times else None,
                })
                r = rows[-1]
                print(f"fail={fail} c={concurrency} isolate={isolate!s:<5} complete={r['completion']:.1%} "
                      f"p50={r['p50_units']} p95={r['p95_units']} (x median call latency)", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=int, default=200)
    ap.add_argument("--shots", type=int, default=10)
    ap.add_argument("--edits", type=int, default=5)
    ap.add_argument("--lat-videos", type=int, default=100)
    ap.add_argument("--lat-shots", type=int, default=8)
    ap.add_argument("--median-latency", type=float, default=0.01, help="seconds (scaled-down provider latency)")
    ap.add_argument("--fail", type=float, nargs="+", default=[0.1, 0.3])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out).resolve() if args.out else None
    logger.remove()
    logger.add(sys.stderr, level="CRITICAL")
    from pixelle_video.services import video as video_module

    def fake_concat(self, videos, output, **kw):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"x")
        return output

    video_module.VideoService.concat_videos = fake_concat
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["PIXELLE_VIDEO_ROOT"] = tmp
        os.chdir(tmp)
        tmp = Path(tmp)
        editing = asyncio.run(editing_session(args.videos, args.shots, args.edits, tmp))
        print("editing:", json.dumps(editing, ensure_ascii=False), flush=True)
        latency = asyncio.run(latency_grid(args.lat_videos, args.lat_shots, args.median_latency, args.fail, tmp))
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"args": {k: v for k, v in vars(args).items() if k != "out"},
                                   "editing_session": editing, "latency": latency}, indent=1))
        print(f"saved {out}")


if __name__ == "__main__":
    main()
