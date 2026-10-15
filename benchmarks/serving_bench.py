"""
Serving-layer benchmarks on a real Redis server.

1. Idempotency storm: concurrent clients submit with a small key set.
   Baseline = the common non-atomic "GET key, if missing create task and SET key".
2. Worker chaos: workers are killed mid-job at random; leases expire, the reaper
   requeues, other workers take over. With checkpoint resume vs without.
3. Distributed rate limit: many workers share one provider token bucket; peak
   calls in any 1 s window vs the configured rate.

    uv run python benchmarks/serving_bench.py --out benchmarks/results/serving.json
(starts a throwaway redis-server unless REDIS_URL is set)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import redis.asyncio as aioredis  # noqa: E402

from pixelle_video.services.persistence import PersistenceService  # noqa: E402
from pixelle_video.serving import (  # noqa: E402
    EventLog,
    JobQueue,
    TokenBucket,
    Worker,
    pixelle_runner,
)
from pixelle_video.testing import FakeCore, FakeFrameProcessor  # noqa: E402

KW = dict(mode="fixed", split_mode="paragraph", title="b",
          frame_template="1080x1920/static_default.html", media_width=1080, media_height=1920)


def start_redis():
    if os.environ.get("REDIS_URL"):
        return os.environ["REDIS_URL"], None
    if not shutil.which("redis-server"):
        sys.exit("redis-server not found; set REDIS_URL")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    d = tempfile.mkdtemp()
    proc = subprocess.Popen(["redis-server", "--port", str(port), "--save", "", "--dir", d],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.5)
    return f"redis://127.0.0.1:{port}/0", proc


async def _client(url):
    return aioredis.from_url(url, decode_responses=True)


# --------------------------------------------------------------------------- #
# 1. Idempotency storm
# --------------------------------------------------------------------------- #


async def naive_submit(r, request, key):
    """The usual non-atomic pattern (no Lua, no SET NX)."""
    existing = await r.get(f"naive:idem:{key}")
    if existing:
        return existing, True
    task_id = uuid.uuid4().hex[:16]
    await r.hset(f"naive:task:{task_id}", mapping={"request": json.dumps(request)})
    await asyncio.sleep(0)  # yield, as any real I/O between steps would
    await r.set(f"naive:idem:{key}", task_id)
    await r.rpush("naive:queue", task_id)
    return task_id, False


async def idempotency_storm(url, submits, keys, clients):
    conns = [await _client(url) for _ in range(clients)]
    await conns[0].flushdb()
    rng = random.Random(1)
    plan = [f"k{rng.randrange(keys)}" for _ in range(submits)]
    req = {"text": "x", "params": {}}

    async def drive(fn):
        sem = asyncio.Semaphore(clients)

        async def one(i, key):
            async with sem:
                return await fn(conns[i % clients], key)

        t0 = time.perf_counter()
        await asyncio.gather(*(one(i, k) for i, k in enumerate(plan)))
        return time.perf_counter() - t0

    queues = [JobQueue(c) for c in conns]
    t_safe = await drive(lambda c, k: queues[conns.index(c)].submit(req, k))
    t_naive = await drive(lambda c, k: naive_submit(c, req, k))
    r = conns[0]
    out = {
        "submits": submits, "unique_keys": len(set(plan)), "clients": clients,
        "atomic_tasks_created": await r.llen("pv:queue"),
        "naive_tasks_created": await r.llen("naive:queue"),
        "atomic_submits_per_s": round(submits / t_safe),
        "naive_submits_per_s": round(submits / t_naive),
    }
    for c in conns:
        await c.aclose()
    return out


# --------------------------------------------------------------------------- #
# 2. Worker chaos
# --------------------------------------------------------------------------- #


class Kill(BaseException):
    pass


async def worker_chaos(url, tasks, workers, kill_prob, checkpoint: bool, seed=7):
    r = await _client(url)
    await r.flushdb()
    tmp = Path(tempfile.mkdtemp())
    os.environ["PIXELLE_VIDEO_ROOT"] = str(tmp)
    os.chdir(tmp)
    q, log = JobQueue(r, lease_ttl=1), EventLog(r)
    for i in range(tasks):
        script = "\n\n".join(f"Task {i} shot {j}." for j in range(5))
        await q.submit({"text": script, "params": KW}, f"t{i}")

    rng = random.Random(seed)
    proc = FakeFrameProcessor()
    armed = {"on": True}

    def maybe_kill(idx, attempt):
        if armed["on"] and rng.random() < kill_prob:
            raise Kill()

    proc.on_call = maybe_kill

    async def no_sleep(_):
        return None

    def core_factory():
        return FakeCore(proc, PersistenceService(output_dir=str(tmp / "output")))

    store = tmp / "cp.sqlite"
    if checkpoint:
        runner = pixelle_runner(core_factory, str(store), sleep=no_sleep)
    else:
        # Same pipeline, but every takeover starts the video from scratch.
        base = pixelle_runner(core_factory, str(store), sleep=no_sleep)

        async def runner(task_id, request, emit):
            return await base(f"{task_id}-{uuid.uuid4().hex[:6]}", request, emit)

    kills = 0

    async def worker_loop(n):
        nonlocal kills
        while True:
            w = Worker(q, log, runner, worker_id=f"w{n}-{uuid.uuid4().hex[:4]}")
            try:
                tid = await w.run_one(timeout=0.2)
            except Kill:
                kills += 1
                continue  # process died; a fresh worker process takes its slot
            if tid is None:
                depth = await q.depth()
                if depth["queued"] == 0 and depth["processing"] == 0:
                    return
                await q.reap()
                await asyncio.sleep(0.2)

    t0 = time.perf_counter()
    await asyncio.gather(*(worker_loop(n) for n in range(workers)))
    elapsed = time.perf_counter() - t0
    armed["on"] = False

    statuses = {}
    for i in range(tasks):
        tid = (await r.get(f"pv:idem:t{i}")).split("|")[0]
        st = (await q.get(tid))["status"]
        statuses[st] = statuses.get(st, 0) + 1
    # duplicates: successful generations beyond one per (task content, shot)
    seen, dup = set(), 0
    for c in proc.ledger:
        if c.outcome != "ok":
            continue
        key = (c.task_id.split("-")[0], c.frame_index)
        dup += key in seen
        seen.add(key)
    await r.aclose()
    return {
        "checkpoint_resume": checkpoint, "tasks": tasks, "workers": workers, "kill_prob_per_call": kill_prob,
        "worker_kills": kills, "statuses": statuses, "paid_shot_calls": proc.total_calls,
        "duplicate_successful_shots": dup, "seconds": round(elapsed, 1),
    }


# --------------------------------------------------------------------------- #
# 3. Distributed rate limit
# --------------------------------------------------------------------------- #


async def rate_limit(url, workers, calls_each, rate, capacity, limited: bool):
    conns = [await _client(url) for _ in range(workers)]
    await conns[0].delete("pv:bucket:bench")
    stamps = []

    async def worker(c):
        b = TokenBucket(c, "bench", rate=rate, capacity=capacity)
        for _ in range(calls_each):
            if limited:
                await b.acquire()
            stamps.append(time.perf_counter())
            await asyncio.sleep(0.005)  # the provider call itself

    t0 = time.perf_counter()
    await asyncio.gather(*(worker(c) for c in conns))
    elapsed = time.perf_counter() - t0
    stamps.sort()
    peak, j = 0, 0
    for i, t in enumerate(stamps):
        while stamps[j] < t - 1.0:
            j += 1
        peak = max(peak, i - j + 1)
    for c in conns:
        await c.aclose()
    return {"limited": limited, "workers": workers, "calls": len(stamps), "rate": rate, "capacity": capacity,
            "peak_calls_in_any_1s": peak, "mean_rate": round(len(stamps) / elapsed, 1)}


async def main_async(args, url):
    res = {}
    res["idempotency"] = await idempotency_storm(url, args.submits, args.keys, args.clients)
    print("idempotency:", res["idempotency"], flush=True)
    res["chaos"] = []
    for cp in (True, False):
        out = await worker_chaos(url, args.tasks, args.workers, args.kill_prob, cp)
        res["chaos"].append(out)
        print("chaos:", out, flush=True)
    res["rate_limit"] = [await rate_limit(url, args.rl_workers, args.rl_calls, args.rate, args.capacity, lim)
                         for lim in (False, True)]
    for x in res["rate_limit"]:
        print("rate_limit:", x, flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submits", type=int, default=5000)
    ap.add_argument("--keys", type=int, default=500)
    ap.add_argument("--clients", type=int, default=50)
    ap.add_argument("--tasks", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--kill-prob", type=float, default=0.03)
    ap.add_argument("--rl-workers", type=int, default=16)
    ap.add_argument("--rl-calls", type=int, default=15)
    ap.add_argument("--rate", type=float, default=20)
    ap.add_argument("--capacity", type=float, default=5)
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
    url, proc = start_redis()
    try:
        res = asyncio.run(main_async(args, url))
    finally:
        if proc:
            proc.terminate()
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"args": {k: v for k, v in vars(args).items() if k != "out"}, **res}, indent=1))
        print(f"saved {out}")


if __name__ == "__main__":
    main()
