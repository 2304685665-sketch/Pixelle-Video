from __future__ import annotations

import asyncio
import json

import pytest

from pixelle_video.serving import (
    EventLog,
    IdempotencyConflict,
    JobQueue,
    TokenBucket,
    Worker,
    pixelle_runner,
)
from pixelle_video.services.persistence import PersistenceService
from pixelle_video.testing import FakeCore, FakeFrameProcessor

from tests.conftest import PIPELINE_KWARGS, SCRIPT_5_SHOTS

REQ = {"text": SCRIPT_5_SHOTS, "params": {k: v for k, v in PIPELINE_KWARGS.items()}}


# --------------------------------------------------------------------------- #
# Idempotent submission
# --------------------------------------------------------------------------- #


async def test_same_key_same_body_is_deduplicated(redis):
    q = JobQueue(redis)
    a = await q.submit(REQ, "k1")
    b = await q.submit(REQ, "k1")
    assert (a.deduplicated, b.deduplicated) == (False, True)
    assert a.task_id == b.task_id
    assert await redis.llen("pv:queue") == 1


async def test_same_key_different_body_conflicts(redis):
    q = JobQueue(redis)
    await q.submit(REQ, "k1")
    with pytest.raises(IdempotencyConflict):
        await q.submit({**REQ, "text": "other"}, "k1")


async def test_concurrent_duplicate_submits_create_one_task_per_key(redis):
    q = JobQueue(redis)
    keys = [f"key-{i % 20}" for i in range(400)]
    subs = await asyncio.gather(*(q.submit(REQ, k) for k in keys))
    assert len({s.task_id for s in subs}) == 20
    assert sum(not s.deduplicated for s in subs) == 20
    assert await redis.llen("pv:queue") == 20


# --------------------------------------------------------------------------- #
# Leases and reaping
# --------------------------------------------------------------------------- #


async def test_expired_lease_is_requeued_live_lease_is_not(redis):
    q = JobQueue(redis, lease_ttl=1)
    t1 = (await q.submit(REQ, "a")).task_id
    t2 = (await q.submit({**REQ, "text": "x"}, "b")).task_id
    assert await q.claim("w1") == t1
    assert await q.claim("w2") == t2
    await asyncio.sleep(0.6)
    await q.heartbeat(t2, "w2")  # w2 alive, w1 silent
    await asyncio.sleep(0.6)
    assert await q.reap() == [t1]
    assert await redis.lrange("pv:queue", 0, -1) == [t1]
    assert (await q.get(t1))["requeues"] == "1"
    assert await q.reap() == []


async def test_heartbeat_detects_takeover(redis):
    q = JobQueue(redis)
    t = (await q.submit(REQ, "a")).task_id
    await q.claim("w1")
    await redis.set(f"pv:lease:{t}", "w2")
    assert await q.heartbeat(t, "w1") is False


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #


async def test_token_bucket_burst_then_wait(redis):
    b = TokenBucket(redis, "seedance", rate=10, capacity=2)
    assert await b.try_acquire() == 0.0
    assert await b.try_acquire() == 0.0
    wait = await b.try_acquire()
    assert 0.0 < wait <= 0.11


async def test_token_bucket_shared_across_clients(redis, redis_url):
    import redis.asyncio as aioredis

    other = aioredis.from_url(redis_url, decode_responses=True)
    try:
        a = TokenBucket(redis, "kling", rate=1, capacity=3)
        b = TokenBucket(other, "kling", rate=1, capacity=3)
        got = [await x.try_acquire() == 0.0 for x in (a, b, a, b)]
        assert got == [True, True, True, False]
    finally:
        await other.aclose()


# --------------------------------------------------------------------------- #
# Events
# --------------------------------------------------------------------------- #


async def test_events_resume_after_last_event_id(redis):
    log = EventLog(redis)
    ids = [await log.emit("t", {"type": "shot", "shot": i}) for i in range(4)]
    await log.emit("t", {"type": "task", "status": "completed"})
    got = [e async for _, e in log.read("t", last_id=ids[1], block_ms=100)]
    assert [e.get("shot") for e in got] == [2, 3, None]


# --------------------------------------------------------------------------- #
# Worker end to end (fake shot generator)
# --------------------------------------------------------------------------- #


class Crash(BaseException):
    pass


def _runner(workdir, processor, store):
    def core_factory():
        return FakeCore(processor, PersistenceService(output_dir=str(workdir / "output")))

    async def no_sleep(_):
        return None

    return pixelle_runner(core_factory, store_path=str(store), sleep=no_sleep)


async def _events(redis, task_id):
    return [e async for _, e in EventLog(redis).read(task_id, block_ms=100)]


async def test_worker_completes_and_streams_ordered_events(redis, workdir, stub_concat):
    q, log = JobQueue(redis), EventLog(redis)
    tid = (await q.submit(REQ, "k")).task_id
    proc = FakeFrameProcessor()
    w = Worker(q, log, _runner(workdir, proc, workdir / "cp.sqlite"))
    assert await w.run_one() == tid

    task = await q.get(tid)
    assert task["status"] == "completed"
    events = await _events(redis, tid)
    assert [e["status"] for e in events] == ["running"] + ["done"] * 5 + ["completed"]
    assert [e["shot"] for e in events if e["type"] == "shot"] == [0, 1, 2, 3, 4]


async def test_worker_crash_is_recovered_without_repaying_shots(redis, workdir, stub_concat):
    q, log = JobQueue(redis, lease_ttl=1), EventLog(redis)
    tid = (await q.submit(REQ, "k")).task_id
    calls = {"n": 0}

    def crash_on_third(idx, attempt):
        calls["n"] += 1
        if calls["n"] == 3:
            raise Crash()

    proc = FakeFrameProcessor(on_call=crash_on_third)
    store = workdir / "cp.sqlite"
    w1 = Worker(q, log, _runner(workdir, proc, store), worker_id="w1")
    with pytest.raises(Crash):
        await w1.run_one()
    assert (await q.get(tid))["status"] == "running"  # worker died holding the job

    await asyncio.sleep(1.2)  # lease expires
    assert await q.reap() == [tid]
    proc.on_call = None
    w2 = Worker(q, log, _runner(workdir, proc, store), worker_id="w2")
    assert await w2.run_one() == tid

    task = await q.get(tid)
    assert (task["status"], task["requeues"], task["worker"]) == ("completed", "1", "w2")
    assert proc.duplicate_successes() == 0
    statuses = [e.get("status") for e in await _events(redis, tid)]
    assert statuses.count("reused") == 2  # shots 0 and 1 reused after takeover


async def test_partial_task_can_be_resumed_via_queue(redis, workdir, stub_concat):
    q, log = JobQueue(redis), EventLog(redis)
    tid = (await q.submit(REQ, "k")).task_id
    proc = FakeFrameProcessor(fail_on={3: ["bad_input"]})
    w = Worker(q, log, _runner(workdir, proc, workdir / "cp.sqlite"))
    await w.run_one()
    task = await q.get(tid)
    assert task["status"] == "partial" and json.loads(task["failed"]) == [3]

    proc.fail_on.clear()
    assert await q.requeue_partial(tid)
    await w.run_one()
    assert (await q.get(tid))["status"] == "completed"
    assert proc.duplicate_successes() == 0


# --------------------------------------------------------------------------- #
# HTTP API
# --------------------------------------------------------------------------- #


async def test_http_api_contract(redis, workdir, stub_concat):
    import httpx

    from pixelle_video.serving.api import create_app

    app = create_app(redis)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        assert (await c.post("/v2/videos", json=REQ)).status_code == 400
        r1 = await c.post("/v2/videos", json=REQ, headers={"Idempotency-Key": "abc"})
        r2 = await c.post("/v2/videos", json=REQ, headers={"Idempotency-Key": "abc"})
        assert r1.status_code == r2.status_code == 202
        assert r1.json()["task_id"] == r2.json()["task_id"] and r2.json()["deduplicated"]
        r3 = await c.post("/v2/videos", json={**REQ, "text": "x"}, headers={"Idempotency-Key": "abc"})
        assert r3.status_code == 409
        bad = await c.post("/v2/videos", json={"text": "x", "params": {"task_id": "evil"}},
                           headers={"Idempotency-Key": "z"})
        assert bad.status_code == 400
        tid = r1.json()["task_id"]
        assert (await c.get("/v2/videos/nope")).status_code == 404
        assert (await c.get(f"/v2/videos/{tid}")).json()["status"] == "queued"

        await Worker(JobQueue(redis), EventLog(redis),
                     _runner(workdir, FakeFrameProcessor(), workdir / "cp.sqlite")).run_one()

        body = (await c.get(f"/v2/videos/{tid}/events")).text
        ids = [line[4:] for line in body.splitlines() if line.startswith("id: ")]
        assert len(ids) == 7
        rest = (await c.get(f"/v2/videos/{tid}/events", headers={"Last-Event-ID": ids[2]})).text
        assert [line[4:] for line in rest.splitlines() if line.startswith("id: ")] == ids[3:]
        assert (await c.get(f"/v2/videos/{tid}")).json()["result"]["duration"] == 15.0
        assert (await c.post(f"/v2/videos/{tid}/resume")).status_code == 409
