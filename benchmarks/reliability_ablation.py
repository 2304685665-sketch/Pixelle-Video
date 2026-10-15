"""
Reliability ablation under fault injection (no real API calls).

Every strategy gets the same budget: one initial run plus up to ``--runs - 1``
recovery runs per video. Upstream recovers by rerunning the whole video (a new
task); checkpointed strategies recover with ``resume(task_id)``.

Each video uses its own seeded fault stream (seed = video index), so outcomes
are paired across strategies and compared with an exact McNemar test.

Fault model (simulator, not measured provider data):
* per-call failure with probability ``--fail``; kinds mixed per FaultProfile.mix
* moderation is sticky per prompt version; a rewritten prompt passes with
  probability ``--rewrite-fix``
* a fraction ``--bad-input`` of shots have invalid parameters and never succeed

Usage:
    uv run python benchmarks/reliability_ablation.py --videos 500 --fail 0.1 0.3 \
        --out benchmarks/results/reliability_ablation.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pixelle_video.editing.evaluate import mcnemar_exact  # noqa: E402
from pixelle_video.reliability import (  # noqa: E402
    ReliablePipeline,
    RetryPolicy,
    ShotsIncompleteError,
)
from pixelle_video.services.persistence import PersistenceService  # noqa: E402
from pixelle_video.testing import (  # noqa: E402
    FakeCore,
    FakeFrameProcessor,
    FaultProfile,
    InjectedFault,
)


async def _no_sleep(_):
    return None


@dataclass
class Strategy:
    name: str
    build: Callable  # core -> pipeline
    resumable: bool


def strategies() -> List[Strategy]:
    from pixelle_video.pipelines.standard import StandardPipeline

    def reliable(**kw):
        return lambda core: ReliablePipeline(core, sleep=_no_sleep, **kw)

    full = dict(isolate=True, policy=RetryPolicy(), checkpoint=True)
    return [
        Strategy("S0 upstream (single run)", StandardPipeline, False),
        Strategy("H1 upstream + whole-video rerun", StandardPipeline, False),
        Strategy("H2 + uniform shot retry x3", reliable(isolate=False, policy=RetryPolicy.uniform(3),
                                                         rewrite=None, checkpoint=False), False),
        Strategy("FULL", reliable(**full), True),
        Strategy("FULL - error classes (uniform x3)", reliable(**{**full, "policy": RetryPolicy.uniform(3)}), True),
        Strategy("FULL - prompt rewrite", reliable(**full, rewrite=None), True),
        Strategy("FULL - checkpoint/resume", reliable(**{**full, "checkpoint": False}), False),
        Strategy("FULL - shot isolation", reliable(**{**full, "isolate": False}), True),
    ]


@dataclass
class VideoOutcome:
    finished: bool
    calls: int
    dup: int
    futile: int
    runs: int


@dataclass
class Row:
    strategy: str
    fail: float
    outcomes: List[VideoOutcome] = field(default_factory=list)

    def summary(self) -> Dict[str, float]:
        n = len(self.outcomes)
        fin = [o for o in self.outcomes if o.finished]
        calls = sum(o.calls for o in self.outcomes)
        return {
            "videos": n,
            "completion_rate": round(len(fin) / n, 4),
            "calls_per_finished_video": round(calls / len(fin), 3) if fin else None,
            "duplicate_success_calls_per_video": round(sum(o.dup for o in self.outcomes) / n, 3),
            "futile_calls_per_video": round(sum(o.futile for o in self.outcomes) / n, 3),
            "mean_runs_per_finished_video": round(sum(o.runs for o in fin) / len(fin), 3) if fin else None,
        }


def _script(v: int, shots: int) -> str:
    return "\n\n".join(f"Video {v} shot {i} narration." for i in range(shots))


async def run_video(strategy: Strategy, v: int, shots: int, fail: float, args, tmp: Path) -> VideoOutcome:
    profile = FaultProfile(
        failure_rate=fail, seed=v, rewrite_fix_rate=args.rewrite_fix, bad_input_rate=args.bad_input,
        **({"mix": args.mix} if args.mix else {}),
    )
    processor = FakeFrameProcessor(profile=profile)
    core = FakeCore(processor, PersistenceService(output_dir=str(tmp / "output")))
    pipeline = strategy.build(core)
    kwargs = dict(mode="fixed", split_mode="paragraph", title=f"v{v}",
                  frame_template="1080x1920/static_default.html", media_width=1080, media_height=1920)
    budget = 1 if strategy.name.startswith("S0") else args.runs

    finished, runs, task_id = False, 0, None
    for _ in range(budget):
        runs += 1
        try:
            if strategy.resumable and task_id is not None:
                await pipeline.resume(task_id)
            else:
                await pipeline(_script(v, shots), **kwargs)
            finished = True
            break
        except ShotsIncompleteError as e:
            task_id = e.task_id
        except InjectedFault:
            pass

    # Duplicate successes: successful calls beyond one per shot (same content across reruns).
    ok_per_shot: Dict[int, int] = {}
    for c in processor.ledger:
        if c.outcome == "ok":
            ok_per_shot[c.frame_index] = ok_per_shot.get(c.frame_index, 0) + 1
    dup = sum(n - 1 for n in ok_per_shot.values() if n > 1)
    return VideoOutcome(finished, processor.total_calls, dup, processor.futile_calls(), runs)


async def bench(args) -> List[Row]:
    rows: List[Row] = []
    for fail in args.fail:
        for strat in strategies():
            row = Row(strat.name, fail)
            with tempfile.TemporaryDirectory() as tmp:
                os.environ["PIXELLE_VIDEO_ROOT"] = tmp
                os.chdir(tmp)
                for v in range(args.videos):
                    row.outcomes.append(await run_video(strat, v, args.shots, fail, args, Path(tmp)))
            rows.append(row)
            s = row.summary()
            print(
                f"fail={fail:.2f} {strat.name:<36} complete={s['completion_rate']:.1%} "
                f"calls/fin={s['calls_per_finished_video']} dup={s['duplicate_success_calls_per_video']} "
                f"futile={s['futile_calls_per_video']} runs/fin={s['mean_runs_per_finished_video']}",
                flush=True,
            )
    return rows


class _Crash(BaseException):
    pass


async def crash_trials(n: int, shots: int, tmp: Path) -> dict:
    """
    Kill the process at a random paid call (uniform over the video's calls), then
    recover: FULL resumes the checkpoint, upstream reruns the whole video.
    Returns duplicate successful calls per trial for both.
    """
    from pixelle_video.pipelines.standard import StandardPipeline

    kwargs = dict(mode="fixed", split_mode="paragraph", title="c",
                  frame_template="1080x1920/static_default.html", media_width=1080, media_height=1920)
    rng = random.Random(123)
    out = {"trials": n, "full_dup": 0, "upstream_dup": 0, "full_recovered": 0, "upstream_recovered": 0}
    for t in range(n):
        crash_at = rng.randint(1, shots)
        for which in ("full", "upstream"):
            calls = {"n": 0}

            def hook(idx, attempt, calls=calls, crash_at=crash_at):
                calls["n"] += 1
                if calls["n"] == crash_at:
                    raise _Crash()

            proc = FakeFrameProcessor(on_call=hook)
            core = FakeCore(proc, PersistenceService(output_dir=str(tmp / "output")))
            pipe = (ReliablePipeline(core, sleep=_no_sleep, store_path=tmp / f"c{t}.sqlite")
                    if which == "full" else StandardPipeline(core))
            script = _script(10_000 + t, shots)
            try:
                await pipe(script, **kwargs)
            except _Crash:
                pass
            proc.on_call = None
            if which == "full":
                (task_id,) = pipe.store.tasks()
                await pipe.resume(task_id)
            else:
                await pipe(script, **kwargs)
            out[f"{which}_dup"] += proc.duplicate_successes()
            out[f"{which}_recovered"] += 1
    return out


def paired(rows: List[Row], a: str, b: str, fail: float) -> dict:
    ra = next(r for r in rows if r.strategy == a and r.fail == fail)
    rb = next(r for r in rows if r.strategy == b and r.fail == fail)
    a_only = sum(1 for x, y in zip(ra.outcomes, rb.outcomes) if x.finished and not y.finished)
    b_only = sum(1 for x, y in zip(ra.outcomes, rb.outcomes) if y.finished and not x.finished)
    return {"a": a, "b": b, "fail": fail, "a_only": a_only, "b_only": b_only, "p": mcnemar_exact(a_only, b_only)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=int, default=500)
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--fail", type=float, nargs="+", default=[0.1, 0.3])
    ap.add_argument("--runs", type=int, default=3, help="initial run + recovery runs")
    ap.add_argument("--rewrite-fix", type=float, default=0.7)
    ap.add_argument("--bad-input", type=float, default=0.02)
    ap.add_argument("--crash-trials", type=int, default=200)
    ap.add_argument("--mix-file", default=None,
                    help="summary JSON from measure_provider_errors.py; uses its measured_mix and bad_input_rate")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = Path(args.out).resolve() if args.out else None
    args.mix = None
    if args.mix_file:
        measured = json.loads(Path(args.mix_file).read_text())
        if not measured.get("measured_mix"):
            sys.exit("mix file has no failures; cannot derive a fault mix")
        args.mix = {k: v for k, v in measured["measured_mix"].items() if v > 0}
        args.bad_input = measured.get("bad_input_rate", args.bad_input)
        print(f"using measured mix {args.mix}, bad_input_rate={args.bad_input}")

    logger.remove()
    logger.add(sys.stderr, level="CRITICAL")
    from pixelle_video.services import video as video_module

    def fake_concat(self, videos, output, **kw):
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"x")
        return output

    video_module.VideoService.concat_videos = fake_concat
    random.seed(0)

    rows = asyncio.run(bench(args))
    comparisons = []
    for fail in args.fail:
        for other in ("H1 upstream + whole-video rerun", "H2 + uniform shot retry x3"):
            comparisons.append(paired(rows, "FULL", other, fail))
    for c in comparisons:
        print(f"McNemar completion fail={c['fail']}: FULL vs {c['b']}: "
              f"FULL-only={c['a_only']} other-only={c['b_only']} p={c['p']:.2e}")

    with tempfile.TemporaryDirectory() as tmp:
        os.environ["PIXELLE_VIDEO_ROOT"] = tmp
        os.chdir(tmp)
        crash = asyncio.run(crash_trials(args.crash_trials, args.shots, Path(tmp)))
    print(f"crash trials={crash['trials']}: duplicate successful calls after recovery "
          f"FULL={crash['full_dup']} upstream={crash['upstream_dup']}")

    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "crash_trials": crash,
            "args": {k: v for k, v in vars(args).items() if k != "out"},
            "fault_mix": args.mix or FaultProfile().mix,
            "fault_mix_source": args.mix_file or "placeholder (not measured)",
            "rows": [{"strategy": r.strategy, "fail": r.fail, **r.summary()} for r in rows],
            "comparisons": comparisons,
        }, indent=1))
        print(f"saved {out}")


if __name__ == "__main__":
    main()
