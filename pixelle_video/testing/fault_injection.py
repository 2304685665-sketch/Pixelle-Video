"""
Fault injection fakes for orchestration tests and reliability benchmarks.

Design notes
------------
* ``FakeFrameProcessor`` stands in for ``PixelleVideoCore.frame_processor``.
  One call to it represents one *paid* shot generation (TTS + media + compose),
  which is the unit we care about when measuring wasted spend.
* Faults are decided by a seeded RNG (``FaultProfile``) and/or a deterministic
  schedule (``fail_on``), so every test and benchmark run is reproducible.
* Every call is appended to ``ledger`` with the frame index, attempt number and
  outcome. Benchmarks derive "API calls per finished video" and "duplicate
  calls after resume" from this ledger.
* The error classes mirror the failure kinds observed across the real provider
  clients (timeouts, HTTP 429, content moderation, unusable output). They are
  intentionally *not* subclasses of each other: a later error taxonomy must map
  each kind to its own handling policy.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type

# --------------------------------------------------------------------------- #
# Failure kinds
# --------------------------------------------------------------------------- #


class InjectedFault(Exception):
    """Base class for all injected faults (lets tests catch 'any fault')."""


class ProviderTimeoutError(InjectedFault):
    """Provider did not respond in time. Transient; retry is reasonable."""


class RateLimitedError(InjectedFault):
    """Provider returned HTTP 429. Transient; retry after backoff."""


class ModerationRejectedError(InjectedFault):
    """Provider content moderation rejected the prompt. Retrying the same prompt is wasteful."""


class CorruptOutputError(InjectedFault):
    """Provider returned output that cannot be used (empty / undecodable file)."""


class BadInputError(InjectedFault):
    """Invalid shot parameters. Deterministic: every retry of the same shot fails."""


FAULT_KINDS: Dict[str, Type[InjectedFault]] = {
    "timeout": ProviderTimeoutError,
    "rate_limit": RateLimitedError,
    "moderation": ModerationRejectedError,
    "corrupt": CorruptOutputError,
    "bad_input": BadInputError,
}


# --------------------------------------------------------------------------- #
# Fault profile
# --------------------------------------------------------------------------- #


@dataclass
class FaultProfile:
    """
    Probability of each fault kind per shot call.

    ``failure_rate`` is split across kinds according to ``mix``. The default mix
    is a placeholder; replace it with the distribution measured from real
    provider runs before reporting benchmark numbers.
    """

    failure_rate: float = 0.0
    mix: Dict[str, float] = field(
        default_factory=lambda: {
            "timeout": 0.4,
            "rate_limit": 0.3,
            "moderation": 0.15,
            "corrupt": 0.15,
        }
    )
    seed: int = 0
    # Content-dependent faults (state is keyed by shot content, so it survives reruns):
    rewrite_fix_rate: float = 0.7  # P(a rewritten prompt passes moderation)
    bad_input_rate: float = 0.0  # P(a shot has invalid parameters; fails forever)

    def __post_init__(self) -> None:
        if not 0.0 <= self.failure_rate <= 1.0:
            raise ValueError("failure_rate must be within [0, 1]")
        unknown = set(self.mix) - (set(FAULT_KINDS) - {"bad_input"})
        if unknown:
            raise ValueError(f"unknown fault kinds: {sorted(unknown)}")
        total = sum(self.mix.values())
        if total <= 0:
            raise ValueError("mix weights must sum to a positive number")
        self._rng = random.Random(self.seed)

    def draw(self) -> Optional[str]:
        """Return a fault kind for this call, or ``None`` for success."""
        if self._rng.random() >= self.failure_rate:
            return None
        kinds = list(self.mix)
        weights = [self.mix[k] for k in kinds]
        return self._rng.choices(kinds, weights=weights, k=1)[0]


# --------------------------------------------------------------------------- #
# Fake frame processor
# --------------------------------------------------------------------------- #


@dataclass
class ShotCall:
    """One paid shot-generation call."""

    frame_index: int
    attempt: int
    outcome: str  # "ok" or a fault kind
    task_id: Optional[str] = None
    futile: bool = False  # failure fully determined by shot content (retrying unchanged cannot help)


class FakeFrameProcessor:
    """
    Drop-in replacement for ``core.frame_processor``.

    Args:
        profile: random fault profile (default: never fails).
        fail_on: deterministic faults, ``{frame_index: [kind_for_attempt_1, ...]}``.
            ``None`` in the list means that attempt succeeds. Attempts beyond the
            list length fall back to ``profile``.
        segment_duration: duration written onto each successful frame.
        on_call: optional hook invoked before each call (e.g. to simulate a crash).
    """

    def __init__(
        self,
        profile: Optional[FaultProfile] = None,
        fail_on: Optional[Dict[int, List[Optional[str]]]] = None,
        segment_duration: float = 3.0,
        on_call: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        self.profile = profile or FaultProfile(failure_rate=0.0)
        self.fail_on = {k: list(v) for k, v in (fail_on or {}).items()}
        self.segment_duration = segment_duration
        self.on_call = on_call
        self.ledger: List[ShotCall] = []
        self._attempts: Dict[tuple, int] = {}
        self._bad_input: Dict[str, bool] = {}
        self._moderated: Dict[tuple, bool] = {}  # (content, prompt_version) -> rejected?

    # Signature mirrors pixelle_video.services.frame_processor.FrameProcessor.__call__
    async def __call__(
        self,
        frame,
        storyboard,
        config,
        total_frames: int = 1,
        progress_callback: Optional[Callable] = None,
        **_: Any,
    ):
        task_id = getattr(config, "task_id", None)
        key = (task_id, frame.index)
        attempt = self._attempts.get(key, 0) + 1
        self._attempts[key] = attempt

        if self.on_call is not None:
            self.on_call(frame.index, attempt)

        kind, futile = self._decide(frame, attempt)
        self.ledger.append(
            ShotCall(frame.index, attempt, kind or "ok", task_id=task_id, futile=futile)
        )
        if kind is not None:
            raise FAULT_KINDS[kind](f"injected {kind} on frame {frame.index} (attempt {attempt})")

        # Simulate a produced segment on disk, like the real processor does.
        segment = self._segment_path(task_id, frame.index)
        segment.parent.mkdir(parents=True, exist_ok=True)
        segment.write_bytes(b"fake-mp4")
        frame.video_segment_path = str(segment)
        frame.duration = self.segment_duration
        return frame

    # ------------------------------------------------------------------ #

    def _decide(self, frame, attempt: int) -> tuple[Optional[str], bool]:
        """Return (fault kind or None, futile)."""
        scheduled = self.fail_on.get(frame.index)
        if scheduled is not None and attempt <= len(scheduled):
            return scheduled[attempt - 1], False

        rng = self.profile._rng
        content = getattr(frame, "narration", None) or str(frame.index)
        if content not in self._bad_input:
            self._bad_input[content] = rng.random() < self.profile.bad_input_rate
        if self._bad_input[content]:
            return "bad_input", True

        version = getattr(frame, "prompt_rewrites", 0)
        key = (content, version)
        if self._moderated.get(key):
            return "moderation", True  # same prompt version again: rejected again
        if key not in self._moderated and any(c == content for c, _ in self._moderated):
            rejected = rng.random() >= self.profile.rewrite_fix_rate  # newly rewritten prompt
            self._moderated[key] = rejected
            if rejected:
                return "moderation", False

        kind = self.profile.draw()
        if kind == "moderation":
            self._moderated[key] = True
        return kind, False

    @staticmethod
    def _segment_path(task_id: Optional[str], frame_index: int) -> Path:
        from pixelle_video.utils.os_util import get_output_path

        return Path(get_output_path(task_id or "no_task", "frames", f"{frame_index:02d}_segment.mp4"))

    # ------------------------------------------------------------------ #
    # Ledger helpers used by tests and benchmarks
    # ------------------------------------------------------------------ #

    @property
    def total_calls(self) -> int:
        return len(self.ledger)

    def calls_for(self, frame_index: int) -> int:
        return sum(1 for c in self.ledger if c.frame_index == frame_index)

    def successful_frames(self) -> List[int]:
        return sorted({c.frame_index for c in self.ledger if c.outcome == "ok"})

    def futile_calls(self) -> int:
        return sum(1 for c in self.ledger if c.futile)

    def duplicate_successes(self) -> int:
        """Number of successful calls beyond the first for each frame (wasted spend)."""
        seen: Dict[int, int] = {}
        for c in self.ledger:
            if c.outcome == "ok":
                seen[c.frame_index] = seen.get(c.frame_index, 0) + 1
        return sum(n - 1 for n in seen.values() if n > 1)


# --------------------------------------------------------------------------- #
# Fake core
# --------------------------------------------------------------------------- #


class FakeCore:
    """
    Minimal stand-in for ``PixelleVideoCore``.

    Only the attributes touched by ``StandardPipeline`` are provided. The LLM,
    TTS and media services are ``None`` because the tests run the pipeline in
    ``fixed`` mode with a static template and a user-supplied title, which never
    reaches them; any accidental call fails loudly.
    """

    def __init__(self, frame_processor: FakeFrameProcessor, persistence) -> None:
        self.frame_processor = frame_processor
        self.persistence = persistence
        self.config: Dict[str, Any] = {}
        self.llm = None
        self.tts = None
        self.media = None
        self.video = None
