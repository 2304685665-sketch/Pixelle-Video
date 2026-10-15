"""
Error taxonomy, per-class retry policy and circuit breaker for shot generation.

Upstream retry logic is scattered across provider clients with different
behaviour (DashScope network retry, Kling urllib3 Retry, Edge-TTS backoff, one
"neutralised prompt" retry in api_media). At pipeline level every failure ends
the whole video the same way. Here every exception is mapped to one class and
each class has its own handling:

=============  ==========================================  ======================
class          examples                                    policy
=============  ==========================================  ======================
TRANSIENT      timeout, HTTP 429/5xx, connection reset     exponential backoff
CONTENT        provider moderation / safety rejection      rewrite prompt, retry
OUTPUT         empty / undecodable media                   regenerate
INPUT          bad parameters, missing file, ValueError    fail fast, no retry
=============  ==========================================  ======================

Retrying an INPUT error or re-sending a moderated prompt unchanged only burns
paid calls, so the classification is what lets retries help instead of hurt.
"""

from __future__ import annotations

import enum
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional


class ErrorClass(str, enum.Enum):
    TRANSIENT = "transient"
    CONTENT = "content"
    OUTPUT = "output"
    INPUT = "input"


_TRANSIENT_MARKERS = (
    "timeout", "timed out", "429", "rate limit", "rate_limit", "ratelimit", "ratelimited",
    "throttl", "too many requests",
    "502", "503", "504", "connection reset", "connection aborted", "temporarily unavailable",
    "remote end closed", "eof occurred",
)
_CONTENT_MARKERS = (
    "moderation", "datainspectionfailed", "inappropriate", "sensitive", "safety",
    "content policy", "content_policy", "审核", "违规",
)
_OUTPUT_MARKERS = ("corrupt", "empty output", "invalid data found", "moov atom not found", "no frames")


def classify(exc: BaseException) -> ErrorClass:
    """Map an exception to an ErrorClass by type first, then type name and message markers."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return ErrorClass.TRANSIENT
    msg = f"{type(exc).__name__} {exc}".lower()
    if any(m in msg for m in _CONTENT_MARKERS):
        return ErrorClass.CONTENT
    if any(m in msg for m in _TRANSIENT_MARKERS):
        return ErrorClass.TRANSIENT
    if any(m in msg for m in _OUTPUT_MARKERS):
        return ErrorClass.OUTPUT
    return ErrorClass.INPUT


@dataclass
class RetryPolicy:
    """Max retries (beyond the first attempt) per class, plus backoff parameters."""

    max_retries: Dict[ErrorClass, int] = field(
        default_factory=lambda: {
            ErrorClass.TRANSIENT: 3,
            ErrorClass.CONTENT: 1,
            ErrorClass.OUTPUT: 2,
            ErrorClass.INPUT: 0,
        }
    )
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: float = 0.5

    def retries_for(self, cls: ErrorClass) -> int:
        return self.max_retries.get(cls, 0)

    def backoff(self, cls: ErrorClass, attempt: int, rng: random.Random) -> float:
        if cls is not ErrorClass.TRANSIENT:
            return 0.0
        return min(self.max_delay, self.base_delay * 2 ** (attempt - 1)) + rng.uniform(0, self.jitter)

    @classmethod
    def uniform(cls, n: int) -> "RetryPolicy":
        """Same retry budget for every class (the 'just retry everything' ablation)."""
        return cls(max_retries={c: n for c in ErrorClass})

    @classmethod
    def none(cls) -> "RetryPolicy":
        return cls.uniform(0)


class CircuitOpenError(Exception):
    """Raised instead of calling a provider while its circuit is open."""


class CircuitBreaker:
    """
    Consecutive-failure breaker on TRANSIENT errors.

    closed -> open after ``threshold`` consecutive transient failures;
    open -> half-open after ``cooldown`` seconds; one probe call decides.
    While open, shots are left pending (not paid for) instead of hammering a
    provider that is down; a later resume picks them up.
    """

    def __init__(self, threshold: int = 5, cooldown: float = 30.0, clock: Callable[[], float] = time.monotonic):
        self.threshold, self.cooldown, self.clock = threshold, cooldown, clock
        self.failures = 0
        self.opened_at: Optional[float] = None

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        return "half-open" if self.clock() - self.opened_at >= self.cooldown else "open"

    def before_call(self) -> None:
        if self.state == "open":
            raise CircuitOpenError("circuit open: provider failing, call skipped")

    def record(self, cls: Optional[ErrorClass]) -> None:
        if cls is ErrorClass.TRANSIENT:
            self.failures += 1
            if self.failures >= self.threshold or self.state == "half-open":
                self.opened_at = self.clock()
        else:  # success or non-transient error: provider is reachable
            self.failures = 0
            self.opened_at = None
