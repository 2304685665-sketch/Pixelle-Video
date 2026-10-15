"""
Test and benchmark utilities for Pixelle-Video.

Nothing in this package is used by the production pipelines. It provides
deterministic fakes and fault injection so that orchestration behaviour
(failure isolation, persistence, resume) can be tested and measured without
paying for real model API calls.
"""

from pixelle_video.testing.fault_injection import (
    CorruptOutputError,
    FakeCore,
    FakeFrameProcessor,
    FaultProfile,
    InjectedFault,
    ModerationRejectedError,
    ProviderTimeoutError,
    RateLimitedError,
    ShotCall,
)

__all__ = [
    "CorruptOutputError",
    "FakeCore",
    "FakeFrameProcessor",
    "FaultProfile",
    "InjectedFault",
    "ModerationRejectedError",
    "ProviderTimeoutError",
    "RateLimitedError",
    "ShotCall",
]
