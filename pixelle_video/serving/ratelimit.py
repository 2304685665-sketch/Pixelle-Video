"""
Distributed token bucket per provider (Redis + Lua).

Video generation providers enforce per-account QPS / concurrency limits; with
several workers, a per-process limiter cannot keep the account under the limit.
The bucket state lives in Redis and is updated by one Lua script, so concurrent
workers never over-spend tokens. Time comes from the Redis server (``TIME``) so
worker clock skew does not matter.
"""

from __future__ import annotations

import asyncio

_BUCKET_LUA = """
local key = KEYS[1]
local rate = tonumber(ARGV[1])
local capacity = tonumber(ARGV[2])
local cost = tonumber(ARGV[3])
local t = redis.call('TIME')
local now = tonumber(t[1]) + tonumber(t[2]) / 1000000
local state = redis.call('HMGET', key, 'tokens', 'ts')
local tokens = tonumber(state[1]) or capacity
local ts = tonumber(state[2]) or now
tokens = math.min(capacity, tokens + (now - ts) * rate)
local wait = 0
if tokens >= cost then
  tokens = tokens - cost
else
  wait = (cost - tokens) / rate
end
redis.call('HSET', key, 'tokens', tostring(tokens), 'ts', tostring(now))
redis.call('EXPIRE', key, math.ceil(capacity / rate) + 60)
return tostring(wait)
"""


class TokenBucket:
    def __init__(self, redis, provider: str, rate: float, capacity: float, prefix: str = "pv:"):
        if rate <= 0 or capacity <= 0:
            raise ValueError("rate and capacity must be positive")
        self.key = f"{prefix}bucket:{provider}"
        self.rate, self.capacity = rate, capacity
        self._script = redis.register_script(_BUCKET_LUA)

    async def try_acquire(self, cost: float = 1.0) -> float:
        """Take tokens if available; returns 0.0 on success, else seconds to wait."""
        return float(await self._script(keys=[self.key], args=[self.rate, self.capacity, cost]))

    async def acquire(self, cost: float = 1.0) -> None:
        while True:
            wait = await self.try_acquire(cost)
            if wait <= 0:
                return
            await asyncio.sleep(wait)


class RateLimitedProcessor:
    """Wraps a frame processor so every paid shot call first takes a token."""

    def __init__(self, processor, bucket: TokenBucket):
        self.processor, self.bucket = processor, bucket

    async def __call__(self, *args, **kwargs):
        await self.bucket.acquire()
        return await self.processor(*args, **kwargs)
